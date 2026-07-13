"""Paper-scale class-conditional U-Net used by discrete DDPM experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from fm_lab.models.mlp import _class_labels_from_context, _embedding_labels


class DDPMUNet(nn.Module):
    """Improved-DDPM style U-Net with timestep and classifier-free conditioning."""

    def __init__(
        self,
        *,
        dim: int,
        image_shape: Sequence[int] = (3, 32, 32),
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 2, 2, 2),
        attention_levels: Sequence[int] = (1,),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int,
        num_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        if len(image_shape) != 3:
            raise ValueError("DDPMUNet image_shape must be [channels, height, width].")
        channels, height, width = (int(value) for value in image_shape)
        if dim != channels * height * width:
            raise ValueError(f"DDPMUNet dim={dim} does not match image_shape={tuple(image_shape)}.")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive.")
        self.dim = dim
        self.image_shape = (channels, height, width)
        self.num_classes = int(num_classes)
        self.num_timesteps = int(num_timesteps)
        self.is_class_conditional = True

        time_channels = 4 * base_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.class_embedding = nn.Embedding(self.num_classes + 1, time_channels)
        self.input_conv = nn.Conv2d(channels, base_channels, 3, padding=1)

        attention = set(int(level) for level in attention_levels)
        self.down_blocks = nn.ModuleList()
        skip_channels = [base_channels]
        current = base_channels
        for level, multiplier in enumerate(channel_multipliers):
            output = base_channels * int(multiplier)
            for _ in range(num_res_blocks):
                block = _ConditionedBlock(
                    current, output, time_channels, dropout, level in attention
                )
                self.down_blocks.append(block)
                current = output
                skip_channels.append(current)
            if level != len(channel_multipliers) - 1:
                self.down_blocks.append(_Downsample(current))
                skip_channels.append(current)

        self.middle = nn.ModuleList(
            [
                _ResBlock(current, current, time_channels, dropout),
                _AttentionBlock(current),
                _ResBlock(current, current, time_channels, dropout),
            ]
        )
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(channel_multipliers))):
            output = base_channels * int(channel_multipliers[level])
            for index in range(num_res_blocks + 1):
                skip = skip_channels.pop()
                block = _ConditionedBlock(
                    current + skip, output, time_channels, dropout, level in attention
                )
                self.up_blocks.append(block)
                current = output
                if level > 0 and index == num_res_blocks:
                    self.up_blocks.append(_Upsample(current))
        self.output_norm = nn.GroupNorm(32, current)
        self.output_conv = nn.Conv2d(current, channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        batch = x.shape[0]
        image = x.reshape(batch, *self.image_shape)
        embedding = self.time_mlp(_timestep_embedding(t, self.time_mlp[0].in_features))
        labels = _class_labels_from_context(context, batch, x.device)
        embedding = embedding + self.class_embedding(_embedding_labels(labels, self.num_classes))

        h = self.input_conv(image)
        skips = [h]
        for block in self.down_blocks:
            h = block(h, embedding) if isinstance(block, _ConditionedBlock) else block(h)
            skips.append(h)
        for block in self.middle:
            h = block(h, embedding) if isinstance(block, _ResBlock) else block(h)
        for block in self.up_blocks:
            if isinstance(block, _ConditionedBlock):
                h = block(torch.cat((h, skips.pop()), dim=1), embedding)
            else:
                h = block(h)
        return self.output_conv(F.silu(self.output_norm(h))).reshape(batch, self.dim)


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = t.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class _ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb = nn.Linear(emb_channels, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(embedding))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return self.skip(x) + h


class _AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv(self.norm(x).reshape(batch, channels, -1)).chunk(3, dim=1)
        scale = channels**-0.25
        weights = torch.einsum("bct,bcs->bts", q * scale, k * scale).softmax(dim=-1)
        attended = torch.einsum("bts,bcs->bct", weights, v)
        return x + self.proj(attended).reshape(batch, channels, height, width)


class _ConditionedBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dropout: float,
        attention: bool,
    ):
        super().__init__()
        self.residual = _ResBlock(in_channels, out_channels, emb_channels, dropout)
        self.attention = _AttentionBlock(out_channels) if attention else nn.Identity()

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return self.attention(self.residual(x, embedding))


class _Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
