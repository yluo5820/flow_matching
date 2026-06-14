"""Vector-space U-Net style velocity models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from fm_lab.models.mlp import SinusoidalTimeEmbedding, _activation, _source_label_from_context


class VectorUNetVelocity(nn.Module):
    """Small 1D U-Net style velocity model for low-dimensional vectors."""

    def __init__(
        self,
        dim: int,
        base_channels: int = 64,
        time_embedding_dim: int = 64,
        activation: str = "silu",
        zero_init_head: bool = True,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("VectorUNetVelocity dim must be positive.")
        if base_channels < 1:
            raise ValueError("VectorUNetVelocity base_channels must be positive.")
        self.dim = dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            _activation(activation),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        c0 = int(base_channels)
        c1 = 2 * c0
        c2 = 4 * c0
        self.input_block = TimeResBlock1D(1, c0, time_embedding_dim, activation)
        self.down1 = nn.Conv1d(c0, c1, kernel_size=3, stride=2, padding=1)
        self.down1_block = TimeResBlock1D(c1, c1, time_embedding_dim, activation)
        self.down2 = nn.Conv1d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.down2_block = TimeResBlock1D(c2, c2, time_embedding_dim, activation)
        self.middle = TimeResBlock1D(c2, c2, time_embedding_dim, activation)
        self.up1_block = TimeResBlock1D(c2 + c1, c1, time_embedding_dim, activation)
        self.up0_block = TimeResBlock1D(c1 + c0, c0, time_embedding_dim, activation)
        self.output_block = nn.Sequential(
            nn.GroupNorm(_group_count(c0), c0),
            _activation(activation),
            nn.Conv1d(c0, 1, kernel_size=3, padding=1),
        )
        if zero_init_head:
            nn.init.zeros_(self.output_block[-1].weight)
            nn.init.zeros_(self.output_block[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del context
        sequence = x[:, None, :]
        time_features = self.time_mlp(self.time_embedding(t))

        h0 = self.input_block(sequence, time_features)
        h1 = self.down1_block(self.down1(h0), time_features)
        h2 = self.down2_block(self.down2(h1), time_features)
        middle = self.middle(h2, time_features)

        u1 = F.interpolate(middle, size=h1.shape[-1], mode="nearest")
        u1 = self.up1_block(torch.cat([u1, h1], dim=1), time_features)
        u0 = F.interpolate(u1, size=h0.shape[-1], mode="nearest")
        u0 = self.up0_block(torch.cat([u0, h0], dim=1), time_features)
        return self.output_block(u0).squeeze(1)


class DirectionSpeedVectorUNet(nn.Module):
    """Direction-only straight-flow model with vector U-Net style backbones."""

    requires_source_label = True

    def __init__(
        self,
        dim: int,
        base_channels: int = 64,
        time_embedding_dim: int = 64,
        activation: str = "silu",
        direction_eps: float = 1e-8,
        direction_zero_init_head: bool = False,
        speed_zero_init_head: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.direction_eps = direction_eps
        self.direction_net = VectorUNetVelocity(
            dim=dim,
            base_channels=base_channels,
            time_embedding_dim=time_embedding_dim,
            activation=activation,
            zero_init_head=direction_zero_init_head,
        )
        self.speed_net = VectorPairScalarUNet(
            dim=dim,
            base_channels=base_channels,
            time_embedding_dim=time_embedding_dim,
            activation=activation,
            zero_init_head=speed_zero_init_head,
        )

    def direction(self, source_label: torch.Tensor) -> torch.Tensor:
        t0 = torch.zeros(
            source_label.shape[0],
            device=source_label.device,
            dtype=source_label.dtype,
        )
        raw_direction = self.direction_net(source_label, t0)
        norm = raw_direction.norm(dim=1, keepdim=True)
        return raw_direction / (norm + self.direction_eps)

    def speed(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        source_label: torch.Tensor,
    ) -> torch.Tensor:
        return self.speed_net(x=x, t=t, source_label=source_label)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = _source_label_from_context(context)
        direction = self.direction(source_label)
        speed = self.speed(x, t, source_label)
        return speed[:, None] * direction


class VectorPairScalarUNet(nn.Module):
    """1D U-Net style scalar predictor from `(x, source_label, t)`."""

    def __init__(
        self,
        dim: int,
        base_channels: int,
        time_embedding_dim: int,
        activation: str,
        zero_init_head: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            _activation(activation),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        c0 = int(base_channels)
        c1 = 2 * c0
        c2 = 4 * c0
        self.input_block = TimeResBlock1D(2, c0, time_embedding_dim, activation)
        self.down1 = nn.Conv1d(c0, c1, kernel_size=3, stride=2, padding=1)
        self.down1_block = TimeResBlock1D(c1, c1, time_embedding_dim, activation)
        self.down2 = nn.Conv1d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.down2_block = TimeResBlock1D(c2, c2, time_embedding_dim, activation)
        self.middle = TimeResBlock1D(c2, c2, time_embedding_dim, activation)
        self.up1_block = TimeResBlock1D(c2 + c1, c1, time_embedding_dim, activation)
        self.up0_block = TimeResBlock1D(c1 + c0, c0, time_embedding_dim, activation)
        self.output_head = nn.Sequential(
            nn.GroupNorm(_group_count(c0), c0),
            _activation(activation),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(c0, 1),
        )
        if zero_init_head:
            nn.init.zeros_(self.output_head[-1].weight)
            nn.init.zeros_(self.output_head[-1].bias)

    def forward(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        source_label: torch.Tensor,
    ) -> torch.Tensor:
        sequence = torch.stack((x, source_label), dim=1)
        time_features = self.time_mlp(self.time_embedding(t))

        h0 = self.input_block(sequence, time_features)
        h1 = self.down1_block(self.down1(h0), time_features)
        h2 = self.down2_block(self.down2(h1), time_features)
        middle = self.middle(h2, time_features)

        u1 = F.interpolate(middle, size=h1.shape[-1], mode="nearest")
        u1 = self.up1_block(torch.cat([u1, h1], dim=1), time_features)
        u0 = F.interpolate(u1, size=h0.shape[-1], mode="nearest")
        u0 = self.up0_block(torch.cat([u0, h0], dim=1), time_features)
        return self.output_head(u0).squeeze(-1)


class TimeResBlock1D(nn.Module):
    """Residual 1D conv block with additive time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.time_proj(time_features)[:, :, None]
        h = self.conv2(self.activation(self.norm2(h)))
        return (h + self.skip(x)) / math.sqrt(2.0)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
