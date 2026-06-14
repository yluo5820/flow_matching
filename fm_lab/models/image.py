"""Image-space velocity models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from fm_lab.models.mlp import SinusoidalTimeEmbedding, _activation


class ImageUNetVelocity(nn.Module):
    """Small time-conditioned U-Net velocity model for flattened grayscale images."""

    def __init__(
        self,
        dim: int,
        image_shape: tuple[int, int] = (28, 28),
        base_channels: int = 32,
        time_embedding_dim: int = 128,
        activation: str = "silu",
        zero_init_head: bool = True,
    ) -> None:
        super().__init__()
        height, width = image_shape
        if dim != height * width:
            raise ValueError(
                f"ImageUNetVelocity dim={dim} does not match image_shape={image_shape}."
            )
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError("ImageUNetVelocity requires image dimensions divisible by 4.")
        self.dim = dim
        self.image_shape = (height, width)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            _activation(activation),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        c0 = int(base_channels)
        c1 = 2 * c0
        c2 = 4 * c0
        self.input_block = TimeResBlock(1, c0, time_embedding_dim, activation)
        self.down1 = nn.Sequential(nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1))
        self.down1_block = TimeResBlock(c1, c1, time_embedding_dim, activation)
        self.down2 = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1))
        self.down2_block = TimeResBlock(c2, c2, time_embedding_dim, activation)
        self.middle = TimeResBlock(c2, c2, time_embedding_dim, activation)
        self.up1_block = TimeResBlock(c2 + c1, c1, time_embedding_dim, activation)
        self.up0_block = TimeResBlock(c1 + c0, c0, time_embedding_dim, activation)
        self.output_block = nn.Sequential(
            nn.GroupNorm(_group_count(c0), c0),
            _activation(activation),
            nn.Conv2d(c0, 1, kernel_size=3, padding=1),
        )
        if zero_init_head:
            nn.init.zeros_(self.output_block[-1].weight)
            nn.init.zeros_(self.output_block[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del context
        batch_size = x.shape[0]
        image = x.reshape(batch_size, 1, *self.image_shape)
        time_features = self.time_mlp(self.time_embedding(t))

        h0 = self.input_block(image, time_features)
        h1 = self.down1_block(self.down1(h0), time_features)
        h2 = self.down2_block(self.down2(h1), time_features)
        middle = self.middle(h2, time_features)

        u1 = F.interpolate(middle, size=h1.shape[-2:], mode="nearest")
        u1 = self.up1_block(torch.cat([u1, h1], dim=1), time_features)
        u0 = F.interpolate(u1, size=h0.shape[-2:], mode="nearest")
        u0 = self.up0_block(torch.cat([u0, h0], dim=1), time_features)
        return self.output_block(u0).reshape(batch_size, self.dim)


class TimeResBlock(nn.Module):
    """Residual conv block with additive time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embedding_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.time_proj(time_features)[:, :, None, None]
        h = self.conv2(self.activation(self.norm2(h)))
        return (h + self.skip(x)) / math.sqrt(2.0)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
