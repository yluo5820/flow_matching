"""Image-space velocity models."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from fm_lab.models.mlp import (
    SinusoidalTimeEmbedding,
    _activation,
    _class_labels_from_context,
    _embedding_labels,
    _source_label_from_context,
)


class ImageUNetVelocity(nn.Module):
    """Small time-conditioned U-Net velocity model for flattened images."""

    def __init__(
        self,
        dim: int,
        image_shape: tuple[int, ...] = (28, 28),
        base_channels: int = 32,
        time_embedding_dim: int = 128,
        activation: str = "silu",
        zero_init_head: bool = True,
        num_classes: int | None = None,
        class_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        channels, height, width, layout = _parse_image_shape(image_shape)
        if dim != channels * height * width:
            raise ValueError(
                f"ImageUNetVelocity dim={dim} does not match image_shape={image_shape}."
            )
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError("ImageUNetVelocity requires image dimensions divisible by 4.")
        self.dim = dim
        self.channels = channels
        self.height = height
        self.width = width
        self.image_shape = tuple(int(value) for value in image_shape)
        self.image_layout = layout
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.num_classes = num_classes
        self.is_class_conditional = num_classes is not None
        condition_dim = int(class_embedding_dim or time_embedding_dim)
        self.class_embedding = (
            nn.Embedding(int(num_classes) + 1, condition_dim)
            if num_classes is not None
            else None
        )
        self.class_projection = (
            nn.Linear(condition_dim, time_embedding_dim)
            if num_classes is not None
            else None
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            _activation(activation),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        c0 = int(base_channels)
        c1 = 2 * c0
        c2 = 4 * c0
        self.input_block = TimeResBlock(channels, c0, time_embedding_dim, activation)
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
            nn.Conv2d(c0, channels, kernel_size=3, padding=1),
        )
        if zero_init_head:
            nn.init.zeros_(self.output_block[-1].weight)
            nn.init.zeros_(self.output_block[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        image = _flat_to_image(
            x,
            channels=self.channels,
            height=self.height,
            width=self.width,
            layout=self.image_layout,
        )
        time_features = self.time_mlp(self.time_embedding(t))
        if self.class_embedding is not None and self.class_projection is not None:
            labels = _class_labels_from_context(context, x.shape[0], x.device)
            class_features = self.class_embedding(_embedding_labels(labels, self.num_classes))
            time_features = time_features + self.class_projection(class_features)

        h0 = self.input_block(image, time_features)
        h1 = self.down1_block(self.down1(h0), time_features)
        h2 = self.down2_block(self.down2(h1), time_features)
        middle = self.middle(h2, time_features)

        u1 = F.interpolate(middle, size=h1.shape[-2:], mode="nearest")
        u1 = self.up1_block(torch.cat([u1, h1], dim=1), time_features)
        u0 = F.interpolate(u1, size=h0.shape[-2:], mode="nearest")
        u0 = self.up0_block(torch.cat([u0, h0], dim=1), time_features)
        return _image_to_flat(self.output_block(u0), layout=self.image_layout)


class DirectionSpeedImageUNet(nn.Module):
    """Direction-only straight-flow model with image U-Net backbones."""

    requires_source_label = True

    def __init__(
        self,
        dim: int,
        image_shape: tuple[int, ...] = (28, 28),
        base_channels: int = 32,
        time_embedding_dim: int = 128,
        activation: str = "silu",
        direction_eps: float = 1e-8,
        direction_zero_init_head: bool = False,
        speed_zero_init_head: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.direction_eps = direction_eps
        self.direction_net = ImageUNetVelocity(
            dim=dim,
            image_shape=image_shape,
            base_channels=base_channels,
            time_embedding_dim=time_embedding_dim,
            activation=activation,
            zero_init_head=direction_zero_init_head,
        )
        self.speed_net = ImagePairScalarUNet(
            dim=dim,
            image_shape=image_shape,
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


class ImagePairScalarUNet(nn.Module):
    """Image U-Net scalar predictor from `(x, source_label, t)`."""

    def __init__(
        self,
        dim: int,
        image_shape: tuple[int, ...],
        base_channels: int,
        time_embedding_dim: int,
        activation: str,
        zero_init_head: bool,
    ) -> None:
        super().__init__()
        channels, height, width, layout = _parse_image_shape(image_shape)
        if dim != channels * height * width:
            raise ValueError(
                f"ImagePairScalarUNet dim={dim} does not match image_shape={image_shape}."
            )
        if height % 4 != 0 or width % 4 != 0:
            raise ValueError("ImagePairScalarUNet requires image dimensions divisible by 4.")
        self.dim = dim
        self.channels = channels
        self.height = height
        self.width = width
        self.image_shape = tuple(int(value) for value in image_shape)
        self.image_layout = layout
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            _activation(activation),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

        c0 = int(base_channels)
        c1 = 2 * c0
        c2 = 4 * c0
        self.input_block = TimeResBlock(2 * channels, c0, time_embedding_dim, activation)
        self.down1 = nn.Sequential(nn.Conv2d(c0, c1, kernel_size=3, stride=2, padding=1))
        self.down1_block = TimeResBlock(c1, c1, time_embedding_dim, activation)
        self.down2 = nn.Sequential(nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1))
        self.down2_block = TimeResBlock(c2, c2, time_embedding_dim, activation)
        self.middle = TimeResBlock(c2, c2, time_embedding_dim, activation)
        self.up1_block = TimeResBlock(c2 + c1, c1, time_embedding_dim, activation)
        self.up0_block = TimeResBlock(c1 + c0, c0, time_embedding_dim, activation)
        self.output_head = nn.Sequential(
            nn.GroupNorm(_group_count(c0), c0),
            _activation(activation),
            nn.AdaptiveAvgPool2d((1, 1)),
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
        image = torch.cat(
            (
                _flat_to_image(
                    x,
                    channels=self.channels,
                    height=self.height,
                    width=self.width,
                    layout=self.image_layout,
                ),
                _flat_to_image(
                    source_label,
                    channels=self.channels,
                    height=self.height,
                    width=self.width,
                    layout=self.image_layout,
                ),
            ),
            dim=1,
        )
        time_features = self.time_mlp(self.time_embedding(t))

        h0 = self.input_block(image, time_features)
        h1 = self.down1_block(self.down1(h0), time_features)
        h2 = self.down2_block(self.down2(h1), time_features)
        middle = self.middle(h2, time_features)

        u1 = F.interpolate(middle, size=h1.shape[-2:], mode="nearest")
        u1 = self.up1_block(torch.cat([u1, h1], dim=1), time_features)
        u0 = F.interpolate(u1, size=h0.shape[-2:], mode="nearest")
        u0 = self.up0_block(torch.cat([u0, h0], dim=1), time_features)
        return self.output_head(u0).squeeze(-1)


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


def _parse_image_shape(image_shape: tuple[int, ...]) -> tuple[int, int, int, str]:
    shape = tuple(int(value) for value in image_shape)
    if len(shape) == 2:
        height, width = shape
        return 1, height, width, "hw"
    if len(shape) == 3 and shape[-1] in {1, 3, 4}:
        height, width, channels = shape
        return channels, height, width, "hwc"
    if len(shape) == 3 and shape[0] in {1, 3, 4}:
        channels, height, width = shape
        return channels, height, width, "chw"
    raise ValueError(
        "image_shape must be [height, width], [height, width, channels], "
        f"or [channels, height, width], got {shape}."
    )


def _flat_to_image(
    x: torch.Tensor,
    *,
    channels: int,
    height: int,
    width: int,
    layout: str,
) -> torch.Tensor:
    batch_size = x.shape[0]
    if layout == "hw":
        return x.reshape(batch_size, 1, height, width)
    if layout == "hwc":
        return x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    if layout == "chw":
        return x.reshape(batch_size, channels, height, width)
    raise ValueError(f"Unsupported image layout: {layout}")


def _image_to_flat(image: torch.Tensor, *, layout: str) -> torch.Tensor:
    if layout == "hw":
        return image.reshape(image.shape[0], -1)
    if layout == "hwc":
        return image.permute(0, 2, 3, 1).reshape(image.shape[0], -1)
    if layout == "chw":
        return image.reshape(image.shape[0], -1)
    raise ValueError(f"Unsupported image layout: {layout}")
