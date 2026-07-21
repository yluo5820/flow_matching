"""Switchable low-rank capacity reserved for minority-class expertise."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class SwitchableLowRankConv2d(nn.Conv2d):
    """Convolution with the switchable LoRA branch used by ImbDiff-CM.

    The released ImbDiff-CM implementation factorizes a ``k x k`` convolution
    update as ``(out_channels * k, rank * k) @ (rank * k, in_channels * k)``
    before reshaping it back to the convolution kernel.  It also stores a
    scaling value from config but does not multiply the update by that value in
    ``forward``.  We preserve that behavior here so CM experiments differ from
    the official repo only where the continuous-flow objective requires it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        rank: int = 0,
        rank_ratio: float = 0.0,
        adapter_scale: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        if rank < 0 or rank_ratio < 0:
            raise ValueError("Low-rank convolution rank settings must be non-negative.")
        if rank > 0 and rank_ratio > 0:
            raise ValueError("Specify either rank or rank_ratio, not both.")
        if adapter_scale < 0:
            raise ValueError("adapter_scale must be non-negative.")
        height, width = self.kernel_size
        if self.groups != 1:
            raise ValueError("CM low-rank convolution requires ungrouped kernels.")
        if rank_ratio > 0:
            rank = _rank_from_ratio(
                rank_ratio,
                rows=out_channels,
                columns=in_channels,
            )
        self.rank = int(rank)
        self.rank_ratio = float(rank_ratio)
        self.adapter_scale = float(adapter_scale)
        if self.rank > 0:
            self.adapter_a = nn.Parameter(
                self.weight.new_empty(self.rank * height, in_channels * width)
            )
            self.adapter_b = nn.Parameter(
                self.weight.new_zeros(out_channels * height, self.rank * height)
            )
            # Adapter initialization must not shift the base model's global RNG
            # stream; otherwise a same-seed capacity model starts from different
            # downstream base weights than its non-capacity control.
            with torch.random.fork_rng(devices=[]):
                nn.init.kaiming_normal_(self.adapter_a, a=math.sqrt(5))
        else:
            self.register_parameter("adapter_a", None)
            self.register_parameter("adapter_b", None)

    def forward(self, inputs: torch.Tensor, *, use_adapter: bool = True) -> torch.Tensor:
        weight = self.weight
        if self.rank > 0 and use_adapter:
            assert self.adapter_a is not None
            assert self.adapter_b is not None
            update = (self.adapter_b @ self.adapter_a).reshape_as(weight)
            weight = weight + update
        return F.conv2d(
            inputs,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SwitchableLowRankLinear(nn.Linear):
    """Linear layer with a zero-initialized low-rank weight branch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 0,
        rank_ratio: float = 0.0,
        adapter_scale: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        if rank < 0 or rank_ratio < 0:
            raise ValueError("Low-rank linear rank settings must be non-negative.")
        if rank > 0 and rank_ratio > 0:
            raise ValueError("Specify either rank or rank_ratio, not both.")
        if adapter_scale < 0:
            raise ValueError("adapter_scale must be non-negative.")
        if rank_ratio > 0:
            rank = _rank_from_ratio(
                rank_ratio,
                rows=out_features,
                columns=in_features,
            )
        self.rank = int(rank)
        self.rank_ratio = float(rank_ratio)
        self.adapter_scale = float(adapter_scale)
        if self.rank > 0:
            self.adapter_a = nn.Parameter(
                self.weight.new_empty(self.rank, in_features)
            )
            self.adapter_b = nn.Parameter(
                self.weight.new_zeros(out_features, self.rank)
            )
            with torch.random.fork_rng(devices=[]):
                nn.init.kaiming_normal_(self.adapter_a, a=math.sqrt(5))
        else:
            self.register_parameter("adapter_a", None)
            self.register_parameter("adapter_b", None)

    def forward(self, inputs: torch.Tensor, *, use_adapter: bool = True) -> torch.Tensor:
        weight = self.weight
        if self.rank > 0 and use_adapter:
            assert self.adapter_a is not None
            assert self.adapter_b is not None
            weight = weight + self.adapter_scale * (self.adapter_b @ self.adapter_a)
        return F.linear(inputs, weight, self.bias)


def _rank_from_ratio(rank_ratio: float, *, rows: int, columns: int) -> int:
    return max(int(rank_ratio * min(rows, columns)), 1)


@dataclass(frozen=True)
class CapacityConfig:
    rank: int
    rank_ratio: float
    adapter_scale: float
    parts: frozenset[str]

    @classmethod
    def build(
        cls,
        *,
        rank: int,
        rank_ratio: float,
        adapter_scale: float,
        parts: Sequence[str],
    ) -> CapacityConfig:
        normalized_parts = frozenset(str(part).lower() for part in parts)
        supported = {"conditioning", "head", "down", "middle", "up", "tail"}
        invalid = normalized_parts - supported
        if invalid:
            raise ValueError(f"Unsupported capacity parts: {sorted(invalid)}")
        return cls(
            rank=int(rank),
            rank_ratio=float(rank_ratio),
            adapter_scale=float(adapter_scale),
            parts=normalized_parts,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.parts) and (self.rank > 0 or self.rank_ratio > 0)

    def conv(
        self,
        part: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
    ) -> nn.Conv2d:
        if part not in self.parts:
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )
        return SwitchableLowRankConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            rank=self.rank,
            rank_ratio=self.rank_ratio,
            adapter_scale=self.adapter_scale,
        )

    def linear(
        self,
        part: str,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
    ) -> nn.Linear:
        if part not in self.parts:
            return nn.Linear(in_features, out_features, bias=bias)
        return SwitchableLowRankLinear(
            in_features,
            out_features,
            rank=self.rank,
            rank_ratio=self.rank_ratio,
            adapter_scale=self.adapter_scale,
            bias=bias,
        )


def apply_capacity_conv(
    layer: nn.Conv2d,
    inputs: torch.Tensor,
    *,
    use_capacity: bool,
) -> torch.Tensor:
    if isinstance(layer, SwitchableLowRankConv2d):
        return layer(inputs, use_adapter=use_capacity)
    return layer(inputs)


def apply_capacity_linear(
    layer: nn.Linear,
    inputs: torch.Tensor,
    *,
    use_capacity: bool,
) -> torch.Tensor:
    if isinstance(layer, SwitchableLowRankLinear):
        return layer(inputs, use_adapter=use_capacity)
    return layer(inputs)


def use_capacity_from_context(context: object) -> bool:
    if isinstance(context, dict) and "use_capacity" in context:
        return bool(context["use_capacity"])
    return True
