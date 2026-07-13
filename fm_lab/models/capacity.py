"""Switchable low-rank capacity reserved for minority-class expertise."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SwitchableLowRankConv2d(nn.Conv2d):
    """Convolution with a zero-initialized low-rank weight branch."""

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
        if rank_ratio > 0:
            rank = max(int(rank_ratio * min(in_channels, out_channels)), 1)
        self.rank = int(rank)
        self.rank_ratio = float(rank_ratio)
        self.adapter_scale = float(adapter_scale)
        if self.rank > 0:
            height, width = self.kernel_size
            if height != width or self.groups != 1:
                raise ValueError("CM low-rank convolution requires square, ungrouped kernels.")
            self.adapter_a = nn.Parameter(
                self.weight.new_empty(self.rank * height, in_channels * width)
            )
            self.adapter_b = nn.Parameter(
                self.weight.new_zeros(out_channels * height, self.rank * height)
            )
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
            weight = weight + self.adapter_scale * update
        return F.conv2d(
            inputs,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
