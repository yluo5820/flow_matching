"""Euclidean linear interpolation path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fm_lab.paths.base import expand_time


@dataclass
class LinearPath:
    name: str = "linear"

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        t_expanded = expand_time(t, x0)
        return (1.0 - t_expanded) * x0 + t_expanded * x1

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return x1 - x0
