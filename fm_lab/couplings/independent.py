"""Independent source-target coupling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class IndependentCoupling:
    shuffle_target: bool = True
    name: str = "independent"

    def pair(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.shuffle_target:
            return x0, x1
        return x0, x1[torch.randperm(x1.shape[0], device=x1.device)]

    def pair_with_indices(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.arange(x1.shape[0], device=x1.device)
        if self.shuffle_target:
            indices = torch.randperm(x1.shape[0], device=x1.device)
        return x0, x1[indices], indices
