"""Minibatch optimal-transport coupling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MinibatchOTCoupling:
    """Pair a minibatch by solving a squared-distance assignment problem."""

    max_exact_size: int = 2048
    name: str = "minibatch_ot"

    def pair(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        paired_x0, paired_x1, _ = self.pair_with_indices(x0, x1)
        return paired_x0, paired_x1

    def pair_with_indices(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x0.shape[0] != x1.shape[0]:
            raise ValueError("Minibatch OT currently requires equal source and target batch sizes.")
        if x0.shape[0] > self.max_exact_size:
            raise ValueError(
                f"Batch size {x0.shape[0]} exceeds max_exact_size={self.max_exact_size}. "
                "Lower the training batch size or raise max_exact_size explicitly."
            )

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject.
            raise RuntimeError("scipy is required for MinibatchOTCoupling.") from exc

        cost = torch.cdist(x0.detach(), x1.detach()).square().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost)
        rows = torch.as_tensor(row_indices, device=x0.device)
        cols = torch.as_tensor(col_indices, device=x1.device)
        return x0[rows], x1[cols], cols
