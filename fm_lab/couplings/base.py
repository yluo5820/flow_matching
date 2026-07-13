"""Base interfaces for source-target couplings."""

from __future__ import annotations

from typing import Protocol

import torch


def pair_with_condition(
    coupling: Coupling,
    x0: torch.Tensor,
    x1: torch.Tensor,
    condition: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Pair a batch while keeping target-side conditions aligned."""

    if condition is None:
        paired_x0, paired_x1 = coupling.pair(x0, x1)
        return paired_x0, paired_x1, None
    pair_with_indices = getattr(coupling, "pair_with_indices", None)
    if not callable(pair_with_indices):
        raise ValueError(
            f"Coupling {getattr(coupling, 'name', coupling.__class__.__name__)} "
            "does not support condition-preserving pairing."
        )
    paired_x0, paired_x1, target_indices = pair_with_indices(x0, x1)
    return paired_x0, paired_x1, condition[target_indices]


class Coupling(Protocol):
    """Pair source and target batches."""

    name: str

    def pair(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return paired source and target batches."""
