"""Base interfaces for source-target couplings."""

from __future__ import annotations

from typing import Protocol

import torch


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
