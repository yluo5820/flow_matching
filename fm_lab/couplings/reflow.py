"""Placeholder for reflow-generated couplings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ReflowCouplingPlaceholder:
    """Explicit placeholder for a learned/reflow-generated coupling.

    A real implementation needs a trained map or trajectory cache. Keeping this
    as a concrete component makes configs fail loudly instead of silently falling
    back to independent coupling.
    """

    checkpoint_path: str | Path | None = None
    name: str = "reflow_placeholder"

    def pair(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Reflow-generated coupling is a placeholder for later milestones. "
            "Provide a learned map or cached trajectories before using it in training."
        )
