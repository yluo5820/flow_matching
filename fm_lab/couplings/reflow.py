"""Learned/reflow-generated couplings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from fm_lab.solvers.base import Solver
from fm_lab.solvers.schedules import make_time_grid


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


@dataclass
class ModelGeneratedCoupling:
    """Use a frozen teacher flow to generate source-target training pairs.

    For each source batch `x0`, the teacher model is integrated from `t=0` to
    `t=1`; the resulting endpoint replaces the sampled `x1`. This is useful
    for reflow/distillation experiments where a straight student should imitate
    an already trained baseline transport.
    """

    teacher_model: nn.Module
    solver: Solver
    nfe: int = 64
    schedule: str = "uniform"
    name: str = "model_generated"
    _device: torch.device | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.nfe < 1:
            raise ValueError("ModelGeneratedCoupling nfe must be at least 1.")
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def pair(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del x1, kwargs
        teacher_dim = getattr(self.teacher_model, "dim", x0.shape[1])
        if x0.shape[1] != teacher_dim:
            raise ValueError(
                f"Source dimension {x0.shape[1]} does not match teacher dimension {teacher_dim}."
            )
        self._move_teacher(x0.device)
        source_label = x0.detach()
        t_grid = make_time_grid(self.nfe, schedule=self.schedule, device=x0.device)

        def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if bool(getattr(self.teacher_model, "requires_source_label", False)):
                return self.teacher_model(x, t, context={"source_label": source_label})
            return self.teacher_model(x, t)

        generated = self.solver.solve(
            v_fn,
            x0.detach().clone(),
            t_grid,
            return_trajectory=False,
        )
        return x0, generated.detach()

    def _move_teacher(self, device: torch.device) -> None:
        if self._device != device:
            self.teacher_model.to(device)
            self._device = device
        self.teacher_model.eval()
