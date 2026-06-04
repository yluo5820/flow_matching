"""Explicit Euler solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fm_lab.solvers.base import VelocityFn, stack_or_final, time_batch


@dataclass
class EulerSolver:
    name: str = "euler"

    def solve(
        self,
        v_fn: VelocityFn,
        x0: torch.Tensor,
        t_grid: torch.Tensor,
        return_trajectory: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        x = x0
        states = [x]
        for idx in range(len(t_grid) - 1):
            t = time_batch(t_grid[idx], x.shape[0], x.device)
            h = t_grid[idx + 1] - t_grid[idx]
            x = x + h * v_fn(x, t)
            states.append(x)
        return stack_or_final(states, return_trajectory)
