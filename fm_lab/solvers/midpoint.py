"""Explicit midpoint solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fm_lab.solvers.base import VelocityFn, stack_or_final, time_batch


@dataclass
class MidpointSolver:
    name: str = "midpoint"

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
            h = t_grid[idx + 1] - t_grid[idx]
            t0 = time_batch(t_grid[idx], x.shape[0], x.device)
            t_mid = time_batch(t_grid[idx] + 0.5 * h, x.shape[0], x.device)
            k1 = v_fn(x, t0)
            x = x + h * v_fn(x + 0.5 * h * k1, t_mid)
            states.append(x)
        return stack_or_final(states, return_trajectory)
