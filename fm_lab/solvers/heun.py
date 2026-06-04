"""Heun predictor-corrector solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fm_lab.solvers.base import VelocityFn, stack_or_final, time_batch


@dataclass
class HeunSolver:
    name: str = "heun"

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
            t0 = time_batch(t_grid[idx], x.shape[0], x.device)
            t1 = time_batch(t_grid[idx + 1], x.shape[0], x.device)
            h = t_grid[idx + 1] - t_grid[idx]
            k1 = v_fn(x, t0)
            predictor = x + h * k1
            k2 = v_fn(predictor, t1)
            x = x + 0.5 * h * (k1 + k2)
            states.append(x)
        return stack_or_final(states, return_trajectory)
