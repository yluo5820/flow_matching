"""Classical fourth-order Runge-Kutta solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fm_lab.solvers.base import VelocityFn, stack_or_final, time_batch


@dataclass
class RK4Solver:
    name: str = "rk4"

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
            t1 = time_batch(t_grid[idx + 1], x.shape[0], x.device)
            k1 = v_fn(x, t0)
            k2 = v_fn(x + 0.5 * h * k1, t_mid)
            k3 = v_fn(x + 0.5 * h * k2, t_mid)
            k4 = v_fn(x + h * k3, t1)
            x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            states.append(x)
        return stack_or_final(states, return_trajectory)
