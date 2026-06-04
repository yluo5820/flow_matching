"""SciPy RK45/Dormand-Prince wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fm_lab.solvers.base import VelocityFn, stack_or_final, time_batch


@dataclass
class ScipyDopri5Solver:
    """Dormand-Prince RK45 wrapper around `scipy.integrate.solve_ivp`."""

    rtol: float = 1e-5
    atol: float = 1e-6
    name: str = "dopri5"

    def solve(
        self,
        v_fn: VelocityFn,
        x0: torch.Tensor,
        t_grid: torch.Tensor,
        return_trajectory: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        try:
            from scipy.integrate import solve_ivp
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject.
            raise RuntimeError("scipy is required for ScipyDopri5Solver.") from exc

        shape = x0.shape
        device = x0.device
        dtype = x0.dtype
        t_eval = t_grid.detach().cpu().numpy() if return_trajectory else None

        def rhs(t_scalar, y_flat):
            x = torch.from_numpy(y_flat).to(device=device, dtype=dtype).reshape(shape)
            t = time_batch(torch.tensor(t_scalar), shape[0], device)
            with torch.no_grad():
                velocity = v_fn(x, t)
            return velocity.detach().cpu().numpy().reshape(-1)

        solution = solve_ivp(
            rhs,
            (float(t_grid[0].detach().cpu()), float(t_grid[-1].detach().cpu())),
            x0.detach().cpu().numpy().reshape(-1),
            method="RK45",
            t_eval=t_eval,
            rtol=float(kwargs.get("rtol", self.rtol)),
            atol=float(kwargs.get("atol", self.atol)),
        )
        if not solution.success:
            raise RuntimeError(f"SciPy RK45 solver failed: {solution.message}")

        if return_trajectory:
            states = [
                torch.from_numpy(solution.y[:, idx]).to(device=device, dtype=dtype).reshape(shape)
                for idx in range(solution.y.shape[1])
            ]
            return stack_or_final(states, return_trajectory=True)
        return torch.from_numpy(solution.y[:, -1]).to(device=device, dtype=dtype).reshape(shape)
