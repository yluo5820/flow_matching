"""Black-box ODE solver interface."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Solver(Protocol):
    """Numerical solver for `dx/dt = v(x, t)`."""

    name: str

    def solve(
        self,
        v_fn: VelocityFn,
        x0: torch.Tensor,
        t_grid: torch.Tensor,
        return_trajectory: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Integrate over `t_grid` and return the final state or full trajectory."""


def time_batch(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Convert scalar time to a batch-shaped tensor."""

    return torch.full((batch_size,), float(t.detach().cpu()), device=device)


def stack_or_final(states: list[torch.Tensor], return_trajectory: bool) -> torch.Tensor:
    """Return a stacked trajectory or its final state."""

    if return_trajectory:
        return torch.stack(states, dim=0)
    return states[-1]
