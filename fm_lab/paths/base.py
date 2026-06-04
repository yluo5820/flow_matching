"""Base interfaces for probability paths."""

from __future__ import annotations

from typing import Protocol

import torch


class FlowPath(Protocol):
    """Path sampler and target-velocity provider."""

    name: str

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Sample intermediate points at time `t`."""

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return target velocity at time `t`."""


def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand a batch of scalar times to broadcast against `x`."""

    if t.ndim == 0:
        t = t[None]
    if t.ndim == 1:
        t = t[:, None]
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t
