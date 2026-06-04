"""Time-grid schedules for black-box solvers."""

from __future__ import annotations

import math

import torch


def make_time_grid(
    nfe: int,
    schedule: str = "uniform",
    device: torch.device | str | None = None,
    rho: float = 2.0,
) -> torch.Tensor:
    """Create a monotone grid from 0 to 1 with `nfe` solver steps."""

    if nfe < 1:
        raise ValueError("nfe must be at least 1.")
    device = torch.device("cpu" if device is None else device)
    s = torch.linspace(0.0, 1.0, nfe + 1, device=device)
    normalized = schedule.lower()
    if normalized == "uniform":
        return s
    if normalized in {"quadratic", "early"}:
        return s.pow(rho)
    if normalized in {"reverse_quadratic", "late"}:
        return 1.0 - (1.0 - s).pow(rho)
    if normalized == "cosine":
        return 0.5 * (1.0 - torch.cos(math.pi * s))
    raise ValueError(f"Unsupported time schedule: {schedule}")
