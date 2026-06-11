"""Sampling helpers shared by experiment entry points."""

from __future__ import annotations

from typing import Any

import torch

from fm_lab.couplings.base import Coupling
from fm_lab.data.base import TargetDistribution
from fm_lab.paths.base import FlowPath
from fm_lab.sources.base import SourceDistribution


def sample_path_batch(
    *,
    source: SourceDistribution,
    target: TargetDistribution,
    coupling: Coupling,
    path: FlowPath,
    n_samples: int,
    t_value: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample paired path states, chunking when a coupling caps exact batch size."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    chunk_size = _coupling_chunk_size(coupling, n_samples)
    batches: list[dict[str, torch.Tensor]] = []
    remaining = n_samples
    while remaining > 0:
        batch_size = min(chunk_size, remaining)
        x0 = source.sample(batch_size, device=device)
        x1 = target.sample(batch_size, device=device)
        x0, x1 = coupling.pair(x0, x1)
        t = torch.full((batch_size,), t_value, device=device)
        xt = path.sample_xt(x0, x1, t)
        velocities = path.target_velocity(x0, x1, t)
        batches.append({"x0": x0, "x1": x1, "t": t, "xt": xt, "velocities": velocities})
        remaining -= batch_size

    return {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in ("x0", "x1", "t", "xt", "velocities")
    }


def _coupling_chunk_size(coupling: Coupling, n_samples: int) -> int:
    max_exact_size: Any = getattr(coupling, "max_exact_size", None)
    if max_exact_size is None:
        return n_samples
    chunk_size = int(max_exact_size)
    if chunk_size < 1:
        raise ValueError("Coupling max_exact_size must be positive.")
    return min(chunk_size, n_samples)
