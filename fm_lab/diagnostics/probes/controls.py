"""Negative permutation controls and planted low-rank positive controls."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def projection_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return normalized overlap between two column subspaces."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Projection-overlap bases must be aligned matrices.")
    if left.shape[1] < 1 or right.shape[1] < 1:
        raise ValueError("Projection-overlap bases must be non-empty.")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Projection-overlap bases must be finite.")
    left_q, _ = torch.linalg.qr(left.double(), mode="reduced")
    right_q, _ = torch.linalg.qr(right.double(), mode="reduced")
    rank = min(left_q.shape[1], right_q.shape[1])
    overlap = torch.linalg.matrix_norm(left_q.T @ right_q, ord="fro").square()
    return float((overlap / rank).clamp(0.0, 1.0))


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    null_values: np.ndarray


def permutation_null(
    values: Any,
    labels: np.ndarray,
    *,
    statistic: Callable[[Any, np.ndarray], float],
    permutations: int,
    seed: int,
) -> PermutationResult:
    """Evaluate a statistic against exact-multiplicity label permutations."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(values):
        raise ValueError("Permutation labels must align with the value rows.")
    if permutations < 1:
        raise ValueError("permutations must be positive.")
    observed = float(statistic(values, labels.copy()))
    if not math.isfinite(observed):
        raise ValueError("Observed permutation statistic must be finite.")
    rng = np.random.RandomState(seed)
    null_values = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        permuted = rng.permutation(labels)
        null_values[index] = float(statistic(values, permuted))
    if not np.all(np.isfinite(null_values)):
        raise ValueError("Permutation null statistics must be finite.")
    exceedances = int(np.sum(null_values >= observed))
    null_values.setflags(write=False)
    return PermutationResult(
        observed=observed,
        p_value=(1.0 + exceedances) / (1.0 + permutations),
        null_values=null_values,
    )


@dataclass(frozen=True)
class PlantedControlResult:
    planted_rank: int
    recovered_rank: int
    subspace_overlap: float
    eigenvalues: torch.Tensor


def planted_low_rank_control(
    *,
    ambient_dim: int,
    rank: int,
    rows: int,
    noise_std: float,
    seed: int,
) -> PlantedControlResult:
    """Plant and recover a known parameter subspace using a sample-space Gram matrix."""

    if rank < 1 or rank > ambient_dim or rank >= rows:
        raise ValueError("Planted rank must be positive and below ambient_dim and rows.")
    if ambient_dim < 2 or rows < 2:
        raise ValueError("Planted control dimensions must be at least two.")
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("Planted control noise_std must be finite and non-negative.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    planted_basis, _ = torch.linalg.qr(
        torch.randn(ambient_dim, rank, generator=generator, dtype=torch.float64),
        mode="reduced",
    )
    coefficient_basis, _ = torch.linalg.qr(
        torch.randn(rows, rank, generator=generator, dtype=torch.float64),
        mode="reduced",
    )
    samples = 10.0 * coefficient_basis @ planted_basis.T
    samples = samples + noise_std * torch.randn(
        rows,
        ambient_dim,
        generator=generator,
        dtype=torch.float64,
    )
    gram = samples @ samples.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    max_rank = min(32, rows - 1, ambient_dim)
    gaps = eigenvalues[:max_rank] - eigenvalues[1 : max_rank + 1]
    recovered_rank = int(torch.argmax(gaps)) + 1
    singular_values = eigenvalues[:recovered_rank].sqrt().clamp_min(1.0e-12)
    recovered_basis = (samples.T @ eigenvectors[:, :recovered_rank]) / singular_values[None, :]
    overlap = projection_overlap(
        planted_basis,
        recovered_basis[:, :rank],
    )
    return PlantedControlResult(
        planted_rank=rank,
        recovered_rank=recovered_rank,
        subspace_overlap=overlap,
        eigenvalues=eigenvalues,
    )
