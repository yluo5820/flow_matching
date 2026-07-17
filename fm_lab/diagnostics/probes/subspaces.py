"""Pure subspace directions for diagnostic probes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PrincipalDirection:
    """Top direction of an exact centered row covariance."""

    vector: torch.Tensor
    eigenvalue: float
    explained_fraction: float


@dataclass(frozen=True)
class ProjectedDirection:
    """Signed unit descent direction inside a one-dimensional subspace."""

    vector: torch.Tensor
    projection_fraction: float


def top_centered_covariance_direction(rows: torch.Tensor) -> PrincipalDirection:
    """Compute the exact top right singular direction through a sample Gram matrix."""

    if rows.ndim != 2:
        raise ValueError("Exact gradient rows must form a matrix.")
    if rows.shape[0] < 2:
        raise ValueError("Centered covariance requires at least two rows.")
    if rows.shape[1] < 1:
        raise ValueError("Exact gradient rows must have at least one parameter.")
    values = rows.detach().float().cpu()
    if not torch.isfinite(values).all():
        raise ValueError("Exact gradient rows must be finite.")
    centered = values - values.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalue = eigenvalues[-1]
    scale = torch.linalg.matrix_norm(gram)
    tolerance = torch.finfo(gram.dtype).eps * max(1, int(rows.shape[0])) * scale
    if not torch.isfinite(eigenvalue) or float(eigenvalue) <= float(tolerance):
        raise ValueError("Exact gradient rows have zero centered rank.")
    vector = centered.T @ eigenvectors[:, -1]
    vector /= torch.linalg.vector_norm(vector)
    vector = vector.contiguous()
    total = torch.trace(gram)
    return PrincipalDirection(
        vector=vector,
        eigenvalue=float(eigenvalue),
        explained_fraction=float(eigenvalue / total),
    )


def projected_descent_direction(
    subspace_direction: torch.Tensor,
    mean_gradient: torch.Tensor,
    *,
    minimum_projection_fraction: float = 1e-8,
) -> ProjectedDirection:
    """Orient a normalized rank-1 projection against a disjoint mean gradient."""

    if subspace_direction.ndim != 1 or mean_gradient.ndim != 1:
        raise ValueError("Subspace direction and mean gradient must be vectors.")
    if subspace_direction.shape != mean_gradient.shape:
        raise ValueError("Subspace direction and mean gradient must have the same shape.")
    if not 0 < minimum_projection_fraction < 1:
        raise ValueError("minimum_projection_fraction must lie in (0, 1).")
    direction = subspace_direction.detach().float().cpu()
    gradient = mean_gradient.detach().float().cpu()
    if not torch.isfinite(direction).all() or not torch.isfinite(gradient).all():
        raise ValueError("Subspace direction and mean gradient must be finite.")
    direction_norm = torch.linalg.vector_norm(direction)
    gradient_norm = torch.linalg.vector_norm(gradient)
    if float(direction_norm) == 0.0 or float(gradient_norm) == 0.0:
        raise ValueError("Subspace direction and mean gradient must be nonzero.")
    unit = direction / direction_norm
    projection = unit * torch.dot(unit, gradient)
    projection_norm = torch.linalg.vector_norm(projection)
    fraction = float(projection_norm / gradient_norm)
    if not np.isfinite(fraction) or fraction < minimum_projection_fraction:
        raise ValueError("Projected mean gradient is numerically negligible.")
    return ProjectedDirection(
        vector=(-projection / projection_norm).contiguous(),
        projection_fraction=fraction,
    )


def deterministic_random_unit_direction(
    dimension: int,
    *,
    base_seed: int,
    key: tuple[Any, ...],
) -> torch.Tensor:
    """Generate a platform-stable keyed Gaussian unit direction on CPU."""

    if int(dimension) < 1:
        raise ValueError("Random direction dimension must be positive.")
    payload = json.dumps(
        {"base_seed": int(base_seed), "key": list(key)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    seed %= 2**63 - 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vector = torch.randn(int(dimension), generator=generator, dtype=torch.float32)
    return (vector / torch.linalg.vector_norm(vector)).contiguous()
