"""Exact directions and local functional tests for Observation 0."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.gradients import resolve_probe_layers
from fm_lab.diagnostics.long_tail_geometry.manifest import ProbeManifest


@dataclass(frozen=True)
class Rank1Direction:
    """Top direction of an exact centered microbatch-gradient covariance."""

    vector: torch.Tensor
    eigenvalue: float
    explained_fraction: float


@dataclass(frozen=True)
class ProjectedDescentDirection:
    """Signed unit descent direction inside a rank-1 subspace."""

    vector: torch.Tensor
    projection_fraction: float


def cell_microbatch_rows(
    manifest: ProbeManifest,
    *,
    class_id: int,
    stratum_id: int,
) -> tuple[np.ndarray, ...]:
    """Return one cell's microbatch row arrays in stable manifest order."""

    selected: list[np.ndarray] = []
    for rows in manifest.microbatch_row_indices():
        labels = np.unique(manifest.labels[rows])
        strata = np.unique(manifest.stratum_ids[rows])
        if len(labels) != 1 or len(strata) != 1:
            raise ValueError("Probe manifest contains a mixed class/stratum microbatch.")
        if int(labels[0]) == int(class_id) and int(strata[0]) == int(stratum_id):
            selected.append(np.asarray(rows, dtype=np.int64))
    if not selected:
        raise ValueError(
            f"Probe manifest has no microbatches for class {class_id}, "
            f"stratum {stratum_id}."
        )
    return tuple(selected)


def top_centered_covariance_direction(rows: torch.Tensor) -> Rank1Direction:
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
    return Rank1Direction(
        vector=vector,
        eigenvalue=float(eigenvalue),
        explained_fraction=float(eigenvalue / total),
    )


def projected_descent_direction(
    subspace_direction: torch.Tensor,
    mean_gradient: torch.Tensor,
    *,
    minimum_projection_fraction: float = 1e-8,
) -> ProjectedDescentDirection:
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
    return ProjectedDescentDirection(
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


@contextmanager
def virtual_layer_update(
    model: nn.Module,
    *,
    layer_name: str,
    direction: torch.Tensor,
    relative_step: float,
) -> Iterator[float]:
    """Apply one relative layerwise update and restore the parameter bit-exactly."""

    if not np.isfinite(relative_step) or float(relative_step) <= 0:
        raise ValueError("Virtual-update relative_step must be positive and finite.")
    layer = resolve_probe_layers(model, (layer_name,))[0]
    flat_direction = direction.detach().reshape(-1).float().cpu()
    if flat_direction.numel() != layer.parameter.numel():
        raise ValueError("Virtual-update direction has the wrong shape.")
    if not torch.isfinite(flat_direction).all():
        raise ValueError("Virtual-update direction must be finite.")
    direction_norm = torch.linalg.vector_norm(flat_direction)
    if not torch.isclose(
        direction_norm,
        torch.ones_like(direction_norm),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Virtual-update direction must be unit norm.")
    original = layer.parameter.detach().clone()
    parameter_norm = torch.linalg.vector_norm(original)
    if not torch.isfinite(parameter_norm) or float(parameter_norm) == 0.0:
        raise ValueError("Virtual-update layer must have a finite nonzero norm.")
    applied_norm = float(relative_step) * float(parameter_norm)
    update = flat_direction.to(
        device=layer.parameter.device,
        dtype=layer.parameter.dtype,
    ).reshape(layer.shape)
    with torch.no_grad():
        layer.parameter.add_(update, alpha=applied_norm)
    try:
        yield applied_norm
    finally:
        with torch.no_grad():
            layer.parameter.copy_(original)
        if not torch.equal(layer.parameter.detach(), original):
            raise RuntimeError("Virtual update failed to restore the base parameter.")

