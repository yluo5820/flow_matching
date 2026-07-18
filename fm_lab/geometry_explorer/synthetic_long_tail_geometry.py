"""Local learned-geometry and memorization diagnostics for the synthetic study."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch import nn

from fm_lab.diagnostics.fm_lid import (
    FMFLIPDEstimator,
    FMJacobianSpectrumEstimator,
    GaussianFMSchedule,
    entropy_rank,
    participation_rank,
    threshold_rank,
)
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.solvers import Solver


class _FixedClassVelocity(nn.Module):
    def __init__(self, model: nn.Module, class_id: int) -> None:
        super().__init__()
        self.model = model
        self.class_id = class_id

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1:
            raise ValueError("class-conditioned velocity expects a batched tensor.")
        labels = torch.full(
            (len(x),),
            self.class_id,
            dtype=torch.long,
            device=x.device,
        )
        return self.model(x, t, context={"class_labels": labels})


def fixed_class_velocity(model: nn.Module, class_id: int) -> nn.Module:
    """Bind a velocity model to one class while preserving its tensor API."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    if isinstance(class_id, bool) or not isinstance(class_id, int):
        raise TypeError("class_id must be an integer.")
    if class_id < 0:
        raise ValueError("class_id must be non-negative.")
    return _FixedClassVelocity(model, class_id)


def tangent_projection_scores(
    pushforward: torch.Tensor,
    renderer_tangents: torch.Tensor,
    *,
    rank: int,
) -> torch.Tensor:
    """Return squared projection of each renderer tangent onto learned directions."""

    if pushforward.ndim != 2 or renderer_tangents.ndim != 2:
        raise ValueError("pushforward and renderer_tangents must both be matrices.")
    if renderer_tangents.shape[1] != pushforward.shape[0]:
        raise ValueError("renderer tangent width must equal pushforward output dimension.")
    if not torch.is_floating_point(pushforward) or not torch.is_floating_point(renderer_tangents):
        raise TypeError("pushforward and renderer tangents must be floating-point.")
    if not bool(torch.isfinite(pushforward).all()) or not bool(
        torch.isfinite(renderer_tangents).all()
    ):
        raise ValueError("pushforward and renderer tangents must be finite.")
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("rank must be an integer.")
    maximum_rank = min(pushforward.shape)
    if rank < 1 or rank > maximum_rank:
        raise ValueError(f"rank must lie in [1, {maximum_rank}].")
    norms = torch.linalg.vector_norm(renderer_tangents, dim=1)
    if bool((norms <= torch.finfo(renderer_tangents.dtype).eps).any()):
        raise ValueError("renderer tangents must be nonzero.")
    left, _, _ = torch.linalg.svd(pushforward, full_matrices=False)
    basis = left[:, :rank]
    normalized = F.normalize(renderer_tangents, dim=1)
    return torch.sum((normalized @ basis) ** 2, dim=1).clamp(0.0, 1.0)


def evaluate_local_geometry(
    *,
    model: nn.Module,
    ode_solver: Solver,
    queries: np.ndarray | torch.Tensor,
    class_ids: np.ndarray | torch.Tensor,
    renderer_tangents: np.ndarray | torch.Tensor,
    tangent_names: Sequence[str],
    output_dir: str | Path,
    t_values: Sequence[float] = (0.6, 0.8, 0.9, 0.95),
    renderer_rank: int | Sequence[int] | None = None,
    eps: float = 1.0e-2,
    num_directions: int = 16,
    nfe: int = 32,
    threshold: float = 1.0e-2,
    fm_schedule: str = "linear",
    num_trace_samples: int | None = 1,
    seed: int = 0,
    device: str | torch.device = "auto",
    source_revision: str = "unknown",
) -> dict[str, Any]:
    """Evaluate factor alignment and FM dimension on deterministic held-out queries.

    ``model`` must already be converted to velocity prediction for its training
    objective. Tangents are rows in normalized renderer-factor coordinates and
    flattened into the same representation as the learned pushforward output.
    """

    points = _float_tensor(queries, "queries")
    labels = _class_vector(class_ids, len(points))
    tangents = _float_tensor(renderer_tangents, "renderer_tangents")
    if tangents.ndim != 3 or tangents.shape[0] != len(points):
        raise ValueError("renderer_tangents must have shape (N, factors, ambient).")
    ambient = int(points[0].numel())
    if tangents.shape[2] != ambient:
        raise ValueError("renderer tangents must match the flattened query dimension.")
    names = tuple(str(name) for name in tangent_names)
    if not names or len(names) != tangents.shape[1] or len(set(names)) != len(names):
        raise ValueError("tangent_names must be unique and match the tangent count.")
    times = _times(t_values)
    ranks = _renderer_ranks(renderer_rank, len(points), tangents.shape[1], num_directions)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    destination = _destination(output_dir, "Local-geometry")

    rows: list[dict[str, Any]] = []
    spectra: list[np.ndarray] = []
    for query_index, (point, class_id) in enumerate(zip(points, labels, strict=True)):
        class_velocity = fixed_class_velocity(model, int(class_id))
        flipd = FMFLIPDEstimator(
            class_velocity,
            GaussianFMSchedule(fm_schedule),
            times,
            num_trace_samples=num_trace_samples,
            device=device,
        ).estimate_batch(point.unsqueeze(0))
        for time_index, time_value in enumerate(times):
            generator = torch.Generator(device="cpu").manual_seed(
                seed + query_index * len(times) + time_index
            )
            estimator = FMJacobianSpectrumEstimator(
                model=class_velocity,
                ode_solver=ode_solver,
                t_values=(time_value,),
                eps=eps,
                num_directions=num_directions,
                threshold=threshold,
                device=device,
                nfe=nfe,
                generator=generator,
            )
            matrix = estimator.compute_pushforward_matrix(point, time_value)
            singular = torch.linalg.svdvals(matrix)
            tangent = tangents[query_index].to(matrix.device, matrix.dtype)
            scores = tangent_projection_scores(matrix, tangent, rank=ranks[query_index])
            principal_angles = _principal_angles(matrix, tangent, ranks[query_index])
            row: dict[str, Any] = {
                "query_index": query_index,
                "class_id": int(class_id),
                "time": time_value,
                "renderer_rank": ranks[query_index],
                "participation_rank": float(participation_rank(singular).cpu()),
                "entropy_rank": float(entropy_rank(singular).cpu()),
                "threshold_rank": int(threshold_rank(singular, threshold=threshold).cpu()),
                "principal_angle_mean": float(principal_angles.mean().cpu()),
                "principal_angle_max": float(principal_angles.max().cpu()),
                "fm_flipd_lid": float(flipd.lid[time_index, 0].detach().cpu()),
            }
            row.update(
                {
                    f"alignment_{name}": float(score.cpu())
                    for name, score in zip(names, scores, strict=True)
                }
            )
            rows.append(row)
            spectra.append(singular.detach().cpu().numpy())

    frame = pd.DataFrame(rows)
    numeric = [column for column in frame if column not in {"query_index", "class_id"}]
    class_summary = (
        frame.groupby(["class_id", "time"], as_index=False)[numeric]
        .mean(numeric_only=True)
        .to_dict(orient="records")
    )
    result = {
        "schema_version": 1,
        "query_count": len(points),
        "t_values": list(times),
        "tangent_names": list(names),
        "class_summary": class_summary,
        "provenance": {
            "seed": seed,
            "eps": float(eps),
            "num_directions": int(num_directions),
            "nfe": int(nfe),
            "threshold": float(threshold),
            "fm_schedule": fm_schedule,
            "source_revision": source_revision,
            "query_sha256": _array_sha256(points.numpy()),
            "tangent_sha256": _array_sha256(tangents.numpy()),
        },
    }

    def write(staging: Path) -> None:
        write_parquet(frame, staging / "per_query.parquet")
        np.savez_compressed(staging / "spectra.npz", singular_values=np.stack(spectra))
        (staging / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    _publish_directory(destination, write)
    return result


def evaluate_memorization(
    *,
    generated_images: np.ndarray,
    generated_factors: np.ndarray,
    generated_features: np.ndarray,
    training_images: np.ndarray,
    training_factors: np.ndarray,
    training_features: np.ndarray,
    heldout_images: np.ndarray,
    heldout_factors: np.ndarray,
    heldout_features: np.ndarray,
    output_dir: str | Path,
    generated_class_ids: np.ndarray | None = None,
    source_revision: str = "unknown",
    distance_chunk_size: int = 1024,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure exact-copy and nearest-neighbour memorization independently."""

    generated_images = _uint8_images(generated_images, "generated_images")
    training_images = _uint8_images(training_images, "training_images")
    heldout_images = _uint8_images(heldout_images, "heldout_images")
    if (
        generated_images.shape[1:] != training_images.shape[1:]
        or generated_images.shape[1:] != heldout_images.shape[1:]
    ):
        raise ValueError("all image arrays must share one image shape.")
    generated_factors = _feature_matrix(
        generated_factors, len(generated_images), "generated_factors"
    )
    training_factors = _feature_matrix(training_factors, len(training_images), "training_factors")
    heldout_factors = _feature_matrix(heldout_factors, len(heldout_images), "heldout_factors")
    generated_features = _feature_matrix(
        generated_features, len(generated_images), "generated_features"
    )
    training_features = _feature_matrix(
        training_features, len(training_images), "training_features"
    )
    heldout_features = _feature_matrix(heldout_features, len(heldout_images), "heldout_features")
    if (
        generated_factors.shape[1] != training_factors.shape[1]
        or generated_factors.shape[1] != heldout_factors.shape[1]
    ):
        raise ValueError("all factor matrices must share one width.")
    if (
        generated_features.shape[1] != training_features.shape[1]
        or generated_features.shape[1] != heldout_features.shape[1]
    ):
        raise ValueError("all oracle-feature matrices must share one width.")
    if len(heldout_features) < 2:
        raise ValueError("heldout_features must contain at least two independent samples.")
    if distance_chunk_size < 1:
        raise ValueError("distance_chunk_size must be positive.")
    destination = _destination(output_dir, "Memorization")
    labels = (
        np.full(len(generated_images), -1, dtype=np.int64)
        if generated_class_ids is None
        else _class_vector_numpy(generated_class_ids, len(generated_images))
    )

    train_factor_distance = _nearest_distances(
        generated_factors, training_factors, distance_chunk_size
    )
    heldout_factor_distance = _nearest_distances(
        generated_factors, heldout_factors, distance_chunk_size
    )
    train_feature_distance = _nearest_distances(
        generated_features, training_features, distance_chunk_size
    )
    heldout_feature_distance = _nearest_distances(
        generated_features, heldout_features, distance_chunk_size
    )
    heldout_to_training_factor_distance = _nearest_distances(
        heldout_factors, training_factors, distance_chunk_size
    )
    heldout_to_training_feature_distance = _nearest_distances(
        heldout_features, training_features, distance_chunk_size
    )
    threshold_value = float(np.percentile(heldout_to_training_feature_distance, 0.5))
    exact_copy = _exact_copy_flags(generated_images, training_images)
    near_duplicate = train_feature_distance <= threshold_value
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(generated_images)),
            "class_id": labels,
            "nearest_training_factor_distance": train_factor_distance,
            "nearest_heldout_factor_distance": heldout_factor_distance,
            "nearest_training_feature_distance": train_feature_distance,
            "nearest_heldout_feature_distance": heldout_feature_distance,
            "exact_training_copy": exact_copy,
            "near_training_duplicate": near_duplicate,
        }
    )
    class_summary = []
    for class_id, group in frame.groupby("class_id", sort=True):
        class_summary.append(_memorization_summary(group, int(class_id)))
    result = {
        "schema_version": 2,
        "summary": {
            **_memorization_summary(frame, None),
            "near_duplicate_threshold": threshold_value,
            "heldout_count": int(len(heldout_images)),
            "median_heldout_to_training_factor_distance": float(
                np.median(heldout_to_training_factor_distance)
            ),
            "median_heldout_to_training_feature_distance": float(
                np.median(heldout_to_training_feature_distance)
            ),
            "generated_to_heldout_training_proximity_factor_ratio": _median_ratio(
                train_factor_distance,
                heldout_to_training_factor_distance,
            ),
            "generated_to_heldout_training_proximity_feature_ratio": _median_ratio(
                train_feature_distance,
                heldout_to_training_feature_distance,
            ),
        },
        "class_summary": class_summary,
        "provenance": {
            "source_revision": source_revision,
            "distance_chunk_size": int(distance_chunk_size),
            "near_duplicate_definition": (
                "generated-to-training oracle-feature distance no greater than the "
                "0.5th percentile of independent heldout-to-the-same-training distances"
            ),
            "training_proximity_ratio_definition": (
                "median generated-to-training distance divided by median independent "
                "heldout-to-the-same-training distance; one is distribution-matched, "
                "below one is unusually training-proximal"
            ),
            "generated_images_sha256": _array_sha256(generated_images),
            "training_images_sha256": _array_sha256(training_images),
            "heldout_images_sha256": _array_sha256(heldout_images),
            "context": dict(context or {}),
        },
    }

    def write(staging: Path) -> None:
        write_parquet(frame, staging / "per_sample.parquet")
        np.savez_compressed(
            staging / "distances.npz",
            nearest_training_factor=train_factor_distance,
            nearest_heldout_factor=heldout_factor_distance,
            nearest_training_feature=train_feature_distance,
            nearest_heldout_feature=heldout_feature_distance,
            heldout_to_training_factor=heldout_to_training_factor_distance,
            heldout_to_training_feature=heldout_to_training_feature_distance,
            exact_training_copy=exact_copy,
            near_training_duplicate=near_duplicate,
        )
        (staging / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    _publish_directory(destination, write)
    return result


def _principal_angles(matrix: torch.Tensor, tangents: torch.Tensor, rank: int) -> torch.Tensor:
    left = torch.linalg.svd(matrix, full_matrices=False).U[:, :rank]
    tangent_basis = torch.linalg.qr(tangents.T, mode="reduced").Q
    cosines = torch.linalg.svdvals(tangent_basis.T @ left).clamp(0.0, 1.0)
    return torch.acos(cosines)


def _memorization_summary(frame: pd.DataFrame, class_id: int | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "generated_count": int(len(frame)),
        "exact_copy_rate": float(frame["exact_training_copy"].mean()),
        "near_duplicate_rate": float(frame["near_training_duplicate"].mean()),
        "median_nearest_training_factor_distance": float(
            frame["nearest_training_factor_distance"].median()
        ),
        "median_nearest_heldout_factor_distance": float(
            frame["nearest_heldout_factor_distance"].median()
        ),
        "median_nearest_training_feature_distance": float(
            frame["nearest_training_feature_distance"].median()
        ),
        "median_nearest_heldout_feature_distance": float(
            frame["nearest_heldout_feature_distance"].median()
        ),
    }
    if class_id is not None:
        result["class_id"] = class_id
    return result


def _median_ratio(numerator: np.ndarray, denominator: np.ndarray) -> float:
    baseline = float(np.median(denominator))
    if baseline <= 0.0:
        raise ValueError("heldout-to-training median distance must be positive.")
    return float(np.median(numerator)) / baseline


def _nearest_distances(query: np.ndarray, reference: np.ndarray, chunk: int) -> np.ndarray:
    result = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), chunk):
        result[start : start + chunk] = cdist(query[start : start + chunk], reference).min(axis=1)
    return result


def _exact_copy_flags(generated: np.ndarray, training: np.ndarray) -> np.ndarray:
    training_hashes = {_row_digest(row) for row in training}
    return np.asarray([_row_digest(row) in training_hashes for row in generated], dtype=bool)


def _row_digest(row: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest()


def _float_tensor(values: np.ndarray | torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32).detach().cpu()
    if tensor.ndim < 2 or len(tensor) == 0:
        raise ValueError(f"{name} must be a non-empty tensor with at least two dimensions.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite.")
    return tensor


def _class_vector(values: np.ndarray | torch.Tensor, count: int) -> np.ndarray:
    array = np.asarray(torch.as_tensor(values).detach().cpu())
    return _class_vector_numpy(array, count)


def _class_vector_numpy(values: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("class_ids must be an integer vector matching the sample count.")
    if bool(np.any(array < 0)):
        raise ValueError("class_ids must be non-negative.")
    return array.astype(np.int64, copy=False)


def _feature_matrix(values: np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (N, D) with nonzero D.")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be finite.")
    return array


def _uint8_images(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 2 or len(array) == 0 or array.dtype != np.uint8:
        raise ValueError(f"{name} must be a non-empty uint8 image batch.")
    return array


def _times(values: Sequence[float]) -> tuple[float, ...]:
    times = tuple(float(value) for value in values)
    if not times or not all(np.isfinite(value) and 0.0 < value < 1.0 for value in times):
        raise ValueError("t_values must be finite and strictly between zero and one.")
    if len(set(times)) != len(times):
        raise ValueError("t_values must not contain duplicates.")
    return times


def _renderer_ranks(
    value: int | Sequence[int] | None,
    count: int,
    tangent_count: int,
    num_directions: int,
) -> tuple[int, ...]:
    values = (
        (min(tangent_count, num_directions),) * count
        if value is None
        else (
            (value,) * count
            if isinstance(value, int) and not isinstance(value, bool)
            else tuple(value)
        )
    )
    maximum = min(tangent_count, num_directions)
    if len(values) != count or any(
        isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= maximum
        for item in values
    ):
        raise ValueError(f"renderer_rank must provide one integer in [1, {maximum}] per query.")
    return tuple(values)


def _destination(value: str | Path, label: str) -> Path:
    requested = Path(value).expanduser()
    destination = requested.parent.resolve() / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{label} destination already exists: {destination}")
    return destination


def _publish_directory(destination: Path, writer: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    published = False
    try:
        writer(staging)
        os.symlink(
            os.path.relpath(staging, start=destination.parent),
            destination,
            target_is_directory=True,
        )
        published = True
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(array.shape).encode("ascii"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()
