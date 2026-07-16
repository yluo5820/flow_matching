"""Immutable paired-probe manifests and deterministic tuple materialization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_SCHEMA_VERSION = 1
_ARRAY_FIELDS = (
    "original_indices",
    "labels",
    "dequantization_seeds",
    "source_seeds",
    "timesteps",
    "stratum_ids",
    "microbatch_ids",
)


@dataclass(frozen=True)
class ProbeBatch:
    """One materialized subset of rows from a paired probe manifest."""

    x0: torch.Tensor
    x1: torch.Tensor
    t: torch.Tensor
    labels: torch.Tensor
    original_indices: np.ndarray
    stratum_ids: np.ndarray
    microbatch_ids: np.ndarray


@dataclass(frozen=True)
class ProbeManifest:
    """Serializable row-level random choices shared by every mapped model."""

    split: str
    original_indices: np.ndarray
    labels: np.ndarray
    dequantization_seeds: np.ndarray
    source_seeds: np.ndarray
    timesteps: np.ndarray
    stratum_ids: np.ndarray
    microbatch_ids: np.ndarray
    time_strata: tuple[tuple[float, float], ...]
    batch_size: int
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        normalized_split = str(self.split).lower()
        if normalized_split not in {"a", "b"}:
            raise ValueError("Probe manifest split must be 'a' or 'b'.")
        object.__setattr__(self, "split", normalized_split)
        if int(self.schema_version) != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported probe manifest schema version: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))
        if int(self.batch_size) < 1:
            raise ValueError("Probe manifest batch_size must be positive.")
        object.__setattr__(self, "batch_size", int(self.batch_size))

        dtypes = {
            "original_indices": np.int64,
            "labels": np.int64,
            "dequantization_seeds": np.int64,
            "source_seeds": np.int64,
            "timesteps": np.float64,
            "stratum_ids": np.int64,
            "microbatch_ids": np.int64,
        }
        lengths: set[int] = set()
        for field_name in _ARRAY_FIELDS:
            values = np.asarray(getattr(self, field_name), dtype=dtypes[field_name])
            if values.ndim != 1:
                raise ValueError(f"Probe manifest {field_name} must be a vector.")
            values = np.ascontiguousarray(values).copy()
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)
            lengths.add(len(values))
        if lengths != {self.num_rows} or self.num_rows < 1:
            raise ValueError("Probe manifest arrays must have one common positive length.")

        strata = tuple((float(low), float(high)) for low, high in self.time_strata)
        if not strata or any(
            not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high <= 1
            for low, high in strata
        ):
            raise ValueError("Probe manifest time strata must be finite intervals in [0, 1].")
        object.__setattr__(self, "time_strata", strata)
        if not np.all(np.isfinite(self.timesteps)):
            raise ValueError("Probe manifest timesteps must be finite.")
        if np.any(self.stratum_ids < 0) or np.any(self.stratum_ids >= len(strata)):
            raise ValueError("Probe manifest stratum_ids are out of range.")
        for row, stratum_id in enumerate(self.stratum_ids):
            low, high = strata[int(stratum_id)]
            if not low < float(self.timesteps[row]) < high:
                raise ValueError("Probe manifest timestep lies outside its open stratum.")

        unique_microbatches = np.unique(self.microbatch_ids)
        if not np.array_equal(
            unique_microbatches,
            np.arange(len(unique_microbatches), dtype=np.int64),
        ):
            raise ValueError("Probe manifest microbatch_ids must be consecutive from zero.")
        if any(
            np.sum(self.microbatch_ids == microbatch_id) != self.batch_size
            for microbatch_id in unique_microbatches
        ):
            raise ValueError("Every probe manifest microbatch must have batch_size rows.")

        classes = np.unique(self.labels)
        cell_counts = [
            int(np.sum((self.labels == class_id) & (self.stratum_ids == stratum_id)))
            for class_id in classes
            for stratum_id in range(len(strata))
        ]
        if len(set(cell_counts)) != 1:
            raise ValueError("Probe manifest must balance every class/stratum cell.")

    @property
    def num_rows(self) -> int:
        return int(len(self.original_indices))

    @property
    def digest(self) -> str:
        hasher = hashlib.sha256()
        metadata = {
            "schema_version": self.schema_version,
            "split": self.split,
            "batch_size": self.batch_size,
            "time_strata": self.time_strata,
        }
        hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
        for field_name in _ARRAY_FIELDS:
            values = np.ascontiguousarray(getattr(self, field_name))
            hasher.update(field_name.encode("utf-8"))
            hasher.update(values.dtype.str.encode("ascii"))
            hasher.update(values.tobytes())
        return hasher.hexdigest()

    def microbatch_row_indices(self) -> tuple[np.ndarray, ...]:
        return tuple(
            np.flatnonzero(self.microbatch_ids == microbatch_id)
            for microbatch_id in np.unique(self.microbatch_ids)
        )

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            schema_version=np.asarray(self.schema_version, dtype=np.int64),
            split=np.asarray(self.split),
            batch_size=np.asarray(self.batch_size, dtype=np.int64),
            time_strata=np.asarray(self.time_strata, dtype=np.float64),
            digest=np.asarray(self.digest),
            **{field_name: getattr(self, field_name) for field_name in _ARRAY_FIELDS},
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> ProbeManifest:
        with np.load(Path(path), allow_pickle=False) as payload:
            manifest = cls(
                schema_version=int(payload["schema_version"].item()),
                split=str(payload["split"].item()),
                batch_size=int(payload["batch_size"].item()),
                time_strata=tuple(
                    (float(low), float(high)) for low, high in payload["time_strata"]
                ),
                **{field_name: payload[field_name] for field_name in _ARRAY_FIELDS},
            )
            stored_digest = str(payload["digest"].item())
        if manifest.digest != stored_digest:
            raise ValueError("Probe manifest digest does not match its contents.")
        return manifest


def build_probe_manifest(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    rows_per_class_per_stratum: int,
    batch_size: int,
    time_strata: tuple[tuple[float, float], ...],
    seed: int,
) -> ProbeManifest:
    """Create balanced probe rows with fixed per-row random choices."""

    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if sample_ids.ndim != 1 or labels.ndim != 1 or len(sample_ids) != len(labels):
        raise ValueError("Probe sample_ids and labels must be aligned vectors.")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Probe sample_ids must be unique.")
    if rows_per_class_per_stratum < 1:
        raise ValueError("rows_per_class_per_stratum must be positive.")
    if batch_size < 1 or rows_per_class_per_stratum % batch_size:
        raise ValueError(
            "rows_per_class_per_stratum must be divisible by batch_size."
        )
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("Probe manifests require at least two classes.")
    rng = np.random.RandomState(seed)
    fields: dict[str, list[np.ndarray]] = {
        field_name: [] for field_name in _ARRAY_FIELDS
    }
    microbatch_id = 0
    max_seed = np.iinfo(np.int64).max
    for class_id in classes:
        class_ids = sample_ids[labels == class_id]
        if len(class_ids) < rows_per_class_per_stratum:
            raise ValueError(
                f"Class {class_id} lacks enough unique examples for each stratum."
            )
        for stratum_id, (low, high) in enumerate(time_strata):
            chosen = rng.permutation(class_ids)[:rows_per_class_per_stratum]
            lower = np.nextafter(float(low), float(high))
            upper = np.nextafter(float(high), float(low))
            fields["original_indices"].append(chosen)
            fields["labels"].append(
                np.full(rows_per_class_per_stratum, class_id, dtype=np.int64)
            )
            fields["dequantization_seeds"].append(
                rng.randint(
                    0,
                    max_seed,
                    size=rows_per_class_per_stratum,
                    dtype=np.int64,
                )
            )
            fields["source_seeds"].append(
                rng.randint(
                    0,
                    max_seed,
                    size=rows_per_class_per_stratum,
                    dtype=np.int64,
                )
            )
            fields["timesteps"].append(
                rng.uniform(lower, upper, size=rows_per_class_per_stratum).astype(
                    np.float64
                )
            )
            fields["stratum_ids"].append(
                np.full(rows_per_class_per_stratum, stratum_id, dtype=np.int64)
            )
            microbatches = rows_per_class_per_stratum // batch_size
            fields["microbatch_ids"].append(
                np.repeat(
                    np.arange(
                        microbatch_id,
                        microbatch_id + microbatches,
                        dtype=np.int64,
                    ),
                    batch_size,
                )
            )
            microbatch_id += microbatches
    return ProbeManifest(
        split=split,
        time_strata=time_strata,
        batch_size=batch_size,
        **{
            field_name: np.concatenate(chunks)
            for field_name, chunks in fields.items()
        },
    )


def build_source_noise_replica(
    manifest: ProbeManifest,
    *,
    seed: int,
) -> ProbeManifest:
    """Copy a manifest while replacing only its source-noise seeds."""

    rng = np.random.RandomState(int(seed))
    source_seeds = rng.randint(
        0,
        np.iinfo(np.int64).max,
        size=manifest.num_rows,
        dtype=np.int64,
    )
    return ProbeManifest(
        split=manifest.split,
        original_indices=manifest.original_indices,
        labels=manifest.labels,
        dequantization_seeds=manifest.dequantization_seeds,
        source_seeds=source_seeds,
        timesteps=manifest.timesteps,
        stratum_ids=manifest.stratum_ids,
        microbatch_ids=manifest.microbatch_ids,
        time_strata=manifest.time_strata,
        batch_size=manifest.batch_size,
        schema_version=manifest.schema_version,
    )


def materialize_probe_batch(
    target: Any,
    source: Any,
    manifest: ProbeManifest,
    row_indices: np.ndarray,
    *,
    device: torch.device | str,
) -> ProbeBatch:
    """Materialize exactly the tuples selected by manifest rows."""

    rows = np.asarray(row_indices, dtype=np.int64)
    if rows.ndim != 1 or len(rows) < 1:
        raise ValueError("Probe batch row_indices must be a non-empty vector.")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Probe batch row_indices must not contain duplicates.")
    if np.any(rows < 0) or np.any(rows >= manifest.num_rows):
        raise ValueError("Probe batch row_indices are out of range.")

    original_indices = manifest.original_indices[rows]
    expected_labels = manifest.labels[rows]
    x1, labels, returned_ids = target.diagnostic_samples(
        manifest.split,
        original_indices=original_indices,
        dequantization_seeds=manifest.dequantization_seeds[rows],
        device="cpu",
    )
    if not np.array_equal(returned_ids.astype(np.int64), original_indices):
        raise ValueError("Target returned diagnostic samples in a different ID order.")
    if not torch.equal(labels.cpu(), torch.from_numpy(expected_labels)):
        raise ValueError("Target diagnostic labels do not match the probe manifest.")

    source_rows: list[torch.Tensor] = []
    for source_seed in manifest.source_seeds[rows]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(source_seed))
            source_rows.append(source.sample(1, device="cpu")[0])
    x0 = torch.stack(source_rows)
    resolved_device = torch.device(device)
    return ProbeBatch(
        x0=x0.to(resolved_device),
        x1=x1.to(resolved_device),
        t=torch.from_numpy(manifest.timesteps[rows].copy()).to(
            resolved_device,
            dtype=x1.dtype,
        ),
        labels=labels.to(resolved_device),
        original_indices=original_indices.copy(),
        stratum_ids=manifest.stratum_ids[rows].copy(),
        microbatch_ids=manifest.microbatch_ids[rows].copy(),
    )
