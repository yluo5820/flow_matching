"""Versioned and provenance-checked Inception feature caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FeatureCache:
    features: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    provenance: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return cache_fingerprint(self.provenance)


def cache_fingerprint(provenance: dict[str, Any]) -> str:
    encoded = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_feature_cache(path: str | Path, cache: FeatureCache) -> None:
    _validate_cache(cache)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_json = json.dumps(cache.provenance, sort_keys=True, separators=(",", ":"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                features=np.asarray(cache.features),
                probabilities=np.asarray(cache.probabilities),
                labels=np.asarray(cache.labels, dtype=np.int64),
                sample_ids=np.asarray(cache.sample_ids, dtype=str),
                provenance_json=np.asarray(provenance_json),
                fingerprint=np.asarray(cache.fingerprint),
            )
        os.replace(temporary_name, output_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_feature_cache(
    path: str | Path,
    *,
    expected_provenance: dict[str, Any] | None = None,
) -> FeatureCache:
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as payload:
        required = {
            "features",
            "probabilities",
            "labels",
            "sample_ids",
            "provenance_json",
            "fingerprint",
        }
        if not required.issubset(payload.files):
            raise ValueError(
                f"Feature cache is missing fields: {sorted(required - set(payload.files))}."
            )
        provenance = json.loads(str(payload["provenance_json"].item()))
        stored_fingerprint = str(payload["fingerprint"].item())
        cache = FeatureCache(
            features=payload["features"].copy(),
            probabilities=payload["probabilities"].copy(),
            labels=payload["labels"].copy(),
            sample_ids=payload["sample_ids"].copy(),
            provenance=provenance,
        )
    _validate_cache(cache)
    if stored_fingerprint != cache.fingerprint:
        raise ValueError("Feature cache fingerprint is corrupted.")
    if expected_provenance is not None:
        expected = cache_fingerprint(expected_provenance)
        if stored_fingerprint != expected:
            raise ValueError("Feature cache fingerprint does not match expected provenance.")
    return cache


def _validate_cache(cache: FeatureCache) -> None:
    features = np.asarray(cache.features)
    probabilities = np.asarray(cache.probabilities)
    labels = np.asarray(cache.labels)
    sample_ids = np.asarray(cache.sample_ids)
    if features.ndim != 2 or probabilities.ndim != 2:
        raise ValueError("Feature and probability arrays must be two-dimensional.")
    size = len(features)
    if size == 0 or len(probabilities) != size or len(labels) != size or len(sample_ids) != size:
        raise ValueError("Feature cache arrays must be non-empty and aligned.")
    if labels.ndim != 1 or sample_ids.ndim != 1:
        raise ValueError("Feature labels and sample identifiers must be vectors.")
    if not np.isfinite(features).all() or not np.isfinite(probabilities).all():
        raise ValueError("Feature cache arrays must contain finite values.")
    if not isinstance(cache.provenance, dict) or not cache.provenance:
        raise ValueError("Feature cache provenance must be a non-empty mapping.")
