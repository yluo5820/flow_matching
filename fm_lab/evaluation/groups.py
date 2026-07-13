"""Frequency-based class group definitions and metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from fm_lab.evaluation.metrics import fid_score


def frequency_ranked_groups(class_counts: Sequence[int]) -> dict[str, list[int]]:
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.ndim != 1 or len(counts) < 3 or np.any(counts < 1):
        raise ValueError("class_counts must contain at least three positive counts.")
    ranking = np.argsort(-counts, kind="stable")
    many, medium, few = np.array_split(ranking, 3)
    return {
        "many": [int(value) for value in many],
        "medium": [int(value) for value in medium],
        "few": [int(value) for value in few],
    }


def grouped_fid(
    generated: np.ndarray,
    generated_labels: np.ndarray,
    real: np.ndarray,
    real_labels: np.ndarray,
    groups: Mapping[str, Sequence[int]],
) -> dict[str, float]:
    generated_labels = np.asarray(generated_labels)
    real_labels = np.asarray(real_labels)
    result = {}
    for name in ("many", "medium", "few"):
        if name not in groups or len(groups[name]) == 0:
            raise ValueError("groups must define non-empty many, medium, and few classes.")
        generated_mask = np.isin(generated_labels, groups[name])
        real_mask = np.isin(real_labels, groups[name])
        result[name] = fid_score(generated[generated_mask], real[real_mask])
    return result
