"""Shared deterministic long-tail class selection."""

from __future__ import annotations

import numpy as np


def long_tail_indices(
    labels: np.ndarray,
    *,
    num_classes: int,
    imbalance_type: str,
    imbalance_factor: float,
    seed: int,
) -> np.ndarray:
    """Select balanced or exponentially decaying per-class subsets."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("Long-tail labels must be a vector.")
    if num_classes < 2:
        raise ValueError("Long-tail selection requires at least two classes.")
    if imbalance_type not in {"exp", "balanced"}:
        raise ValueError("imbalance_type must be 'exp' or 'balanced'.")
    if not 0.0 < imbalance_factor <= 1.0:
        raise ValueError("imbalance_factor must be in (0, 1].")
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts != counts[0]):
        raise ValueError("Source split must be class-balanced before long-tail selection.")

    n_max = int(counts[0])
    rng = np.random.RandomState(seed)
    selected: list[np.ndarray] = []
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        if imbalance_type == "exp":
            exponent = class_id / (num_classes - 1.0)
            keep = int(n_max * imbalance_factor**exponent)
        else:
            keep = n_max
        selected.append(class_indices[:keep])
    return np.concatenate(selected).astype(np.int64, copy=False)
