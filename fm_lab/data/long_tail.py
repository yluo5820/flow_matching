"""Shared deterministic long-tail class selection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrequencySplit:
    """Nested training indices and disjoint diagnostic pools for one mapping."""

    train_indices: np.ndarray
    probe_a_indices: np.ndarray
    probe_b_indices: np.ndarray
    class_counts: tuple[int, ...]
    class_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("train_indices", "probe_a_indices", "probe_b_indices"):
            values = np.asarray(getattr(self, field_name), dtype=np.int64).copy()
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)
        object.__setattr__(
            self,
            "class_counts",
            tuple(int(value) for value in self.class_counts),
        )
        object.__setattr__(
            self,
            "class_ranks",
            tuple(int(value) for value in self.class_ranks),
        )


def frequency_rank_mapping(
    num_classes: int,
    *,
    multiplier: int,
    offset: int,
) -> np.ndarray:
    """Assign every class a frequency rank using a cyclic Latin mapping."""

    if num_classes < 2:
        raise ValueError("Frequency mapping requires at least two classes.")
    if math.gcd(int(multiplier), num_classes) != 1:
        raise ValueError(
            "Frequency mapping multiplier must be coprime with num_classes."
        )
    classes = np.arange(num_classes, dtype=np.int64)
    mapping = (int(multiplier) * classes + int(offset)) % num_classes
    mapping.setflags(write=False)
    return mapping


def nested_frequency_split(
    labels: np.ndarray,
    *,
    num_classes: int,
    imbalance_factor: float,
    seed: int,
    diagnostic_pool_per_class: int,
    multiplier: int,
    offset: int,
) -> FrequencySplit:
    """Reserve probes, then retain nested class prefixes at mapped frequencies."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError("Long-tail labels must be a vector.")
    if diagnostic_pool_per_class <= 0 or diagnostic_pool_per_class % 2:
        raise ValueError("diagnostic_pool_per_class must be positive and even.")
    if not 0.0 < imbalance_factor <= 1.0:
        raise ValueError("imbalance_factor must be in (0, 1].")
    counts = np.bincount(labels, minlength=num_classes)
    if len(counts) != num_classes or np.any(counts != counts[0]):
        raise ValueError("Source split must be class-balanced before splitting.")
    if diagnostic_pool_per_class >= int(counts[0]):
        raise ValueError("Diagnostic pool must leave at least one training example.")

    ranks = frequency_rank_mapping(
        num_classes,
        multiplier=multiplier,
        offset=offset,
    )
    probe_half = diagnostic_pool_per_class // 2
    n_max = int(counts[0]) - diagnostic_pool_per_class
    rng = np.random.RandomState(seed)
    train_parts: list[np.ndarray] = []
    probe_a_parts: list[np.ndarray] = []
    probe_b_parts: list[np.ndarray] = []
    retained_counts: list[int] = []
    for class_id in range(num_classes):
        ordered = np.flatnonzero(labels == class_id)
        rng.shuffle(ordered)
        probe_a_parts.append(ordered[:probe_half])
        probe_b_parts.append(ordered[probe_half:diagnostic_pool_per_class])
        candidates = ordered[diagnostic_pool_per_class:]
        exponent = float(ranks[class_id]) / (num_classes - 1.0)
        keep = max(1, int(n_max * imbalance_factor**exponent))
        train_parts.append(candidates[:keep])
        retained_counts.append(keep)

    return FrequencySplit(
        train_indices=np.concatenate(train_parts),
        probe_a_indices=np.concatenate(probe_a_parts),
        probe_b_indices=np.concatenate(probe_b_parts),
        class_counts=tuple(retained_counts),
        class_ranks=tuple(int(rank) for rank in ranks),
    )


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
            keep = max(1, int(n_max * imbalance_factor**exponent))
        else:
            keep = n_max
        selected.append(class_indices[:keep])
    return np.concatenate(selected).astype(np.int64, copy=False)
