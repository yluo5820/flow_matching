"""Metric orchestration and report serialization for cached features."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fm_lab.evaluation.cache import FeatureCache
from fm_lab.evaluation.groups import frequency_ranked_groups, grouped_fid
from fm_lab.evaluation.metrics import (
    classwise_fid,
    fid_score,
    generative_recall,
    inception_score,
    kid_score,
    summarize,
)


def evaluate_reference_calibration(
    real: FeatureCache,
    *,
    seed: int = 0,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    recall_k: int = 5,
    inception_splits: int = 10,
) -> dict[str, Any]:
    """Compare two fixed stratified halves of a real reference cache."""

    rng = np.random.default_rng(seed)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    for class_id in sorted(np.unique(real.labels).tolist()):
        indices = np.flatnonzero(real.labels == class_id)
        if len(indices) < 2:
            raise ValueError("Reference calibration requires two samples per class.")
        indices = rng.permutation(indices)
        half = len(indices) // 2
        left_parts.append(indices[:half])
        right_parts.append(indices[half : 2 * half])
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    score = inception_score(real.probabilities[left], splits=inception_splits)
    return {
        "seed": seed,
        "samples_per_half": int(len(left)),
        "metrics": {
            "fid": fid_score(real.features[left], real.features[right]),
            "kid": kid_score(
                real.features[left],
                real.features[right],
                num_subsets=kid_subsets,
                max_subset_size=kid_subset_size,
                seed=seed,
            ),
            "recall": generative_recall(
                real.features[left],
                real.features[right],
                nearest_k=recall_k,
            ),
            "inception_score": score,
        },
    }


def evaluate_feature_caches(
    generated: FeatureCache,
    real: FeatureCache,
    *,
    class_counts: list[int],
    repeats: int = 2,
    overall_samples: int = 50_000,
    seed: int = 0,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    recall_k: int = 5,
    inception_splits: int = 10,
    require_balanced_generated: bool = False,
    per_class_recall: bool = False,
    conditional_diagnostics: bool = False,
) -> dict[str, Any]:
    if repeats < 1 or overall_samples < 2:
        raise ValueError("Evaluation requires positive repeats and at least two samples.")
    if len(class_counts) != len(np.unique(real.labels)):
        raise ValueError("class_counts must align with the classes in the real cache.")
    if require_balanced_generated:
        generated_counts = np.bincount(generated.labels, minlength=len(class_counts))
        if len(generated_counts) != len(class_counts) or np.any(
            generated_counts != generated_counts[0]
        ):
            raise ValueError("Balanced evaluation requires equal generated samples per class.")
    sample_size = min(overall_samples, len(generated.features))
    if require_balanced_generated and sample_size % len(class_counts):
        raise ValueError("Balanced overall sample count must be divisible by class count.")
    groups = frequency_ranked_groups(class_counts)
    rng = np.random.default_rng(seed)
    overall_values = {name: [] for name in ("fid", "kid", "recall", "inception_score")}
    class_values: dict[int, list[float]] = {}
    group_values = {name: [] for name in groups}
    macro_class_values: list[float] = []
    worst_class_values: list[float] = []
    class_recall_values: dict[int, list[float]] = {}
    repeat_records = []

    for repeat in range(repeats):
        if require_balanced_generated:
            indices = _balanced_sample_indices(
                generated.labels,
                sample_size=sample_size,
                num_classes=len(class_counts),
                rng=rng,
            )
        else:
            indices = rng.choice(len(generated.features), sample_size, replace=False)
        features = generated.features[indices]
        probabilities = generated.probabilities[indices]
        labels = generated.labels[indices]
        fid = fid_score(features, real.features)
        kid = kid_score(
            features,
            real.features,
            num_subsets=kid_subsets,
            max_subset_size=kid_subset_size,
            seed=seed + repeat,
        )
        recall = generative_recall(features, real.features, nearest_k=recall_k)
        score = inception_score(probabilities, splits=inception_splits)
        per_class = classwise_fid(features, labels, real.features, real.labels)
        per_group = grouped_fid(features, labels, real.features, real.labels, groups)
        per_recall = (
            _classwise_recall(
                features,
                labels,
                real.features,
                real.labels,
                nearest_k=recall_k,
            )
            if per_class_recall
            else {}
        )
        overall_values["fid"].append(fid)
        overall_values["kid"].append(kid)
        overall_values["recall"].append(recall)
        overall_values["inception_score"].append(float(score["mean"]))
        for class_id, value in per_class.items():
            class_values.setdefault(class_id, []).append(value)
        for name, value in per_group.items():
            group_values[name].append(value)
        macro_class_values.append(float(np.mean(list(per_class.values()))))
        worst_class_values.append(float(np.max(list(per_class.values()))))
        for class_id, value in per_recall.items():
            class_recall_values.setdefault(class_id, []).append(value)
        repeat_records.append(
            {
                "repeat": repeat,
                "selection_seed": seed,
                "kid_seed": seed + repeat,
                "sample_count": sample_size,
                "generated_class_counts": np.bincount(
                    labels, minlength=len(class_counts)
                ).astype(int).tolist(),
                "fid": fid,
                "kid": kid,
                "recall": recall,
                "inception_score": score,
                "classwise_fid": {str(key): value for key, value in per_class.items()},
                "group_fid": per_group,
                "classwise_recall": {
                    str(key): value for key, value in per_recall.items()
                },
            }
        )

    metrics: dict[str, Any] = {name: summarize(values) for name, values in overall_values.items()}
    metrics["classwise_fid"] = {
        f"class_{class_id}": summarize(values) for class_id, values in sorted(class_values.items())
    }
    metrics["group_fid"] = {name: summarize(values) for name, values in group_values.items()}
    if per_class_recall:
        metrics["macro_classwise_fid"] = summarize(macro_class_values)
        metrics["worst_class_fid"] = summarize(worst_class_values)
        metrics["classwise_recall"] = {
            f"class_{class_id}": summarize(values)
            for class_id, values in sorted(class_recall_values.items())
        }
    report = {
        "metrics": metrics,
        "groups": groups,
        "class_counts": [int(value) for value in class_counts],
        "repeats": repeat_records,
        "provenance": {
            "evaluator": "fm_lab.imbdiff",
            "evaluator_version": 1,
            "fid_kid_compatibility": "ImbDiff-CM reference",
            "extended_metrics": [
                "recall",
                "inception_score",
                "classwise_fid",
                "group_fid",
            ],
            "real_cache": real.provenance,
            "generated_cache": generated.provenance,
            "seed": seed,
            "overall_samples": sample_size,
            "kid_subsets": kid_subsets,
            "kid_subset_size": kid_subset_size,
            "recall_k": recall_k,
            "inception_splits": inception_splits,
            "require_balanced_generated": require_balanced_generated,
            "per_class_recall": per_class_recall,
            "conditional_diagnostics": conditional_diagnostics,
        },
    }
    if conditional_diagnostics:
        report["conditional"] = _conditional_diagnostics(
            generated.labels,
            generated.probabilities,
            num_classes=len(class_counts),
        )
    return report


def _balanced_sample_indices(
    labels: np.ndarray,
    *,
    sample_size: int,
    num_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    per_class = sample_size // num_classes
    parts = []
    for class_id in range(num_classes):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < per_class:
            raise ValueError(f"Class {class_id} has too few samples for balanced evaluation.")
        parts.append(rng.choice(candidates, per_class, replace=False))
    indices = np.concatenate(parts)
    rng.shuffle(indices)
    return indices


def _classwise_recall(
    generated: np.ndarray,
    generated_labels: np.ndarray,
    real: np.ndarray,
    real_labels: np.ndarray,
    *,
    nearest_k: int,
) -> dict[int, float]:
    result = {}
    for class_id in sorted(np.unique(real_labels).tolist()):
        generated_class = generated[generated_labels == class_id]
        real_class = real[real_labels == class_id]
        result[int(class_id)] = generative_recall(
            generated_class,
            real_class,
            nearest_k=nearest_k,
        )
    return result


def _conditional_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), num_classes):
        raise ValueError("Conditional probabilities must align with labels and classes.")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Conditional labels must be valid class identifiers.")
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    return {
        "requested_class_accuracy": float(np.mean(predicted == labels)),
        "mean_requested_class_probability": float(
            np.mean(probabilities[np.arange(len(labels)), labels])
        ),
        "predicted_class_histogram": np.bincount(
            predicted, minlength=num_classes
        ).astype(int).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def write_evaluation_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for name, value in report["metrics"].items():
        if "mean" in value:
            rows.append(_summary_row(name, value))
        else:
            rows.extend(
                _summary_row(f"{name}.{scope}", summary) for scope, summary in value.items()
            )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "mean", "std", "all"])
        writer.writeheader()
        writer.writerows(rows)
    return {"json": json_path, "csv": csv_path}


def _summary_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric": name,
        "mean": summary["mean"],
        "std": summary["std"],
        "all": json.dumps(summary["all"]),
    }
