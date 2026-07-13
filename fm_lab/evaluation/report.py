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
    recall_k: int = 3,
    inception_splits: int = 10,
) -> dict[str, Any]:
    if repeats < 1 or overall_samples < 2:
        raise ValueError("Evaluation requires positive repeats and at least two samples.")
    if len(class_counts) != len(np.unique(real.labels)):
        raise ValueError("class_counts must align with the classes in the real cache.")
    sample_size = min(overall_samples, len(generated.features))
    groups = frequency_ranked_groups(class_counts)
    rng = np.random.default_rng(seed)
    overall_values = {name: [] for name in ("fid", "kid", "recall", "inception_score")}
    class_values: dict[int, list[float]] = {}
    group_values = {name: [] for name in groups}
    repeat_records = []

    for repeat in range(repeats):
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
        overall_values["fid"].append(fid)
        overall_values["kid"].append(kid)
        overall_values["recall"].append(recall)
        overall_values["inception_score"].append(float(score["mean"]))
        for class_id, value in per_class.items():
            class_values.setdefault(class_id, []).append(value)
        for name, value in per_group.items():
            group_values[name].append(value)
        repeat_records.append(
            {
                "repeat": repeat,
                "selection_seed": seed,
                "kid_seed": seed + repeat,
                "sample_count": sample_size,
                "fid": fid,
                "kid": kid,
                "recall": recall,
                "inception_score": score,
                "classwise_fid": {str(key): value for key, value in per_class.items()},
                "group_fid": per_group,
            }
        )

    metrics: dict[str, Any] = {name: summarize(values) for name, values in overall_values.items()}
    metrics["classwise_fid"] = {
        f"class_{class_id}": summarize(values) for class_id, values in sorted(class_values.items())
    }
    metrics["group_fid"] = {name: summarize(values) for name, values in group_values.items()}
    return {
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
        },
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
