from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from fm_lab.experiments.fashion_mnist_frequency_response import (
    analyze_frequency_response,
    evaluate_calibration_gate,
    load_stage1_config,
    prepare_stage1,
    stage1_conditions,
)
from fm_lab.utils.config import load_config, save_config
from fm_lab.utils.logging import write_json


def _stage1_config(
    tmp_path: Path,
    *,
    class_ids: list[int] | None = None,
    frequency_multiplier: int = 3,
):
    class_ids = class_ids or list(range(10))
    stage0 = tmp_path / "stage0"
    stage0.mkdir()
    write_json(
        {
            "stage": "fashion_mnist_geometry_frequency_stage0",
            "config_hash": "stage0-hash",
            "passed": False,
            "reasons": ["no_stable_high_class"],
            "outcome_training_enabled": False,
        },
        stage0 / "selection_gate.json",
    )
    records = []
    for representation in ("raw_pca50", "dinov2_pca50"):
        for estimator in (
            "two_nn",
            "mle_lid_k10",
            "mle_lid_k20",
            "participation_ratio",
            "pca_dim_90",
        ):
            for class_id in range(10):
                records.append(
                    {
                        "representation": representation,
                        "probe_split": "a",
                        "subsample": 0,
                        "class_id": class_id,
                        "estimator": estimator,
                        "estimate": float(class_id),
                        "percentile_rank": (class_id + 1) / 10,
                    }
                )
    pd.DataFrame(records).to_csv(stage0 / "geometry_percentile_ranks.csv", index=False)
    base = tmp_path / "base.yaml"
    save_config(
        {
            "experiment": {"name": "base", "seed": 0, "output_dir": "unused"},
            "data": {"name": "fashion_mnist_lt"},
            "source": {"name": "gaussian", "dim": 784},
            "coupling": {"name": "independent"},
            "path": {"name": "linear"},
            "model": {
                "name": "image_unet",
                "image_shape": [1, 28, 28],
                "base_channels": 32,
            },
            "conditioning": {"enabled": True, "num_classes": 10},
            "objective": {
                "name": "flow_matching",
                "model_output": "target",
                "loss_space": "velocity",
            },
            "training": {"steps": 2000, "batch_size": 16},
            "solvers": {"names": ["euler"], "nfes": [64]},
            "sampling": {"n_samples": 10000},
        },
        base,
    )
    config_path = tmp_path / "stage1.yaml"
    save_config(
        {
            "experiment": {"name": "stage1", "seed": 13},
            "stage0": {"output_dir": str(stage0)},
            "data": {
                "root": str(tmp_path / "data"),
                "download": False,
                "subset_seed": 17,
                "diagnostic_pool_per_class": 1000,
                "imbalance_factor": 0.01,
                "class_ids": class_ids,
            },
            "design": {
                "frequency_multiplier": frequency_multiplier,
                "rotation_offsets": list(range(len(class_ids))),
                "sampling_policies": ["class_balanced"],
            },
            "training": {
                "base_config": str(base),
                "calibration": {
                    "steps": [2000, 5000, 10000],
                    "min_class_accuracy": 0.8,
                    "max_macro_fid_relative_improvement": 0.1,
                    "max_accuracy_improvement": 0.02,
                },
            },
            "evaluation": {
                "samples_per_class": 1000,
                "classifier_checkpoint": str(tmp_path / "classifier.pt"),
                "classifier_steps": 10,
                "classifier_minimum_accuracy": 0.9,
                "repeats": 1,
                "kid_subsets": 1,
            },
            "output": {
                "protocol_dir": str(tmp_path / "protocol"),
                "runs_dir": str(tmp_path / "runs"),
            },
        },
        config_path,
    )
    return load_stage1_config(config_path, project_root=tmp_path)


def _report(
    *,
    macro_fid: float = 100.0,
    accuracy: float = 0.9,
    fid_by_class: list[float] | None = None,
    recall_by_class: list[float] | None = None,
) -> dict:
    fid_by_class = fid_by_class or [macro_fid] * 10
    recall_by_class = recall_by_class or [0.8] * 10
    per_class = {
        f"class_{class_id}": {
            "sample_count": 1000,
            "requested_class_accuracy": accuracy,
            "mean_requested_class_probability": accuracy,
        }
        for class_id in range(10)
    }
    return {
        "metrics": {
            "macro_classwise_fid": {"mean": macro_fid},
            "classwise_fid": {
                f"class_{class_id}": {"mean": fid_by_class[class_id]}
                for class_id in range(10)
            },
            "classwise_recall": {
                f"class_{class_id}": {"mean": recall_by_class[class_id]}
                for class_id in range(10)
            },
        },
        "conditional": {
            "requested_class_accuracy": accuracy,
            "per_class": per_class,
        },
    }


def test_all_class_conditions_form_complete_cyclic_support_design(tmp_path: Path) -> None:
    config = _stage1_config(tmp_path)
    conditions = stage1_conditions(config)

    assert len(conditions) == 11
    assert conditions[0].class_counts == (5000,) * 10
    rotations = conditions[1:]
    for class_id in range(10):
        assert sorted(item.class_ranks[class_id] for item in rotations) == list(range(10))
        assert sorted(item.class_counts[class_id] for item in rotations) == sorted(
            rotations[0].class_counts
        )


def test_prepare_stage1_freezes_all_competing_geometry_predictors(tmp_path: Path) -> None:
    config = _stage1_config(tmp_path)

    result = prepare_stage1(config)
    geometry = pd.read_csv(config.output_dir / "frozen_geometry_predictors.csv")
    manifest = json.loads((config.output_dir / "condition_manifest.json").read_text())

    assert result["geometry_predictor_count"] == 100
    assert result["geometry_predictors_are_competing"] is True
    assert len(geometry) == 100
    assert len(manifest["conditions"]) == 11
    assert prepare_stage1(config)["reused"] is True


def test_stage1_subset_protocol_uses_compact_five_class_rotations(
    tmp_path: Path,
) -> None:
    config = _stage1_config(tmp_path, class_ids=[1, 5, 7, 8, 9], frequency_multiplier=2)

    result = prepare_stage1(config)
    conditions = stage1_conditions(config)
    geometry = pd.read_csv(config.output_dir / "frozen_geometry_predictors.csv")
    generated = load_config(
        config.output_dir
        / "condition_configs"
        / "class_balanced_offset_00.yaml"
    )

    assert result["class_ids"] == [1, 5, 7, 8, 9]
    assert result["geometry_predictor_count"] == 50
    assert len(conditions) == 6
    assert conditions[0].class_counts == (5000,) * 5
    assert sorted(conditions[1].class_counts) == [50, 158, 500, 1581, 5000]
    assert set(geometry["class_id"]) == set(range(5))
    assert set(geometry["original_class_id"]) == {1, 5, 7, 8, 9}
    assert generated["conditioning"]["num_classes"] == 5
    assert generated["sampling"]["classes"] == [0, 1, 2, 3, 4]


def test_calibration_gate_selects_earliest_converged_budget(tmp_path: Path) -> None:
    config = _stage1_config(tmp_path)
    prepare_stage1(config)
    for steps, report in ((2000, _report()), (5000, _report(macro_fid=95, accuracy=0.91))):
        path = config.runs_dir / "calibration" / f"steps_{steps:08d}" / "evaluation"
        write_json(report, path / "metrics.json")

    gate = evaluate_calibration_gate(config)

    assert gate["status"] == "passed"
    assert gate["selected_steps"] == 2000
    assert gate["comparisons"][0]["qualifies"] is True


def test_analysis_reports_support_and_geometry_effects_without_predictor_selection(
    tmp_path: Path,
) -> None:
    config = _stage1_config(tmp_path)
    prepare_stage1(config)
    balanced = _report(macro_fid=20.0, accuracy=0.95, fid_by_class=[20.0] * 10)
    calibration_root = config.runs_dir / "calibration"
    write_json(balanced, calibration_root / "steps_00002000" / "evaluation" / "metrics.json")
    write_json(
        _report(macro_fid=19.5, accuracy=0.955, fid_by_class=[19.5] * 10),
        calibration_root / "steps_00005000" / "evaluation" / "metrics.json",
    )
    max_log_support = math.log10(5000)
    for condition in stage1_conditions(config)[1:]:
        fids = []
        recalls = []
        for class_id, support in enumerate(condition.class_counts):
            gap = max_log_support - math.log10(support)
            sensitivity = class_id + 1
            fids.append(20.0 + sensitivity * gap)
            recalls.append(0.9 - 0.02 * sensitivity * gap)
        run_dir = (
            config.runs_dir
            / "rotations"
            / condition.condition_id
            / "steps_00002000"
            / "evaluation"
        )
        write_json(
            _report(
                macro_fid=float(sum(fids) / 10),
                accuracy=0.9,
                fid_by_class=fids,
                recall_by_class=recalls,
            ),
            run_dir / "metrics.json",
        )

    result = analyze_frequency_response(config)

    assert result["evidence"]["support_effect"]["supported"] is True
    assert all(
        item["strong_support"]
        for item in result["evidence"]["geometry_effect_by_representation"].values()
    )
    correlations = pd.read_csv(config.output_dir / "analysis" / "geometry_correlations.csv")
    assert len(correlations) == 10
