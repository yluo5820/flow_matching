from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import fm_lab.geometry_explorer.synthetic_factor_oracle as oracle_module
import fm_lab.geometry_explorer.synthetic_long_tail_metrics as metrics_module
from fm_lab.experiments.synthetic_long_tail_geometry import (
    RunLedger,
    StageBlockedError,
    SyntheticLongTailRunner,
    _balanced_pilot_gate,
    _bounded_rotation_followup_summary,
    _frequency_factorial_summary,
    _paired_local_geometry_summary,
    build_matrix_commands,
    require_gate,
)
from fm_lab.utils.config import load_config


def test_matrix_dry_run_lists_exactly_36_training_commands(tmp_path: Path) -> None:
    config_paths = {
        replicate: tuple(
            tmp_path / f"rep{replicate:02d}" / f"condition_{condition:02d}.yaml"
            for condition in range(12)
        )
        for replicate in range(3)
    }

    commands = build_matrix_commands(config_paths, run_root=tmp_path / "runs")

    assert len(commands) == 36
    assert len({command.condition_id for command in commands}) == 12
    assert len({command.replicate for command in commands}) == 3
    assert all(command.argv("cpu")[0] == sys.executable for command in commands)
    assert len({command.run_dir for command in commands}) == 36


def test_v2_config_uses_isolated_artifact_and_training_roots() -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    assert runner.output_root.name == "synthetic_long_tail_geometry_v2"
    assert runner.run_root.name == "synthetic_long_tail_geometry_v2"
    assert runner.config["pilot"]["training_steps"] == 1000
    assert runner.config["pilot"]["batch_size"] == 64


def test_balanced_learning_curve_budget_has_isolated_paths() -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    result = runner.balanced_pilots(device="cpu", dry_run=True, training_steps=2_000)

    assert result["training_steps"] == 2_000
    assert len(result["commands"]) == 3
    assert {item["condition_id"] for item in result["commands"]} == {
        "g0_balanced",
        "g1_balanced",
        "g2_balanced",
    }
    assert all(
        "balanced_learning_curve/steps_00002000" in item["run_dir"] for item in result["commands"]
    )
    assert all(
        "balanced_learning_curve/steps_00002000" in item["config_path"]
        for item in result["commands"]
    )


@pytest.mark.parametrize("training_steps", [0, -1, True])
def test_balanced_learning_curve_rejects_invalid_budget(training_steps: object) -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    with pytest.raises(ValueError, match="training_steps"):
        runner.balanced_pilots(
            device="cpu",
            dry_run=True,
            training_steps=training_steps,  # type: ignore[arg-type]
        )


def test_frequency_factorial_dry_run_lists_nine_isolated_conditions() -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    result = runner.frequency_pilots(device="cpu", dry_run=True, training_steps=5_000)

    assert result["training_steps"] == 5_000
    assert len(result["commands"]) == 9
    assert {item["condition_id"] for item in result["commands"]} == {
        f"g{geometry}_f{frequency}" for geometry in range(3) for frequency in range(3)
    }
    assert all(
        "frequency_factorial/steps_00005000" in item["run_dir"] for item in result["commands"]
    )

    balanced_result = runner.frequency_pilots(
        device="cpu",
        dry_run=True,
        training_steps=5_000,
        training_sampling_policy="class_balanced",
    )
    assert balanced_result["training_sampling_policy"] == "class_balanced"
    assert all(
        "frequency_factorial_class_balanced/steps_00005000" in item["run_dir"]
        for item in balanced_result["commands"]
    )


def test_frequency_factorial_rejects_unknown_training_sampling_policy() -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    with pytest.raises(ValueError, match="training_sampling_policy"):
        runner.frequency_pilots(
            device="cpu",
            dry_run=True,
            training_steps=5_000,
            training_sampling_policy="unknown",
        )


def test_bounded_rotation_control_dry_run_is_one_paired_2000_step_command() -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    result = runner.bounded_rotation_control(device="cpu", dry_run=True)

    assert result["training_steps"] == 2_000
    assert result["command"]["condition_id"] == "g0_balanced_bounded_azimuth"
    assert "bounded_rotation_control/steps_00002000" in result["command"]["run_dir"]
    assert "balanced_learning_curve/steps_00002000" in result["baseline_evaluation"]


@pytest.mark.parametrize("training_steps", [0, -1, True])
def test_bounded_rotation_control_rejects_invalid_budget(training_steps: object) -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    with pytest.raises(ValueError, match="training_steps"):
        runner.bounded_rotation_control(
            device="cpu",
            dry_run=True,
            training_steps=training_steps,  # type: ignore[arg-type]
        )


def test_bounded_rotation_followups_dry_run_lists_only_four_targeted_runs() -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    result = runner.bounded_rotation_followups(device="cpu", dry_run=True)

    assert result["training_steps"] == 2_000
    assert len(result["commands"]) == 4
    assert [item["training_sampling_policy"] for item in result["commands"]] == [
        "empirical",
        "empirical",
        "empirical",
        "class_balanced",
    ]
    assert [item["condition_id"] for item in result["commands"]].count(
        "g0_bounded_azimuth_tail"
    ) == 2
    assert all("steps_00002000" in item["run_dir"] for item in result["commands"])


def test_bounded_rotation_memorization_dry_run_reuses_completed_tail_run() -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    result = runner.bounded_rotation_memorization(device="cpu", dry_run=True)

    assert result["training_steps"] == 2_000
    assert result["requested_class"] == 0
    assert result["training_unique_count"] == 50
    assert result["retraining"] is False
    assert "bounded_rotation_frequency_slice_class_balanced" in result["generated_run"]
    assert result["output_dir"].endswith("memorization_bounded_5d_tail_v2")


def test_bounded_rotation_geometry_dry_run_is_paired_and_does_not_retrain() -> None:
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")

    result = runner.bounded_rotation_geometry(device="cpu", dry_run=True)

    assert result["query_count"] == 8
    assert result["num_directions"] == 16
    assert result["nfe"] == 32
    assert result["t_values"] == [0.8, 0.9]
    assert result["retraining"] is False
    assert "bounded_rotation_control" in result["head_5000_run"]
    assert "frequency_slice_class_balanced" in result["tail_50_class_balanced_run"]


def test_paired_local_geometry_summary_keeps_query_pairing_and_delta_direction() -> None:
    head = pd.DataFrame(
        {
            "query_index": [0, 1],
            "class_id": [0, 0],
            "time": [0.8, 0.8],
            "renderer_rank": [5, 5],
            "participation_rank": [5.0, 4.0],
            "principal_angle_mean": [0.2, 0.4],
            "alignment_camera_elevation": [0.8, 0.6],
        }
    )
    tail = head.copy()
    tail["participation_rank"] -= 1.0
    tail["principal_angle_mean"] += 0.1
    tail["alignment_camera_elevation"] -= 0.2

    result = _paired_local_geometry_summary(
        head=head,
        tail=tail,
        metadata={"schema_version": 1},
    )

    comparison = result["paired_comparisons"][0]
    assert comparison["paired_query_count"] == 2
    assert comparison["metrics"]["participation_rank"]["tail_minus_head_mean"] == (
        pytest.approx(-1.0)
    )
    assert comparison["metrics"]["principal_angle_mean"]["tail_minus_head_mean"] == (
        pytest.approx(0.1)
    )
    assert comparison["metrics"]["alignment_camera_elevation"][
        "tail_minus_head_mean"
    ] == pytest.approx(-0.2)


def test_bounded_rotation_followup_summary_separates_frequency_and_exposure() -> None:
    objects = ("stepped_monument", "crooked_arch", "three_arm_vane")

    def evaluation(error: float) -> dict[str, object]:
        return {
            "classes": [
                {
                    "requested_class": class_id,
                    "object_id": object_id,
                    "target_dimension_id": "high_bounded_azimuth",
                    "true_dimension": 5,
                    "validity": {
                        "class_leakage_rate": 0.0,
                        "off_renderer_rate": error,
                        "joint_valid_rate": 1.0 - error,
                    },
                    "all_requested": {
                        "metrics": {
                            "active_factors": {
                                "multivariate_energy_distance": error,
                            },
                            "oracle_feature_fid": error,
                        }
                    },
                }
                for class_id, object_id in enumerate(objects)
            ]
        }

    evaluations = {
        "g2_full_empirical": evaluation(0.8),
        "g2_bounded_empirical": evaluation(0.2),
        "g0_head_empirical": evaluation(0.1),
        "g0_medium_empirical": evaluation(0.3),
        "g0_tail_empirical": evaluation(0.7),
        "g0_tail_class_balanced": evaluation(0.4),
    }
    counts = {
        "g0_head_empirical": (5_000, 5_000, 5_000),
        "g0_medium_empirical": (500, 5_000, 5_000),
        "g0_tail_empirical": (50, 5_000, 5_000),
        "g0_tail_class_balanced": (50, 5_000, 5_000),
    }

    summary = _bounded_rotation_followup_summary(
        evaluations=evaluations,
        condition_counts=counts,
        followup_spec={"schema_version": 1},
        training_steps=2_000,
    )

    assert summary["object_replication"]["bounded_minus_full"]["joint_valid_rate"] == (
        pytest.approx(0.6)
    )
    assert summary["frequency_slice"]["tail_empirical_minus_head"][
        "joint_valid_rate"
    ] == pytest.approx(-0.6)
    assert summary["frequency_slice"]["tail_balanced_minus_tail_empirical"][
        "joint_valid_rate"
    ] == pytest.approx(0.3)


def test_frequency_factorial_summary_computes_paired_frequency_changes() -> None:
    objects = ("stepped_monument", "crooked_arch", "three_arm_vane")
    dimensions = ((5, 3, 1), (3, 1, 5), (1, 5, 3))
    dimension_ids = {1: "low", 3: "medium", 5: "high"}
    count_rotations = {
        "balanced": (5_000, 5_000, 5_000),
        "f0": (5_000, 500, 50),
        "f1": (500, 50, 5_000),
        "f2": (50, 5_000, 500),
    }
    role_offset = {"balanced": 0.0, "head": 0.01, "medium": 0.02, "tail": 0.03}
    evaluations = {}
    condition_counts = {}
    for geometry in range(3):
        for frequency, counts in count_rotations.items():
            condition_id = f"g{geometry}_{frequency}"
            condition_counts[condition_id] = counts
            if frequency == "balanced":
                roles = ("balanced",) * 3
            else:
                roles = tuple({5_000: "head", 500: "medium", 50: "tail"}[n] for n in counts)
            classes = []
            for class_id, (object_id, dimension, role) in enumerate(
                zip(objects, dimensions[geometry], roles, strict=True)
            ):
                error = dimension / 10.0 + role_offset[role]
                classes.append(
                    {
                        "requested_class": class_id,
                        "object_id": object_id,
                        "target_dimension_id": dimension_ids[dimension],
                        "true_dimension": dimension,
                        "validity": {
                            "class_leakage_rate": 0.0,
                            "off_renderer_rate": error,
                            "joint_valid_rate": 1.0 - error,
                        },
                        "all_requested": {
                            "metrics": {
                                "active_factors": {
                                    "multivariate_energy_distance": error,
                                },
                                "oracle_feature_fid": error,
                            }
                        },
                    }
                )
            evaluations[condition_id] = {"classes": classes}

    summary = _frequency_factorial_summary(
        evaluations=evaluations,
        condition_counts=condition_counts,
        training_steps=5_000,
    )

    assert len(summary["records"]) == 36
    assert summary["means_by_true_dimension_and_frequency_role"]["5"]["tail"][
        "off_renderer_rate"
    ] == pytest.approx(0.53)
    assert summary["mean_changes_from_balanced_by_true_dimension"]["5"]["tail"][
        "off_renderer_rate"
    ] == pytest.approx(0.03)


def test_balanced_pilot_gate_checks_learning_and_each_class(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot"
    (run_dir / "diagnostics").mkdir(parents=True)
    (run_dir / "metrics.json").write_text(json.dumps({"trained_steps": 100}), encoding="utf-8")
    (run_dir / "diagnostics" / "training_history.csv").write_text(
        "step,loss\n1,1.0\n20,0.9\n40,0.8\n60,0.6\n80,0.5\n100,0.4\n",
        encoding="utf-8",
    )
    evaluation = {
        "classes": [
            {
                "requested_class": class_id,
                "validity": {
                    "class_leakage_rate": 0.05,
                    "off_renderer_rate": 0.1,
                    "joint_valid_rate": 0.85,
                },
            }
            for class_id in range(3)
        ]
    }
    pilot = {
        "training_steps": 100,
        "max_final_to_initial_loss_ratio": 0.8,
        "max_class_leakage_rate": 0.25,
        "max_off_renderer_rate": 0.5,
        "min_joint_valid_rate": 0.4,
    }

    passed = _balanced_pilot_gate(run_dir=run_dir, evaluation=evaluation, pilot=pilot)
    assert passed["passed"] is True
    assert passed["loss"]["final_to_initial_ratio"] < 0.8

    evaluation["classes"][2]["validity"]["class_leakage_rate"] = 0.5
    failed = _balanced_pilot_gate(run_dir=run_dir, evaluation=evaluation, pilot=pilot)
    assert failed["passed"] is False
    assert failed["reasons"] == ["class_2:class_leakage"]


def test_train_oracle_reuses_qualified_checkpoint_and_reads_control_ordering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/synthetic_long_tail_geometry/experiment_v2.yaml")
    config["output_root"] = str(tmp_path / "outputs")
    config["run_root"] = str(tmp_path / "runs")
    config["evaluation"]["samples_per_class"] = 3
    runner = SyntheticLongTailRunner("configs/synthetic_long_tail_geometry/experiment_v2.yaml")
    runner.config = config
    runner.output_root = tmp_path / "outputs"
    runner.run_root = tmp_path / "runs"
    runner.ledger = RunLedger(runner.output_root / "run_ledger.json")
    calibration = tmp_path / "outputs" / "calibration"
    oracle_dir = calibration / "oracle"
    oracle_dir.mkdir(parents=True)
    (calibration / "renderer").mkdir()
    (calibration / "renderer" / "renderer_gate.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (oracle_dir / "oracle_gate.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    (oracle_dir / "factor_oracle.pt").write_bytes(b"existing checkpoint")

    def unexpected_training(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("an existing qualified oracle must not be retrained")

    observed: dict[str, object] = {}

    def fake_controls(
        *,
        oracle_checkpoint: Path,
        oracle_gate: Path,
        output_dir: Path,
        device: str,
        samples_per_class: int,
        seed: int,
        source_revision: str,
    ) -> dict[str, object]:
        observed.update(
            checkpoint=oracle_checkpoint,
            gate=oracle_gate,
            output_dir=output_dir,
            device=device,
            samples_per_class=samples_per_class,
            seed=seed,
            source_revision=source_revision,
        )
        return {"control_ordering": {"passed": True}}

    monkeypatch.setattr(oracle_module, "train_factor_oracle", unexpected_training)
    monkeypatch.setattr(metrics_module, "calibrate_metric_controls", fake_controls)

    result = runner.train_oracle(device="cpu")

    assert result["metric_controls"]["control_ordering"]["passed"] is True
    assert observed["checkpoint"] == oracle_dir / "factor_oracle.pt"
    assert observed["samples_per_class"] == 3
    metric_gate = json.loads((calibration / "metric_gate.json").read_text(encoding="utf-8"))
    assert metric_gate["passed"] is True


def test_failed_gate_blocks_training(tmp_path: Path) -> None:
    gate_path = tmp_path / "renderer_gate.json"
    gate_path.write_text(
        json.dumps({"passed": False, "reasons": ["renderer rank"]}), encoding="utf-8"
    )

    with pytest.raises(StageBlockedError, match="renderer rank"):
        require_gate(gate_path, stage="renderer_calibration")


def test_gate_requires_literal_boolean_pass(tmp_path: Path) -> None:
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({"passed": 1}), encoding="utf-8")

    with pytest.raises(StageBlockedError, match="literal true"):
        require_gate(gate_path, stage="oracle")


def test_completed_ledger_entry_is_not_overwritten(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "run_ledger.json")
    ledger.complete("rep00_g0_f0", {"metrics": "first.json"})

    with pytest.raises(FileExistsError, match="rep00_g0_f0"):
        ledger.complete("rep00_g0_f0", {"metrics": "second.json"})

    payload = json.loads((tmp_path / "run_ledger.json").read_text(encoding="utf-8"))
    assert payload["entries"][0]["artifacts"] == {"metrics": "first.json"}


def test_ledger_resume_requires_matching_config_hash(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "run_ledger.json")
    ledger.complete("rep00_g0_f0", {}, config_hash="abc")

    assert ledger.is_complete("rep00_g0_f0", config_hash="abc") is True
    assert ledger.is_complete("rep00_g0_f0", config_hash="different") is False
