import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from long_tail_geometry_helpers import write_geometry_toy_checkpoint

import fm_lab.experiments.run_long_tail_geometry_stage0 as stage0_module
from fm_lab.experiments.run_long_tail_geometry_stage0 import (
    Stage0ValidationError,
    run_stage0_validation,
)


def test_stage0_validator_writes_passing_report_for_valid_pipeline(
    tmp_path: Path,
) -> None:
    config, checkpoint_path = write_geometry_toy_checkpoint(tmp_path)

    report = run_stage0_validation(
        config=config,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "out",
        device=torch.device("cpu"),
    )

    artifact_dir = tmp_path / "out/diagnostics/long_tail_geometry"
    persisted = json.loads((artifact_dir / "stage0_report.json").read_text())
    assert report["passed"] is True
    assert persisted == report
    assert list(report["checks"]) == [
        "config_fence",
        "frequency_mappings",
        "paired_probe_manifests",
        "checkpoint_replay",
        "gradient_sketch_fidelity",
        "permutation_nulls",
        "planted_low_rank_control",
    ]
    assert all(check["passed"] for check in report["checks"].values())
    assert (artifact_dir / "probe_a.npz").exists()
    assert (artifact_dir / "probe_b.npz").exists()
    assert len(list(artifact_dir.glob("gradient_rows_*.npz"))) == 2


def test_stage0_validator_fails_before_gradients_when_pairing_breaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, checkpoint_path = write_geometry_toy_checkpoint(tmp_path)
    output_dir = tmp_path / "out"
    real_materialize = stage0_module.materialize_probe_batch

    def corrupt_second_mapping(*args, **kwargs):
        batch = real_materialize(*args, **kwargs)
        if getattr(args[0], "frequency_mapping_offset", None) == 7:
            return dataclasses.replace(batch, x1=batch.x1 + 0.01)
        return batch

    monkeypatch.setattr(
        stage0_module,
        "materialize_probe_batch",
        corrupt_second_mapping,
    )

    with pytest.raises(Stage0ValidationError, match="paired probe tuples"):
        run_stage0_validation(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=torch.device("cpu"),
        )

    report_path = output_dir / "diagnostics/long_tail_geometry/stage0_report.json"
    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert report["checks"]["paired_probe_manifests"]["passed"] is False
    assert not list(report_path.parent.glob("gradient_rows_*.npz"))


def test_stage0_validator_rejects_cm_at_config_fence(tmp_path: Path) -> None:
    config, checkpoint_path = write_geometry_toy_checkpoint(tmp_path)
    config = copy.deepcopy(config)
    config["objective"]["modifiers"] = [
        {
            "name": "cm",
            "consistency_weight": 1.0,
            "diversity_weight": 0.2,
            "comparison_space": "target",
        }
    ]
    output_dir = tmp_path / "out"

    with pytest.raises(Stage0ValidationError, match="including CM"):
        run_stage0_validation(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            device=torch.device("cpu"),
        )

    report_path = output_dir / "diagnostics/long_tail_geometry/stage0_report.json"
    report = json.loads(report_path.read_text())
    assert report["checks"]["config_fence"]["passed"] is False
    assert list(report["checks"]) == ["config_fence"]
    assert not list(report_path.parent.glob("gradient_rows_*.npz"))
