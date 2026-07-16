import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import fm_lab.diagnostics.long_tail_geometry.functional_calibration as calibration
from fm_lab.diagnostics.long_tail_geometry.functional_preregistration import (
    FunctionalCalibrationPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)


def _toy_functional_preregistration() -> FunctionalCalibrationPreregistration:
    canonical = FunctionalCalibrationPreregistration.load(
        "configs/fashion_mnist_lt/long_tail_geometry_functional_calibration.yaml"
    )
    return dataclasses.replace(
        canonical,
        relative_step_grid=(1e-4, 3e-4),
        random_controls=3,
        bootstrap_resamples=199,
    )


def _scale_rows(prereg, seed: int) -> pd.DataFrame:
    rows = []
    for layer in prereg.layers:
        for class_id in prereg.classes:
            for relative_step, benefit, doubled in (
                (1e-4, 0.004, 0.008),
                (3e-4, 0.010, 0.0195),
            ):
                rows.append(
                    {
                        "checkpoint_step": prereg.primary_checkpoint_step,
                        "layer": layer,
                        "seed": seed,
                        "class_id": class_id,
                        "relative_step": relative_step,
                        "benefit": benefit,
                        "doubled_benefit": doubled,
                    }
                )
    return pd.DataFrame(rows)


def _response_rows(prereg, seed: int, checkpoint_step: int) -> pd.DataFrame:
    rows = []
    for layer in prereg.layers:
        for direction_class in prereg.classes:
            for evaluation_class in prereg.classes:
                rows.append(
                    {
                        "checkpoint_step": checkpoint_step,
                        "layer": layer,
                        "seed": seed,
                        "direction_class": direction_class,
                        "evaluation_class": evaluation_class,
                        "direction_kind": "primary",
                        "control_id": -1,
                        "relative_step": 3e-4,
                        "projection_fraction": 0.3,
                        "base_loss": 1.0,
                        "perturbed_loss": 0.982 if direction_class == evaluation_class else 0.999,
                        "benefit": 0.018 if direction_class == evaluation_class else 0.001,
                    }
                )
                for control_id in range(prereg.random_controls):
                    rows.append(
                        {
                            "checkpoint_step": checkpoint_step,
                            "layer": layer,
                            "seed": seed,
                            "direction_class": direction_class,
                            "evaluation_class": evaluation_class,
                            "direction_kind": "random",
                            "control_id": control_id,
                            "relative_step": 3e-4,
                            "projection_fraction": 0.01,
                            "base_loss": 1.0,
                            "perturbed_loss": 0.997 if direction_class == evaluation_class else 1.004,
                            "benefit": 0.003 if direction_class == evaluation_class else -0.004,
                        }
                    )
    return pd.DataFrame(rows)


def test_noise_ceiling_scope_requires_locked_pair_at_both_checkpoints() -> None:
    functional = _toy_functional_preregistration()
    observation0 = Observation0Preregistration.load(
        "configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml"
    )
    pair = {
        "checkpoint_step": 500,
        "stratum_id": 0,
        "rank": 1,
        "layers": list(functional.layers),
        "common_classes": list(functional.classes),
    }
    noise = {
        "preregistration_digest": observation0.digest,
        "status": functional.required_observation0_status,
        "network_wide_gate_passed": True,
        "required_escalation": False,
        "next_action": functional.required_observation0_next_action,
        "probe_phase": "primary",
        "microbatches_per_cell": 16,
        "complete_seeds": [0, 1, 2],
        "incomplete_seeds": [],
        "passing_pairs": [
            pair,
            {**pair, "checkpoint_step": 1_000},
            {**pair, "checkpoint_step": 20_000},
        ],
    }

    calibration.validate_noise_ceiling_scope(noise, observation0, functional)

    changed = {**noise, "status": "output_layer_only"}
    with pytest.raises(ValueError, match="network-wide"):
        calibration.validate_noise_ceiling_scope(changed, observation0, functional)
    changed = {**noise, "passing_pairs": [pair]}
    with pytest.raises(ValueError, match="both locked checkpoints"):
        calibration.validate_noise_ceiling_scope(changed, observation0, functional)


def test_service_writes_digest_bound_lock_and_resumes_complete_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    functional = _toy_functional_preregistration()
    observation0 = Observation0Preregistration.load(
        "configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml"
    )
    artifact_dir = tmp_path / "aggregate/functional_calibration"
    context = calibration.CalibrationContext(
        study_dir=tmp_path,
        artifact_dir=artifact_dir,
        observation0=observation0,
        preregistration=functional,
        probe_a_manifest=None,
        run_dirs={seed: tmp_path / f"seed_{seed}" for seed in observation0.training_seeds},
        checkpoint_sha256={
            (seed, step): f"{seed}{step}".encode().hex().ljust(64, "0")[:64]
            for seed in observation0.training_seeds
            for step in functional.checkpoint_steps
        },
        input_digests={"noise_ceiling.json": "a" * 64},
    )
    calls = []
    monkeypatch.setattr(
        calibration,
        "prepare_calibration_context",
        lambda **kwargs: context,
    )

    def fake_scale(*, context, seed, device):
        calls.append(("scale", seed))
        return _scale_rows(functional, seed)

    def fake_responses(*, context, seed, checkpoint_step, relative_steps, device):
        calls.append(("response", seed, checkpoint_step, dict(relative_steps)))
        return _response_rows(functional, seed, checkpoint_step)

    monkeypatch.setattr(calibration, "collect_scale_chunk", fake_scale)
    monkeypatch.setattr(calibration, "collect_response_chunk", fake_responses)
    monkeypatch.setattr(calibration, "direction_index_digest", lambda _: "d" * 64)

    result = calibration.calibrate_observation0_functional_overlap(
        study_dir=tmp_path,
        calibration_preregistration_path=tmp_path / "input.yaml",
        device=torch.device("cpu"),
    )

    assert result.decision.stage1_unlocked
    assert len([call for call in calls if call[0] == "scale"]) == 3
    assert len([call for call in calls if call[0] == "response"]) == 6
    for name in (
        "preregistration.yaml",
        "scale_grid.csv",
        "responses.csv",
        "functional_lock.json",
        "complete.json",
    ):
        assert (artifact_dir / name).is_file()
    lock = json.loads((artifact_dir / "functional_lock.json").read_text())
    assert lock["stage1_unlocked"] is True
    assert lock["probe_view"] == "a"
    assert lock["probe_b_opened"] is False
    assert lock["direction_index_sha256"] == "d" * 64
    assert not (tmp_path / "stage1").exists()

    calls.clear()
    repeated = calibration.calibrate_observation0_functional_overlap(
        study_dir=tmp_path,
        calibration_preregistration_path=tmp_path / "input.yaml",
        device=torch.device("cpu"),
    )
    assert repeated.decision == result.decision
    assert calls == []


def test_complete_service_rejects_tampered_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_service_writes_digest_bound_lock_and_resumes_complete_result(
        tmp_path,
        monkeypatch,
    )
    artifact_dir = tmp_path / "aggregate/functional_calibration"
    with (artifact_dir / "responses.csv").open("a") as handle:
        handle.write("tampered\n")

    with pytest.raises(ValueError, match="changed after completion"):
        calibration.calibrate_observation0_functional_overlap(
            study_dir=tmp_path,
            calibration_preregistration_path=tmp_path / "input.yaml",
            device=torch.device("cpu"),
        )
