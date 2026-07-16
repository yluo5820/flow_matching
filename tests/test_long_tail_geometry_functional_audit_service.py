import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import fm_lab.diagnostics.long_tail_geometry.functional_audit as audit
from fm_lab.diagnostics.long_tail_geometry.functional_audit_preregistration import (
    FunctionalGeometryAuditPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.manifest import ProbeBatch


def _preregistration() -> FunctionalGeometryAuditPreregistration:
    canonical = FunctionalGeometryAuditPreregistration.load(
        "configs/fashion_mnist_lt/long_tail_geometry_functional_audit.yaml"
    )
    return dataclasses.replace(canonical, bootstrap_resamples=199)


def test_blocked_functional_lock_is_required() -> None:
    prereg = _preregistration()
    lock = {
        "stage1_unlocked": False,
        "probe_view": "a",
        "probe_b_opened": False,
        "next_action": "stop_stage1_and_revise_functional_geometry",
        "functional_preregistration_sha256": (
            prereg.functional_preregistration_sha256
        ),
    }

    audit.validate_blocked_functional_lock(lock, prereg)

    for field, value in (
        ("stage1_unlocked", True),
        ("probe_view", "b"),
        ("probe_b_opened", True),
        ("next_action", "stage1_unlocked_for_separate_preregistration"),
        ("functional_preregistration_sha256", "0" * 64),
    ):
        changed = {**lock, field: value}
        with pytest.raises(ValueError, match="blocked functional calibration"):
            audit.validate_blocked_functional_lock(changed, prereg)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv2 = nn.Linear(2, 2, bias=False)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down2_block = _Block()
        self.middle = _Block()
        with torch.no_grad():
            self.down2_block.conv2.weight.copy_(
                torch.tensor([[0.8, -0.3], [0.2, 0.6]])
            )
            self.middle.conv2.weight.copy_(
                torch.tensor([[0.5, 0.1], [-0.4, 0.9]])
            )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.down2_block.conv2(values))
        return self.middle.conv2(hidden)


class _Objective:
    modifiers: tuple[()] = ()

    def __call__(self, *, model, x0, x1, **kwargs):
        prediction = model(x0)
        return (prediction - x1).square().mean(), {}


def _batches(prereg) -> dict[int, tuple[ProbeBatch, ...]]:
    result = {}
    for class_id in prereg.classes:
        batches = []
        for position in range(prereg.microbatches_per_cell):
            angle = 0.17 * (position + 1) + 0.03 * class_id
            x0 = torch.tensor(
                [[np.cos(angle), np.sin(angle)]],
                dtype=torch.float32,
            )
            x1 = torch.tensor(
                [[0.2 * class_id + np.sin(1.7 * angle), np.cos(0.7 * angle)]],
                dtype=torch.float32,
            )
            batches.append(
                ProbeBatch(
                    x0=x0,
                    x1=x1,
                    t=torch.tensor([[0.05]], dtype=torch.float32),
                    labels=torch.tensor([class_id], dtype=torch.long),
                    original_indices=np.asarray([position], dtype=np.int64),
                    stratum_ids=np.asarray([0], dtype=np.int64),
                    microbatch_ids=np.asarray([position], dtype=np.int64),
                )
            )
        result[class_id] = tuple(batches)
    return result


def test_collect_audit_metrics_pairs_bases_and_restores_model() -> None:
    prereg = _preregistration()
    model = _TinyModel()
    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    chunk = audit.collect_audit_metrics(
        model=model,
        objective=_Objective(),
        path=None,
        batches_by_class=_batches(prereg),
        preregistration=prereg,
        seed=0,
        checkpoint_step=prereg.primary_checkpoint_step,
    )

    expected_slopes = 2 * 2 * 4 * 6 * 6
    expected_finite = 2 * 2 * 4 * 6 * 2 * 3
    expected_basis = 2 * 2 * 4 * 6
    assert len(chunk.slopes) == expected_slopes
    assert len(chunk.finite_steps) == expected_finite
    assert len(chunk.basis_comparison) == expected_basis
    paired = chunk.basis_comparison.pivot(
        index=["layer", "fold", "direction_class"],
        columns="basis_kind",
        values="orientation_gradient_sha256",
    )
    assert (paired["raw"] == paired["row_normalized"]).all()
    assert set(chunk.finite_steps["partition"]) == {"scale", "evaluation"}
    smallest = chunk.finite_steps[
        chunk.finite_steps["relative_step"] == min(prereg.relative_step_grid)
    ]
    sign_match = np.sign(smallest["benefit"]) == np.sign(
        smallest["predicted_slope"]
    )
    assert float(sign_match.mean()) > 0.9
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, original[name])


def _synthetic_chunk(
    prereg: FunctionalGeometryAuditPreregistration,
    seed: int,
    checkpoint_step: int,
) -> audit.FunctionalGeometryAuditChunk:
    slopes = []
    finite = []
    basis = []
    for layer in prereg.layers:
        for fold in range(len(prereg.fold_offsets)):
            for direction_class in prereg.classes:
                for basis_kind, target_slope in (
                    ("raw", 0.1),
                    ("row_normalized", 1.0),
                ):
                    basis.append(
                        {
                            "checkpoint_step": checkpoint_step,
                            "layer": layer,
                            "seed": seed,
                            "fold": fold,
                            "basis_kind": basis_kind,
                            "direction_class": direction_class,
                            "fit_explained_fraction": 0.7,
                            "projection_fraction": 0.5,
                            "fit_row_norm_cv": 0.3,
                            "basis_vector_sha256": "a" * 64,
                            "direction_vector_sha256": "b" * 64,
                            "orientation_gradient_sha256": "c" * 64,
                            "raw_normalized_basis_abs_cosine": 0.2,
                        }
                    )
                    for evaluation_class in prereg.classes:
                        slopes.append(
                            {
                                "checkpoint_step": checkpoint_step,
                                "layer": layer,
                                "seed": seed,
                                "fold": fold,
                                "basis_kind": basis_kind,
                                "direction_class": direction_class,
                                "evaluation_class": evaluation_class,
                                "base_loss": 1.0,
                                "parameter_norm": 2.0,
                                "benefit_slope": (
                                    target_slope
                                    if evaluation_class == direction_class
                                    else 0.05
                                ),
                            }
                        )
                    for partition in ("scale", "evaluation"):
                        for step in prereg.relative_step_grid:
                            predicted = target_slope + (
                                0.1 if partition == "scale" else 0.0
                            )
                            benefit = step * predicted
                            finite.append(
                                {
                                    "checkpoint_step": checkpoint_step,
                                    "layer": layer,
                                    "seed": seed,
                                    "fold": fold,
                                    "basis_kind": basis_kind,
                                    "direction_class": direction_class,
                                    "partition": partition,
                                    "relative_step": step,
                                    "base_loss": 1.0,
                                    "perturbed_loss": 1.0 - benefit,
                                    "benefit": benefit,
                                    "predicted_slope": predicted,
                                }
                            )
    return audit.FunctionalGeometryAuditChunk(
        slopes=pd.DataFrame(slopes),
        finite_steps=pd.DataFrame(finite),
        basis_comparison=pd.DataFrame(basis),
    )


def test_audit_service_writes_digest_bound_artifacts_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prereg = _preregistration()
    artifact_dir = tmp_path / "aggregate/functional_geometry_audit"
    context = audit.FunctionalGeometryAuditContext(
        study_dir=tmp_path,
        artifact_dir=artifact_dir,
        calibration=SimpleNamespace(
            observation0=SimpleNamespace(training_seeds=(0, 1, 2))
        ),
        preregistration=prereg,
        input_digests={"functional_calibration/functional_lock.json": "f" * 64},
        functional_lock_sha256="f" * 64,
    )
    calls = []
    monkeypatch.setattr(
        audit,
        "prepare_functional_geometry_audit_context",
        lambda **kwargs: context,
    )

    def fake_collect(*, context, seed, checkpoint_step, device):
        calls.append((seed, checkpoint_step))
        return _synthetic_chunk(prereg, seed, checkpoint_step)

    monkeypatch.setattr(audit, "collect_audit_chunk", fake_collect)

    result = audit.run_functional_geometry_audit(
        study_dir=tmp_path,
        audit_preregistration_path=tmp_path / "audit.yaml",
        device=torch.device("cpu"),
    )

    assert result.decision.status == "normalized_representation_rescue"
    assert result.decision.stage1_unlocked is False
    assert len(calls) == 6
    for name in (
        "preregistration.yaml",
        "slopes.csv",
        "finite_steps.csv",
        "basis_comparison.csv",
        "audit_summary.json",
        "complete.json",
    ):
        assert (artifact_dir / name).is_file()
    summary = json.loads((artifact_dir / "audit_summary.json").read_text())
    assert summary["stage1_unlocked"] is False
    assert summary["probe_b_opened"] is False
    assert summary["original_functional_lock_remains_blocked"] is True
    assert summary["functional_lock_sha256"] == "f" * 64
    assert not (tmp_path / "stage1").exists()

    calls.clear()
    repeated = audit.run_functional_geometry_audit(
        study_dir=tmp_path,
        audit_preregistration_path=tmp_path / "audit.yaml",
        device=torch.device("cpu"),
    )
    assert repeated.decision == result.decision
    assert calls == []


def test_completed_audit_rejects_tampered_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_audit_service_writes_digest_bound_artifacts_and_resumes(
        tmp_path,
        monkeypatch,
    )
    path = tmp_path / "aggregate/functional_geometry_audit/slopes.csv"
    path.write_text(path.read_text() + "tampered\n")

    with pytest.raises(ValueError, match="changed after completion"):
        audit.run_functional_geometry_audit(
            study_dir=tmp_path,
            audit_preregistration_path=tmp_path / "audit.yaml",
            device=torch.device("cpu"),
        )


def test_file_digest_helper_is_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("first")
    first = audit.file_sha256(path)
    path.write_text("second")
    second = audit.file_sha256(path)

    assert first == hashlib.sha256(b"first").hexdigest()
    assert first != second
