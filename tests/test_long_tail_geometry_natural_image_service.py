import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import fm_lab.diagnostics.long_tail_geometry.natural_image as natural
from fm_lab.diagnostics.long_tail_geometry.functional_audit import (
    FunctionalGeometryAuditChunk,
)
from fm_lab.diagnostics.long_tail_geometry.natural_image_preregistration import (
    NaturalImageTransportPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)

CLASS_RANKS = (0, 3, 6, 9, 2, 5, 8, 1, 4, 7)
CLASS_COUNTS = (1000, 215, 46, 10, 359, 77, 16, 599, 129, 27)


def _preregistration() -> NaturalImageTransportPreregistration:
    canonical = NaturalImageTransportPreregistration.load(
        "configs/cifar10_lt/long_tail_geometry_natural_image_transport.yaml"
    )
    return dataclasses.replace(canonical, bootstrap_resamples=199)


def _reliability(prereg: NaturalImageTransportPreregistration) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "training_seed": seed,
                "checkpoint_step": prereg.primary_checkpoint_step,
                "stratum_id": prereg.stratum_id,
                "class_id": class_id,
                "layer_name": layer,
                "representation": "centered_covariance",
                "rank": prereg.rank,
                "measurable": True,
            }
            for layer in prereg.layers
            for seed in (0, 1, 2)
            for class_id in prereg.classes
        ]
    )


def _synthetic_chunk(
    prereg: NaturalImageTransportPreregistration,
    seed: int,
    checkpoint_step: int,
) -> FunctionalGeometryAuditChunk:
    base_loss = {
        prereg.baseline_checkpoint_step: 1.0,
        prereg.early_checkpoint_step: 0.7,
        prereg.primary_checkpoint_step: 0.5,
    }[checkpoint_step]
    slopes = []
    finite = []
    basis = []
    for layer in prereg.layers:
        for fold in range(len(prereg.fold_offsets)):
            for direction_class in prereg.classes:
                for basis_kind, target_slope in (
                    ("raw", 0.2 + 0.005 * direction_class),
                    ("row_normalized", 1.0 + 0.005 * direction_class),
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
                            "raw_normalized_basis_abs_cosine": 0.9,
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
                                "base_loss": base_loss,
                                "parameter_norm": 2.0,
                                "benefit_slope": (
                                    target_slope
                                    if evaluation_class == direction_class
                                    else -0.02 - 0.001 * evaluation_class
                                ),
                            }
                        )
                    for partition in ("scale", "evaluation"):
                        predicted = target_slope + (
                            0.1 if partition == "scale" else 0.0
                        )
                        for step in prereg.relative_step_grid:
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
                                    "base_loss": base_loss,
                                    "perturbed_loss": base_loss * (1.0 - benefit),
                                    "benefit": benefit,
                                    "predicted_slope": predicted,
                                }
                            )
    return FunctionalGeometryAuditChunk(
        slopes=pd.DataFrame(slopes),
        finite_steps=pd.DataFrame(finite),
        basis_comparison=pd.DataFrame(basis),
    )


def _context(
    tmp_path: Path,
    prereg: NaturalImageTransportPreregistration,
) -> natural.NaturalImageTransportContext:
    return natural.NaturalImageTransportContext(
        study_dir=tmp_path,
        artifact_dir=(
            tmp_path / "aggregate/natural_image_transport_falsification"
        ),
        observation0=SimpleNamespace(training_seeds=(0, 1, 2)),
        preregistration=prereg,
        probe_a_manifest=None,
        run_dirs={},
        checkpoint_sha256={},
        input_digests={"reliability.csv": "f" * 64},
        reliability=_reliability(prereg),
        class_counts=CLASS_COUNTS,
        class_ranks=CLASS_RANKS,
    )


def test_natural_image_service_writes_digest_bound_artifacts_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prereg = _preregistration()
    context = _context(tmp_path, prereg)
    calls = []
    monkeypatch.setattr(
        natural,
        "prepare_natural_image_transport_context",
        lambda **kwargs: context,
    )

    def fake_collect(*, context, seed, checkpoint_step, device):
        calls.append((seed, checkpoint_step))
        return _synthetic_chunk(prereg, seed, checkpoint_step)

    monkeypatch.setattr(natural, "collect_natural_image_transport_chunk", fake_collect)

    result = natural.run_natural_image_transport_falsification(
        study_dir=tmp_path,
        preregistration_path=tmp_path / "transport.yaml",
        device=torch.device("cpu"),
    )

    assert result.decision.status == "natural_image_transport_confirmed"
    assert result.decision.stage1_unlocked is False
    assert result.decision.method_opened is False
    assert calls == [
        (seed, checkpoint)
        for seed in (0, 1, 2)
        for checkpoint in prereg.checkpoint_steps
    ]
    for name in (
        "preregistration.yaml",
        "slopes.csv",
        "finite_steps.csv",
        "basis_comparison.csv",
        "class_transport.csv",
        "frequency_associations.csv",
        "interference_matrices.npz",
        "falsification_summary.json",
        "complete.json",
    ):
        assert (context.artifact_dir / name).is_file()
    summary = json.loads(
        (context.artifact_dir / "falsification_summary.json").read_text()
    )
    assert summary["stage1_unlocked"] is False
    assert summary["method_opened"] is False
    assert summary["probe_b_used_for_transport"] is False
    assert summary["status"] == "natural_image_transport_confirmed"
    assert len(summary["reliable_common_classes"]) == 10
    assert not (tmp_path / "stage1").exists()

    calls.clear()
    repeated = natural.run_natural_image_transport_falsification(
        study_dir=tmp_path,
        preregistration_path=tmp_path / "transport.yaml",
        device=torch.device("cpu"),
    )
    assert repeated.decision == result.decision
    assert calls == []


def test_completed_natural_image_service_rejects_tampered_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_natural_image_service_writes_digest_bound_artifacts_and_resumes(
        tmp_path,
        monkeypatch,
    )
    path = (
        tmp_path
        / "aggregate/natural_image_transport_falsification/class_transport.csv"
    )
    path.write_text(path.read_text() + "tampered\n")

    with pytest.raises(ValueError, match="changed after completion"):
        natural.run_natural_image_transport_falsification(
            study_dir=tmp_path,
            preregistration_path=tmp_path / "transport.yaml",
            device=torch.device("cpu"),
        )


def test_natural_image_service_rejects_partial_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prereg = _preregistration()
    context = _context(tmp_path, prereg)
    monkeypatch.setattr(
        natural,
        "prepare_natural_image_transport_context",
        lambda **kwargs: context,
    )
    partial = context.artifact_dir / "chunks/seed_0_checkpoint_000000"
    partial.mkdir(parents=True)

    with pytest.raises(ValueError, match="partial chunk"):
        natural.run_natural_image_transport_falsification(
            study_dir=tmp_path,
            preregistration_path=tmp_path / "transport.yaml",
            device=torch.device("cpu"),
        )


def test_prepare_natural_image_context_rejects_non_cifar_observation(
    tmp_path: Path,
) -> None:
    fashion = Observation0Preregistration.load(
        "configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml"
    )
    fashion.lock(tmp_path / "aggregate/preregistration.yaml")

    with pytest.raises(ValueError, match="CIFAR-10-LT"):
        natural.prepare_natural_image_transport_context(
            study_dir=tmp_path,
            preregistration_path=(
                "configs/cifar10_lt/long_tail_geometry_natural_image_transport.yaml"
            ),
        )
