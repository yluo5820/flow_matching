import dataclasses
from pathlib import Path

import pytest

from fm_lab.diagnostics.long_tail_geometry.functional_audit_preregistration import (
    FunctionalGeometryAuditPreregistration,
)


AUDIT_CONFIG = (
    "configs/fashion_mnist_lt/long_tail_geometry_functional_audit.yaml"
)


def test_canonical_audit_preregistration_locks_representation_comparison() -> None:
    prereg = FunctionalGeometryAuditPreregistration.load(AUDIT_CONFIG)

    assert prereg.functional_preregistration_sha256 == (
        "f40251443426eab2f24d89cdf359f07615ccda788341d16719c1f0ec40836bdf"
    )
    assert prereg.required_functional_next_action == (
        "stop_stage1_and_revise_functional_geometry"
    )
    assert prereg.checkpoint_steps == (500, 20_000)
    assert prereg.primary_checkpoint_step == 20_000
    assert prereg.positive_control_checkpoint_step == 500
    assert prereg.layers == (
        "down2_block.conv2.weight",
        "middle.conv2.weight",
    )
    assert prereg.classes == (0, 2, 3, 4, 6, 9)
    assert prereg.basis_kinds == ("raw", "row_normalized")
    assert prereg.orientation_gradient == "raw_scale_mean"
    assert prereg.fold_offsets == (0, 4, 8, 12)
    assert prereg.relative_step_grid == (1e-4, 3e-4, 1e-3)
    assert prereg.local_linearity_relative_error_max == 0.10

    folds = prereg.fold_positions
    assert folds[0] == {
        "fit": tuple(range(8)),
        "scale": tuple(range(8, 12)),
        "evaluation": tuple(range(12, 16)),
    }
    assert folds[3] == {
        "fit": (12, 13, 14, 15, 0, 1, 2, 3),
        "scale": (4, 5, 6, 7),
        "evaluation": (8, 9, 10, 11),
    }
    for position in range(prereg.microbatches_per_cell):
        assert sum(position in fold["fit"] for fold in folds) == 2
        assert sum(position in fold["scale"] for fold in folds) == 1
        assert sum(position in fold["evaluation"] for fold in folds) == 1


def test_audit_preregistration_round_trips_and_locks_immutably(
    tmp_path: Path,
) -> None:
    prereg = FunctionalGeometryAuditPreregistration.load(AUDIT_CONFIG)
    reconstructed = FunctionalGeometryAuditPreregistration.from_dict(
        prereg.to_dict()
    )

    assert reconstructed == prereg
    assert reconstructed.digest == prereg.digest
    assert len(prereg.digest) == 64

    path = prereg.lock(tmp_path / "audit.yaml")
    assert FunctionalGeometryAuditPreregistration.load(path) == prereg
    assert prereg.lock(path) == path

    changed = dataclasses.replace(prereg, bootstrap_seed=prereg.bootstrap_seed + 1)
    with pytest.raises(ValueError, match="replace a locked functional geometry audit"):
        changed.lock(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"probe_view": "b"}, "Probe-A"),
        ({"required_stage1_unlocked": True}, "blocked"),
        ({"rank": 2}, "rank-1"),
        ({"basis_kinds": ("raw", "raw")}, "basis"),
        ({"orientation_gradient": "normalized_scale_mean"}, "raw scale"),
        ({"fold_offsets": (0, 4, 8, 8)}, "offset"),
        ({"relative_step_grid": (1e-5, 1e-4)}, "finite-step grid"),
        ({"layers": ("down2_block.conv2.weight",)}, "two locked layers"),
        ({"classes": (0, 2)}, "six locked classes"),
        ({"checkpoint_steps": (20_000,)}, "both locked checkpoints"),
        ({"local_linearity_relative_error_max": 0.2}, "10%"),
    ],
)
def test_audit_preregistration_rejects_scope_or_method_drift(
    changes: dict[str, object],
    message: str,
) -> None:
    prereg = FunctionalGeometryAuditPreregistration.load(AUDIT_CONFIG)

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(prereg, **changes)


def test_audit_preregistration_rejects_unknown_or_missing_keys() -> None:
    prereg = FunctionalGeometryAuditPreregistration.load(AUDIT_CONFIG)
    payload = prereg.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        FunctionalGeometryAuditPreregistration.from_dict(payload)

    payload = prereg.to_dict()
    del payload["directions"]["basis_kinds"]
    with pytest.raises(ValueError, match="missing fields"):
        FunctionalGeometryAuditPreregistration.from_dict(payload)


def test_audit_preregistration_rejects_changed_input_identity() -> None:
    prereg = FunctionalGeometryAuditPreregistration.load(AUDIT_CONFIG)

    with pytest.raises(ValueError, match="functional-calibration identity"):
        dataclasses.replace(
            prereg,
            functional_preregistration_sha256="0" * 64,
        )
