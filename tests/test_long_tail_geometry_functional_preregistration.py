import dataclasses
from pathlib import Path

import pytest

from fm_lab.diagnostics.long_tail_geometry.functional_preregistration import (
    FunctionalCalibrationPreregistration,
)


CANONICAL_PATH = Path(
    "configs/fashion_mnist_lt/long_tail_geometry_functional_calibration.yaml"
)
OBSERVATION0_DIGEST = (
    "6cd1bcf18692dc947573dbfb0da7b4d98b16fd291ea8d436a32e3e9abec78e24"
)


def test_canonical_functional_calibration_is_fully_locked() -> None:
    prereg = FunctionalCalibrationPreregistration.load(CANONICAL_PATH)

    assert prereg.observation0_preregistration_sha256 == OBSERVATION0_DIGEST
    assert prereg.checkpoint_steps == (500, 20_000)
    assert prereg.primary_checkpoint_step == 20_000
    assert prereg.positive_control_checkpoint_step == 500
    assert prereg.probe_view == "a"
    assert prereg.stratum_id == 0
    assert prereg.stratum_bounds == (0.02, 0.10)
    assert prereg.rank == 1
    assert prereg.layers == (
        "down2_block.conv2.weight",
        "middle.conv2.weight",
    )
    assert prereg.classes == (0, 2, 3, 4, 6, 9)
    assert prereg.fit_positions == tuple(range(8))
    assert prereg.scale_positions == tuple(range(8, 12))
    assert prereg.evaluation_positions == tuple(range(12, 16))
    assert prereg.microbatches_per_cell == 16
    assert prereg.relative_step_grid == (
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
    )
    assert prereg.target_loss_change_fraction == 0.01
    assert prereg.target_benefit_interval == (0.0075, 0.0125)
    assert prereg.local_linearity_relative_error_max == 0.10
    assert prereg.max_relative_layer_step == 0.01
    assert prereg.random_controls == 99
    assert prereg.bootstrap_resamples == 10_000
    assert len(prereg.digest) == 64


def test_functional_preregistration_round_trips_without_digest_change(
    tmp_path: Path,
) -> None:
    prereg = FunctionalCalibrationPreregistration.load(CANONICAL_PATH)

    path = prereg.lock(tmp_path / "functional/preregistration.yaml")
    restored = FunctionalCalibrationPreregistration.load(path)

    assert restored == prereg
    assert restored.to_dict() == prereg.to_dict()
    assert restored.digest == prereg.digest


def test_functional_lock_refuses_a_changed_preregistration(tmp_path: Path) -> None:
    prereg = FunctionalCalibrationPreregistration.load(CANONICAL_PATH)
    path = prereg.lock(tmp_path / "preregistration.yaml")

    changed = dataclasses.replace(prereg, bootstrap_seed=prereg.bootstrap_seed + 1)
    with pytest.raises(ValueError, match="locked functional preregistration"):
        changed.lock(path)


def test_functional_preregistration_rejects_unknown_keys() -> None:
    prereg = FunctionalCalibrationPreregistration.load(CANONICAL_PATH)
    payload = prereg.to_dict()
    payload["probe"]["allow_probe_b"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        FunctionalCalibrationPreregistration.from_dict(payload)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"observation0_preregistration_sha256": "bad"}, "SHA-256"),
        ({"checkpoint_steps": (500,)}, "both calibration checkpoints"),
        ({"primary_checkpoint_step": 500}, "distinct"),
        ({"probe_view": "b"}, "Probe-A only"),
        ({"stratum_bounds": (0.10, 0.02)}, "stratum"),
        ({"rank": 2}, "rank-1"),
        ({"layers": ("middle.conv2.weight",)}, "adjacent layers"),
        ({"classes": (0,)}, "classes"),
        ({"scale_positions": tuple(range(7, 12))}, "partition"),
        ({"evaluation_positions": tuple(range(13, 16))}, "consecutive"),
        ({"target_loss_change_fraction": 0.02}, "1%"),
        ({"target_benefit_interval": (0.02, 0.03)}, "target interval"),
        ({"local_linearity_relative_error_max": 0.2}, "10%"),
        ({"max_relative_layer_step": 0.02}, "1%"),
        ({"random_controls": 2}, "random controls"),
        ({"random_control_quantile": 0.95}, "0.99"),
        ({"bootstrap_resamples": 98}, "bootstrap"),
        ({"required_seed_repeats": 1}, "two seed"),
    ],
)
def test_functional_preregistration_rejects_protocol_violations(
    updates: dict,
    message: str,
) -> None:
    prereg = FunctionalCalibrationPreregistration.load(CANONICAL_PATH)

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(prereg, **updates)

