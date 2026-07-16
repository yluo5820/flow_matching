import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from fm_lab.diagnostics.long_tail_geometry.natural_image_preregistration import (
    NaturalImageTransportPreregistration,
)

CANONICAL_PATH = Path(
    "configs/cifar10_lt/long_tail_geometry_natural_image_transport.yaml"
)
OBSERVATION0_DIGEST = (
    "4a87fbc8b3a0e3e67a3f71080ce8702cdc524f35a4b7dc8d997200fc601067a7"
)


def test_canonical_natural_image_transport_contract_is_fully_locked() -> None:
    prereg = NaturalImageTransportPreregistration.load(CANONICAL_PATH)

    assert prereg.observation0_preregistration_sha256 == OBSERVATION0_DIGEST
    assert prereg.checkpoint_steps == (0, 10_000, 100_000)
    assert prereg.baseline_checkpoint_step == 0
    assert prereg.early_checkpoint_step == 10_000
    assert prereg.primary_checkpoint_step == 100_000
    assert prereg.probe_view == "a"
    assert prereg.stratum_id == 0
    assert prereg.stratum_bounds == (0.02, 0.10)
    assert prereg.rank == 1
    assert prereg.layers == (
        "down2_block.conv2.weight",
        "middle.conv2.weight",
    )
    assert prereg.classes == tuple(range(10))
    assert prereg.microbatches_per_cell == 16
    assert prereg.fold_offsets == (0, 4, 8, 12)
    assert prereg.basis_kinds == ("raw", "row_normalized")
    assert prereg.orientation_gradient == "raw_scale_mean"
    assert prereg.relative_step_grid == (3e-5, 1e-4, 3e-4, 1e-3)
    assert prereg.local_linearity_relative_error_max == 0.10
    assert prereg.maximum_final_to_baseline_loss_ratio == 0.70
    assert prereg.minimum_reliable_common_classes == 5
    assert prereg.required_seed_repeats == 2
    assert len(prereg.digest) == 64


def test_natural_image_transport_folds_balance_every_microbatch() -> None:
    prereg = NaturalImageTransportPreregistration.load(CANONICAL_PATH)

    fit = Counter(position for fold in prereg.fold_positions for position in fold["fit"])
    scale = Counter(
        position for fold in prereg.fold_positions for position in fold["scale"]
    )
    evaluation = Counter(
        position
        for fold in prereg.fold_positions
        for position in fold["evaluation"]
    )

    assert set(fit) == set(scale) == set(evaluation) == set(range(16))
    assert set(fit.values()) == {2}
    assert set(scale.values()) == {1}
    assert set(evaluation.values()) == {1}


def test_natural_image_transport_round_trips_and_locks_immutably(
    tmp_path: Path,
) -> None:
    prereg = NaturalImageTransportPreregistration.load(CANONICAL_PATH)
    restored = NaturalImageTransportPreregistration.from_dict(prereg.to_dict())

    assert restored == prereg
    assert restored.digest == prereg.digest
    path = prereg.lock(tmp_path / "transport.yaml")
    assert NaturalImageTransportPreregistration.load(path) == prereg
    assert prereg.lock(path) == path

    changed = dataclasses.replace(prereg, bootstrap_seed=prereg.bootstrap_seed + 1)
    with pytest.raises(ValueError, match="locked natural-image transport"):
        changed.lock(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"observation0_preregistration_sha256": "0" * 64}, "Observation-0"),
        ({"checkpoint_steps": (0, 100_000)}, "three locked checkpoints"),
        ({"baseline_checkpoint_step": 10_000}, "baseline checkpoint"),
        ({"primary_checkpoint_step": 10_000}, "primary checkpoint"),
        ({"probe_view": "b"}, "Probe-A"),
        ({"rank": 2}, "rank-1"),
        ({"layers": ("middle.conv2.weight",)}, "two locked layers"),
        ({"classes": tuple(range(9))}, "all ten classes"),
        ({"fold_offsets": (0, 4, 8, 8)}, "fold offsets"),
        ({"basis_kinds": ("raw",)}, "paired bases"),
        ({"relative_step_grid": (1e-4, 3e-4)}, "finite-step grid"),
        ({"maximum_final_to_baseline_loss_ratio": 0.9}, "70%"),
        ({"minimum_reliable_common_classes": 4}, "five common classes"),
        ({"required_seed_repeats": 3}, "two seed repeats"),
    ],
)
def test_natural_image_transport_rejects_protocol_drift(
    changes: dict[str, object],
    message: str,
) -> None:
    prereg = NaturalImageTransportPreregistration.load(CANONICAL_PATH)

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(prereg, **changes)


def test_natural_image_transport_rejects_unknown_or_missing_fields() -> None:
    prereg = NaturalImageTransportPreregistration.load(CANONICAL_PATH)
    payload = prereg.to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        NaturalImageTransportPreregistration.from_dict(payload)

    payload = prereg.to_dict()
    del payload["decision"]["minimum_reliable_common_classes"]
    with pytest.raises(ValueError, match="missing fields"):
        NaturalImageTransportPreregistration.from_dict(payload)
