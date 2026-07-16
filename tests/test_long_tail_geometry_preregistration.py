import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.diagnostics.long_tail_geometry.registry import (
    prepare_observation0_registry,
    update_observation0_run,
)

CANONICAL_PATH = Path(
    "configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml"
)
CIFAR_CANONICAL_PATH = Path(
    "configs/cifar10_lt/long_tail_geometry_observation0_preregistration.yaml"
)
FASHION_CANONICAL_DIGEST = (
    "6cd1bcf18692dc947573dbfb0da7b4d98b16fd291ea8d436a32e3e9abec78e24"
)


def test_canonical_observation0_preregistration_is_fully_locked() -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)

    assert prereg.training_seeds == (0, 1, 2)
    assert prereg.checkpoint_steps == (0, 500, 1000, 3000, 7000, 13000, 20000)
    assert prereg.observation0_mapping_offsets == (0,)
    assert prereg.stage1_mapping_offsets == tuple(range(10))
    assert prereg.source_noise_replicas == 2
    assert prereg.gate_ranks == (1, 2, 4, 8)
    assert prereg.descriptive_ranks == (1, 2, 4, 8, 16, 32)
    assert prereg.required_seed_repeats == 2
    assert prereg.minimum_common_classes == 5
    assert prereg.stage1_requires_functional_lock is True
    assert len(prereg.digest) == 64
    assert prereg.digest == FASHION_CANONICAL_DIGEST


def test_cifar_observation0_preregistration_locks_natural_image_scale() -> None:
    prereg = Observation0Preregistration.load(CIFAR_CANONICAL_PATH)

    assert prereg.dataset == "cifar10_lt"
    assert prereg.training_seeds == (0, 1, 2)
    assert prereg.checkpoint_steps == (0, 2500, 10_000, 25_000, 50_000, 100_000)
    assert prereg.layers == Observation0Preregistration.load(CANONICAL_PATH).layers
    assert prereg.primary_microbatches_per_cell == 16
    assert prereg.minimum_common_classes == 5
    assert len(prereg.digest) == 64


def test_preregistration_round_trips_without_digest_change(tmp_path: Path) -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)

    path = prereg.lock(tmp_path / "aggregate/preregistration.yaml")
    restored = Observation0Preregistration.load(path)

    assert restored == prereg
    assert restored.to_dict() == prereg.to_dict()
    assert restored.digest == prereg.digest


def test_lock_refuses_to_replace_a_different_preregistration(tmp_path: Path) -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)
    path = prereg.lock(tmp_path / "preregistration.yaml")
    prereg.lock(path)
    changed = dataclasses.replace(
        prereg,
        null_permutations=prereg.null_permutations + 1,
    )

    with pytest.raises(ValueError, match="locked preregistration"):
        changed.lock(path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"training_seeds": (0, 0, 2)}, "unique"),
        ({"checkpoint_steps": (500, 1000)}, "step zero"),
        ({"observation0_mapping_offsets": (1,)}, "offset 0"),
        ({"source_noise_replicas": 1}, "two source-noise replicas"),
        ({"gate_ranks": (1, 2, 16)}, "sample rank"),
        ({"null_quantile": 0.95}, "0.99"),
        ({"required_seed_repeats": 4}, "training seeds"),
        ({"stage1_requires_functional_lock": False}, "Stage 1"),
    ],
)
def test_preregistration_rejects_protocol_changes(updates, message: str) -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(prereg, **updates)


def test_prepare_registry_contains_all_pilot_runs_and_exclusion_header(
    tmp_path: Path,
) -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)

    registry = prepare_observation0_registry(prereg, tmp_path)

    assert list(registry["seed"]) == [0, 1, 2]
    assert set(registry["mapping_offset"]) == {0}
    assert set(registry["status"]) == {"planned"}
    assert set(registry["study_digest"]) == {prereg.digest}
    assert (tmp_path / "aggregate/preregistration.yaml").exists()
    exclusion = pd.read_csv(tmp_path / "aggregate/exclusion_log.csv")
    assert list(exclusion) == [
        "study_digest",
        "condition",
        "mapping_offset",
        "seed",
        "run_dir",
        "status",
        "measurement_digest",
        "exclusion_reason",
    ]
    assert exclusion.empty


def test_prepare_registry_is_idempotent_for_same_preregistration(tmp_path: Path) -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)

    first = prepare_observation0_registry(prereg, tmp_path)
    second = prepare_observation0_registry(prereg, tmp_path)

    pd.testing.assert_frame_equal(first, second)


def test_registry_updates_registered_run_atomically(tmp_path: Path) -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)
    prepare_observation0_registry(prereg, tmp_path)

    updated = update_observation0_run(
        tmp_path,
        seed=1,
        status="measured",
        run_dir=tmp_path / "mapping_0/seed_1",
        measurement_digest="a" * 64,
    )

    row = updated.loc[updated["seed"] == 1].iloc[0]
    assert row["status"] == "measured"
    assert row["measurement_digest"] == "a" * 64
    assert row["run_dir"] == str(tmp_path / "mapping_0/seed_1")
    persisted = pd.read_csv(tmp_path / "aggregate/run_registry.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(persisted, updated)
    assert not (tmp_path / "aggregate/run_registry.csv.tmp").exists()


def test_registry_rejects_unregistered_seed_and_invalid_status(tmp_path: Path) -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)
    prepare_observation0_registry(prereg, tmp_path)

    with pytest.raises(ValueError, match="not preregistered"):
        update_observation0_run(tmp_path, seed=9, status="measured")
    with pytest.raises(ValueError, match="status"):
        update_observation0_run(tmp_path, seed=0, status="silently_dropped")


def test_exclusion_requires_reason_and_remains_in_both_logs(tmp_path: Path) -> None:
    prereg = Observation0Preregistration.load(CANONICAL_PATH)
    prepare_observation0_registry(prereg, tmp_path)

    with pytest.raises(ValueError, match="reason"):
        update_observation0_run(tmp_path, seed=0, status="excluded")

    registry = update_observation0_run(
        tmp_path,
        seed=0,
        status="excluded",
        exclusion_reason="checkpoint checksum mismatch",
    )

    row = registry.loc[registry["seed"] == 0].iloc[0]
    assert row["status"] == "excluded"
    assert row["exclusion_reason"] == "checkpoint checksum mismatch"
    exclusions = pd.read_csv(
        tmp_path / "aggregate/exclusion_log.csv",
        keep_default_na=False,
    )
    assert len(exclusions) == 1
    assert exclusions.iloc[0]["exclusion_reason"] == "checkpoint checksum mismatch"
