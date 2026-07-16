import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from fm_lab.diagnostics.long_tail_geometry.measurements import (
    CheckpointMeasurements,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.diagnostics.long_tail_geometry.reliability import (
    aggregate_observation0_reliability,
    analyze_seed_reliability,
    centered_cell_statistics,
)

CANONICAL_PATH = Path(
    "configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml"
)


def _preregistration(**updates) -> Observation0Preregistration:
    preregistration = Observation0Preregistration.load(CANONICAL_PATH)
    return dataclasses.replace(preregistration, null_permutations=99, **updates)


def _normalize(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _planted_rows(
    *,
    rows: int,
    dimension: int,
    rank: int,
    noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.RandomState(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    coefficients = rng.normal(size=(rows, rank))
    a = coefficients @ basis.T + noise * rng.normal(size=(rows, dimension))
    b = coefficients @ basis.T + noise * rng.normal(size=(rows, dimension))
    return torch.tensor(_normalize(a), dtype=torch.float32), torch.tensor(
        _normalize(b), dtype=torch.float32
    )


def _synthetic_measurement(*, shared_by_class: bool, seed: int) -> CheckpointMeasurements:
    rng = np.random.RandomState(seed)
    classes = range(6)
    rows_per_cell = 10
    dimension = 24
    rank = 2
    records: list[dict[str, object]] = []
    rows_by_view: dict[str, list[np.ndarray]] = {
        "a": [],
        "a_source_1": [],
        "b": [],
        "b_source_1": [],
    }
    for class_id in classes:
        start = class_id * rank
        basis_a = np.eye(dimension)[:, start : start + rank]
        if shared_by_class:
            basis_b = basis_a
        else:
            raw = rng.normal(size=(dimension, rank))
            basis_b, _ = np.linalg.qr(raw)
        coefficients = rng.normal(size=(rows_per_cell, rank))
        view_rows = {
            "a": coefficients @ basis_a.T + 0.005 * rng.normal(
                size=(rows_per_cell, dimension)
            ),
            "a_source_1": coefficients @ basis_a.T + 0.008 * rng.normal(
                size=(rows_per_cell, dimension)
            ),
            "b": coefficients @ basis_b.T + 0.005 * rng.normal(
                size=(rows_per_cell, dimension)
            ),
            "b_source_1": coefficients @ basis_b.T + 0.008 * rng.normal(
                size=(rows_per_cell, dimension)
            ),
        }
        for view, values in view_rows.items():
            normalized = _normalize(values)
            for microbatch_id, row in enumerate(normalized):
                rows_by_view[view].append(row)
                records.append(
                    {
                        "checkpoint_step": 500,
                        "probe_view": view,
                        "target_split": view[0],
                        "class_id": class_id,
                        "stratum_id": 0,
                        "microbatch_id": microbatch_id,
                        "batch_size": 8,
                        "loss": 1.0,
                        "original_indices_sha256": f"{class_id:064x}",
                    }
                )
    # Records are class/view interleaved, so reconstruct sketches in the same order.
    ordered_rows: list[np.ndarray] = []
    offsets = {view: 0 for view in rows_by_view}
    for record in records:
        view = str(record["probe_view"])
        ordered_rows.append(rows_by_view[view][offsets[view]])
        offsets[view] += 1
    sketches = torch.tensor(np.stack(ordered_rows), dtype=torch.float32)
    return CheckpointMeasurements(
        metadata=pd.DataFrame.from_records(records),
        sketches={"hidden.weight": sketches},
        exact_norms={"hidden.weight": torch.ones(len(records))},
        layer_shapes={"hidden.weight": (dimension,)},
        manifest_digests={view: str(index + 1) * 64 for index, view in enumerate(rows_by_view)},
        checkpoint_step=500,
        checkpoint_sha256="a" * 64,
        preregistration_sha256=_preregistration().digest,
    )


def test_centered_cell_statistics_report_overlap_angles_rank_and_concentration() -> None:
    a, b = _planted_rows(rows=16, dimension=64, rank=4, noise=0.01, seed=7)

    result = centered_cell_statistics(a, b, ranks=(1, 2, 4, 8, 16))

    rank4 = result[result["rank"] == 4].iloc[0]
    assert rank4["available"]
    assert rank4["projection_overlap"] > 0.95
    assert rank4["largest_principal_angle_degrees"] < 15
    assert not result[result["rank"] == 16].iloc[0]["available"]
    assert 0 <= rank4["directional_concentration_a"] <= 1
    assert rank4["effective_rank_a"] >= 1


def test_zero_gradient_control_cell_is_unavailable_not_corrupt() -> None:
    zeros = torch.zeros(16, 64)

    result = centered_cell_statistics(zeros, zeros, ranks=(1, 2, 4, 8, 16))

    assert not result["available"].any()
    assert result["projection_overlap"].isna().all()
    assert set(result["directional_concentration_a"]) == {0.0}


def test_class_permutation_null_controls_the_maximum_across_cells() -> None:
    measurements = _synthetic_measurement(shared_by_class=True, seed=3)
    second_metadata = measurements.metadata.copy()
    second_metadata["checkpoint_step"] = 1000
    second_checkpoint = dataclasses.replace(
        measurements,
        metadata=second_metadata,
        checkpoint_step=1000,
        checkpoint_sha256="c" * 64,
    )

    table = analyze_seed_reliability(
        (measurements, second_checkpoint),
        _preregistration(),
        training_seed=0,
    )
    primary = table[
        (table["representation"] == "centered_covariance") & (table["rank"] == 2)
    ]

    assert primary["null_threshold"].nunique() == 1
    assert primary["projection_overlap"].max() > primary["null_threshold"].iloc[0]
    assert primary["measurable"].all()
    assert {"probe_a_split_half_overlap", "source_noise_replica_overlap"} <= set(
        table.columns
    )
    assert table.attrs["measurement_digests"] == [
        measurements.digest,
        second_checkpoint.digest,
    ]
    assert table.attrs["gram_matrices"]
    uncentered_rank2 = table[
        (table["representation"] == "uncentered_second_moment")
        & (table["rank"] == 2)
    ]
    assert uncentered_rank2["probe_a_split_half_overlap"].notna().all()
    assert uncentered_rank2["source_noise_replica_overlap"].notna().all()


def test_random_probe_b_subspaces_do_not_pass_the_reliability_null() -> None:
    measurements = _synthetic_measurement(shared_by_class=False, seed=4)

    table = analyze_seed_reliability(measurements, _preregistration(), training_seed=0)
    primary = table[
        (table["representation"] == "centered_covariance")
        & (table["rank"].isin((1, 2)))
    ]

    assert not primary["measurable"].any()


def _seed_table(
    preregistration: Observation0Preregistration,
    *,
    seed: int,
    measurable_layers: tuple[str, ...],
    measurable_classes: range,
) -> pd.DataFrame:
    records = []
    for layer_name in preregistration.layers:
        for class_id in range(10):
            records.append(
                {
                    "checkpoint_step": 500,
                    "stratum_id": 0,
                    "class_id": class_id,
                    "layer_name": layer_name,
                    "rank": 2,
                    "representation": "centered_covariance",
                    "measurable": layer_name in measurable_layers
                    and class_id in measurable_classes,
                }
            )
    table = pd.DataFrame.from_records(records)
    table.attrs["measurement_digests"] = [f"{seed + 1}" * 64]
    table.attrs["gram_matrices"] = {f"seed_{seed}": np.eye(2, dtype=np.float32)}
    table.attrs["microbatches_per_cell"] = 16
    return table


def test_observation0_passes_only_for_repeated_adjacent_nonoutput_layers(
    tmp_path: Path,
) -> None:
    preregistration = _preregistration()
    layers = preregistration.layers[1:3]
    seed_tables = {
        seed: _seed_table(
            preregistration,
            seed=seed,
            measurable_layers=layers if seed < 2 else (),
            measurable_classes=range(6),
        )
        for seed in preregistration.training_seeds
    }

    decision = aggregate_observation0_reliability(seed_tables, preregistration)

    assert decision.status == "network_wide_measurable"
    assert decision.network_wide_gate_passed
    assert decision.required_escalation is False
    assert decision.probe_phase == "primary"
    assert decision.microbatches_per_cell == 16
    assert decision.passing_pairs[0]["common_classes"] == list(range(6))
    decision.save(tmp_path)
    persisted = pd.read_csv(tmp_path / "reliability.csv")
    assert len(persisted) == sum(len(table) for table in seed_tables.values())
    assert (tmp_path / "noise_ceiling.json").exists()
    assert (tmp_path / "gram_matrices.npz").exists()


def test_output_only_reliability_is_reported_but_does_not_pass() -> None:
    preregistration = _preregistration()
    output_layer = (preregistration.layers[-1],)
    seed_tables = {
        seed: _seed_table(
            preregistration,
            seed=seed,
            measurable_layers=output_layer,
            measurable_classes=range(10),
        )
        for seed in preregistration.training_seeds
    }

    decision = aggregate_observation0_reliability(seed_tables, preregistration)

    assert decision.status == "output_layer_only"
    assert not decision.network_wide_gate_passed
    assert decision.required_escalation is True
    assert decision.output_only_cells


def test_incomplete_seed_set_is_not_interpreted_as_a_null() -> None:
    preregistration = _preregistration()
    table = _seed_table(
        preregistration,
        seed=0,
        measurable_layers=(),
        measurable_classes=range(0),
    )

    with pytest.raises(ValueError, match="all preregistered training seeds"):
        aggregate_observation0_reliability({0: table}, preregistration)


def test_only_escalated_complete_analysis_can_return_practical_null() -> None:
    preregistration = _preregistration()
    seed_tables = {
        seed: _seed_table(
            preregistration,
            seed=seed,
            measurable_layers=(),
            measurable_classes=range(0),
        )
        for seed in preregistration.training_seeds
    }
    primary = aggregate_observation0_reliability(seed_tables, preregistration)
    for table in seed_tables.values():
        table.attrs["microbatches_per_cell"] = 32
    escalated = aggregate_observation0_reliability(seed_tables, preregistration)

    assert primary.status == "escalate_probe_rows"
    assert primary.required_escalation
    assert escalated.status == "network_wide_practical_null"
    assert not escalated.required_escalation
    assert escalated.probe_phase == "escalated"
    assert escalated.microbatches_per_cell == 32


def test_mixed_primary_and_escalated_seed_tables_are_rejected() -> None:
    preregistration = _preregistration()
    seed_tables = {
        seed: _seed_table(
            preregistration,
            seed=seed,
            measurable_layers=(),
            measurable_classes=range(0),
        )
        for seed in preregistration.training_seeds
    }
    seed_tables[2].attrs["microbatches_per_cell"] = 32

    with pytest.raises(ValueError, match="same preregistered probe phase"):
        aggregate_observation0_reliability(seed_tables, preregistration)


def test_escalated_output_only_result_does_not_request_a_second_escalation() -> None:
    preregistration = _preregistration()
    output_layer = (preregistration.layers[-1],)
    seed_tables = {
        seed: _seed_table(
            preregistration,
            seed=seed,
            measurable_layers=output_layer,
            measurable_classes=range(10),
        )
        for seed in preregistration.training_seeds
    }
    for table in seed_tables.values():
        table.attrs["microbatches_per_cell"] = 32

    decision = aggregate_observation0_reliability(seed_tables, preregistration)

    assert decision.status == "output_layer_only"
    assert not decision.required_escalation
    assert decision.next_action == "stop_network_wide_stage1_and_report_output_layer_only"
