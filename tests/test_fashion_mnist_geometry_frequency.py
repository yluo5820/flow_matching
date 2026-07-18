from __future__ import annotations

import numpy as np
import pandas as pd

from fm_lab.experiments.fashion_mnist_geometry_frequency import (
    SelectionThresholds,
    estimate_geometry_records,
    load_stage0_config,
    run_stage0,
    select_geometry_trio,
)


def test_stage0_dry_run_declares_only_geometry_work() -> None:
    config = load_stage0_config(
        "configs/fashion_mnist_geometry_frequency/stage0.yaml"
    )

    result = run_stage0(config, device="cpu", dry_run=True)

    assert result["probe_images"] == 10_000
    assert result["training_candidates_per_class"] == 5_000
    assert result["representations"] == ["raw_pca50", "dinov2_pca50"]
    assert result["estimator_records"] == 4_000
    assert result["outcome_training_enabled"] is False


def test_geometry_selection_uses_stable_ordinal_ranks() -> None:
    records = _ordered_records()

    gate, ranks = select_geometry_trio(records, SelectionThresholds())

    assert gate["passed"] is True
    assert gate["selected_class_ids"] == {"low": 0, "middle": 4, "high": 9}
    assert len(ranks) == len(records) * 5
    assert all(summary["eligible"] for summary in gate["class_summaries"])


def test_geometry_selection_fails_on_probe_reversal() -> None:
    records = _ordered_records()
    mask = records["probe_split"] == "b"
    for estimator in (
        "two_nn",
        "mle_lid_k10",
        "mle_lid_k20",
        "participation_ratio",
        "pca_dim_90",
    ):
        records.loc[mask, estimator] = 100.0 - records.loc[mask, "class_id"]

    gate, _ = select_geometry_trio(
        records,
        SelectionThresholds(max_probe_score_gap=0.01),
    )

    summary = next(row for row in gate["class_summaries"] if row["class_id"] == 0)
    assert gate["passed"] is False
    assert summary["eligible"] is False
    assert "probe_gap" in summary["reasons"]
    assert "probe_third_reversal" in summary["reasons"]


def test_geometry_subsampling_is_deterministic_and_paired() -> None:
    generator = np.random.default_rng(7)
    labels = np.repeat(np.arange(10), 50)
    probe_splits = np.tile(np.asarray(["a"] * 25 + ["b"] * 25), 10)
    raw = generator.normal(size=(500, 6)).astype(np.float32)
    dino = (raw @ generator.normal(size=(6, 6))).astype(np.float32)
    representations = {"raw_pca50": raw, "dinov2_pca50": dino}

    first = estimate_geometry_records(
        representations,
        labels=labels,
        probe_splits=probe_splits,
        subsamples=2,
        subsample_fraction=0.84,
        seed=13,
    )
    second = estimate_geometry_records(
        representations,
        labels=labels,
        probe_splits=probe_splits,
        subsamples=2,
        subsample_fraction=0.84,
        seed=13,
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 80
    assert first[list(_estimator_columns())].notna().all().all()


def _ordered_records() -> pd.DataFrame:
    rows = []
    for representation in ("raw_pca50", "dinov2_pca50"):
        for split in ("a", "b"):
            for draw in range(3):
                for class_id in range(10):
                    value = float(class_id + 1) + 0.001 * draw
                    rows.append(
                        {
                            "representation": representation,
                            "probe_split": split,
                            "subsample": draw,
                            "class_id": class_id,
                            **{name: value for name in _estimator_columns()},
                        }
                    )
    return pd.DataFrame(rows)


def _estimator_columns() -> tuple[str, ...]:
    return (
        "two_nn",
        "mle_lid_k10",
        "mle_lid_k20",
        "participation_ratio",
        "pca_dim_90",
    )
