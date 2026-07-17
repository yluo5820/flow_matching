from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fm_lab.geometry_explorer.synthetic_long_tail_calibration import (
    RendererGateThresholds,
    _max_nuisance_standardized_difference,
    calibrate_renderer,
    renderer_gate,
)
from fm_lab.geometry_explorer.synthetic_long_tail_design import build_master_pools


@pytest.mark.parametrize(
    ("overrides", "failed_check"),
    [
        ({"object_accuracy": 0.98}, "object_separability"),
        ({"max_nuisance_difference": 0.26}, "nuisance_matching"),
        ({"full_rank_fraction": 0.94}, "renderer_rank"),
        ({"pullback_norm_ratio": 4.01}, "factor_visibility"),
    ],
)
def test_renderer_gate_reports_each_threshold_failure_independently(
    overrides: dict[str, float],
    failed_check: str,
) -> None:
    metrics = {
        "object_accuracy": 0.99,
        "max_nuisance_difference": 0.25,
        "full_rank_fraction": 0.95,
        "pullback_norm_ratio": 4.0,
    }
    metrics.update(overrides)
    result = renderer_gate(thresholds=RendererGateThresholds(), **metrics)

    assert result["passed"] is False
    assert result["checks"] == {
        "object_separability": failed_check != "object_separability",
        "nuisance_matching": failed_check != "nuisance_matching",
        "renderer_rank": failed_check != "renderer_rank",
        "factor_visibility": failed_check != "factor_visibility",
    }


def calibration_config() -> dict[str, Any]:
    return {
        "seed": 17,
        "image_size": 8,
        "master_count": 2,
        "counts": [2, 1, 1],
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {
            "background": [1.0, 1.0, 1.0],
            "camera_distance": 4.0,
            "supersample": 1,
            "render_batch_size": 8,
        },
        "calibration": {
            "renderer_points_per_cell": 2,
            "min_object_accuracy": 0.0,
            "max_nuisance_standardized_difference": 1.0e9,
            "relative_singular_threshold": 0.02,
            "full_rank_fraction": 0.0,
            "max_pullback_norm_ratio": 1.0e9,
        },
    }


def _nuisance_rows(object_means: tuple[float, float, float]) -> list[dict[str, Any]]:
    rows = []
    for object_id, mean in zip(
        ("stepped_monument", "crooked_arch", "three_arm_vane"),
        object_means,
        strict=True,
    ):
        for dimension_id in ("high", "medium", "low"):
            row: dict[str, Any] = {
                "object_id": object_id,
                "dimension_id": dimension_id,
            }
            for metric in ("foreground_occupancy", "luminance", "contrast"):
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = 1.0
            rows.append(row)
    return rows


def test_nuisance_statistic_detects_cross_object_mismatch_after_dimension_averaging() -> None:
    difference = _max_nuisance_standardized_difference(
        _nuisance_rows((0.0, 10.0, 20.0))
    )
    result = renderer_gate(
        object_accuracy=1.0,
        max_nuisance_difference=difference,
        full_rank_fraction=1.0,
        pullback_norm_ratio=1.0,
        thresholds=RendererGateThresholds(),
    )

    assert difference == pytest.approx(20.0)
    assert result["passed"] is False
    assert result["checks"]["nuisance_matching"] is False


def test_nuisance_statistic_accepts_objects_with_matched_dimension_averages() -> None:
    difference = _max_nuisance_standardized_difference(
        _nuisance_rows((3.0, 3.0, 3.0))
    )
    result = renderer_gate(
        object_accuracy=1.0,
        max_nuisance_difference=difference,
        full_rank_fraction=1.0,
        pullback_norm_ratio=1.0,
        thresholds=RendererGateThresholds(),
    )

    assert difference == pytest.approx(0.0)
    assert result["passed"] is True


def test_calibration_writes_real_statistics_without_mutating_pools(tmp_path: Path) -> None:
    config = calibration_config()
    cells = build_master_pools(config, tmp_path / "dataset", replicate=0)
    pool_snapshots = {
        cell.cell_id: (np.load(cell.image_path).copy(), np.load(cell.factor_path).copy())
        for cell in cells
    }

    result = calibrate_renderer(config, tmp_path / "calibration")

    assert result["artifacts"] == {
        "renderer_gate": "renderer_gate.json",
        "class_statistics": "renderer_class_statistics.csv",
        "singular_values": "renderer_singular_values.npz",
    }
    for artifact in result["artifacts"].values():
        assert (tmp_path / "calibration" / artifact).is_file()

    with (tmp_path / "calibration" / "renderer_class_statistics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert {row["object_id"] for row in rows} == {
        "stepped_monument",
        "crooked_arch",
        "three_arm_vane",
    }
    assert all(float(row["foreground_occupancy_mean"]) >= 0.0 for row in rows)
    assert all(float(row["luminance_mean"]) >= 0.0 for row in rows)
    assert all(float(row["contrast_mean"]) >= 0.0 for row in rows)

    singular = np.load(tmp_path / "calibration" / "renderer_singular_values.npz")
    assert singular["values"].shape == (18, 5)
    assert singular["dimensions"].shape == (18,)
    assert np.all(np.isfinite(singular["values"][:, 0]))
    assert result["object_accuracy"] >= 0.0
    assert result["full_rank_fraction"] >= 0.0
    assert result["pullback_norm_ratio"] >= 0.0

    for cell in cells:
        expected_images, expected_factors = pool_snapshots[cell.cell_id]
        np.testing.assert_array_equal(np.load(cell.image_path), expected_images)
        np.testing.assert_array_equal(np.load(cell.factor_path), expected_factors)


def test_calibration_rerun_refuses_overwrite_and_preserves_every_artifact(
    tmp_path: Path,
) -> None:
    config = calibration_config()
    output_dir = tmp_path / "calibration"
    first = calibrate_renderer(config, output_dir)
    artifacts_before = {
        name: (output_dir / name).read_bytes()
        for name in first["artifacts"].values()
    }

    with pytest.raises(FileExistsError, match="Calibration destination already exists"):
        calibrate_renderer(config, output_dir)

    artifacts_after = {
        path.name: path.read_bytes()
        for path in output_dir.iterdir()
        if path.is_file()
    }
    assert artifacts_after == artifacts_before
