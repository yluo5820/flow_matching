from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fm_lab.geometry_explorer.synthetic_long_tail_metrics as metrics_module
from fm_lab.geometry_explorer.latent_factors import sample_values
from fm_lab.geometry_explorer.synthetic_factor_oracle import (
    _factor_space,
    train_factor_oracle,
)
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    OBJECT_IDS,
    _object_configs,
    _render_map,
)
from fm_lab.geometry_explorer.synthetic_long_tail_metrics import (
    calibrate_metric_controls,
    central_range_ratio,
    deterministic_subsample,
    evaluate_generated_distribution,
    multivariate_energy_distance,
    normalized_wasserstein,
    oracle_feature_fid,
    summarize_validity,
)


def _config() -> dict[str, Any]:
    return {
        "seed": 17,
        "image_size": 8,
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {
            "background": [1.0, 1.0, 1.0],
            "camera_distance": 4.0,
            "elevation_bounds_degrees": [-30.0, 30.0],
            "supersample": 1,
            "render_batch_size": 2,
        },
        "oracle": {
            "training_samples_per_object": 2,
            "validation_samples_per_object": 1,
            "batch_size": 2,
            "steps": 1,
            "learning_rate": 1.0e-3,
            "min_object_accuracy": 0.0,
            "max_normalized_factor_mae": 2.0,
        },
    }


def _write_real_generated_samples(
    config: dict[str, Any],
    root: Path,
    *,
    count_per_class: int,
) -> None:
    samples_dir = root / "samples"
    samples_dir.mkdir(parents=True)
    object_configs = _object_configs(config)
    factor = _factor_space((-np.pi / 6.0, np.pi / 6.0))
    images = []
    requested = []
    for class_id, object_id in enumerate(OBJECT_IDS):
        values = sample_values(factor.sample(count_per_class, seed=100 + class_id))
        render_map = _render_map(config, object_configs[object_id], factor)
        for value in values:
            image = np.asarray(render_map.render(value), dtype=np.float32)
            images.append(image.transpose(2, 0, 1))
            requested.append(class_id)
    normalized = np.asarray(images, dtype=np.float32) * 2.0 - 1.0
    np.save(samples_dir / "euler_nfe64.npy", normalized.reshape(len(normalized), -1))
    np.save(samples_dir / "generated_labels.npy", np.asarray(requested, dtype=np.int64))


def test_factor_metrics_order_full_half_and_collapsed_controls() -> None:
    rng = np.random.default_rng(3)
    reference = rng.uniform(-1.0, 1.0, 5_000)
    full = rng.uniform(-1.0, 1.0, 5_000)
    half = rng.uniform(-0.5, 0.5, 5_000)
    collapsed = np.zeros(5_000)

    errors = [
        normalized_wasserstein(values, reference, value_range=2.0)
        for values in (full, half, collapsed)
    ]
    ratios = [central_range_ratio(values, reference) for values in (full, half, collapsed)]

    assert errors[0] < errors[1] < errors[2]
    assert ratios[0] > ratios[1] > ratios[2]


def test_distribution_summary_keeps_invalid_mass_visible() -> None:
    summary = summarize_validity(
        predicted_class=np.asarray([0, 0, 1, 0]),
        requested_class=np.asarray([0, 0, 0, 0]),
        render_residual=np.asarray([0.01, 0.50, 0.01, 0.02]),
        residual_threshold=0.05,
    )

    assert summary == {
        "class_leakage_rate": 0.25,
        "off_renderer_rate": 0.25,
        "joint_valid_rate": 0.50,
    }


def test_deterministic_subsample_is_seeded_sorted_and_without_replacement() -> None:
    values = np.arange(40, dtype=np.float64).reshape(20, 2)
    first = deterministic_subsample(values, 7, seed=9)
    second = deterministic_subsample(values, 7, seed=9)

    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first[:, 0])) == 7
    assert np.all(np.diff(first[:, 0]) > 0)


@pytest.mark.parametrize(
    ("values", "maximum", "seed", "error", "message"),
    [
        ([1.0, 2.0], 1, 0, TypeError, "numpy array"),
        (np.ones((2, 0)), 1, 0, ValueError, "non-empty feature dimensions"),
        (np.ones(2), True, 0, TypeError, "maximum must be an integer"),
        (np.ones(2), 1, -1, ValueError, "seed must be non-negative"),
    ],
)
def test_deterministic_subsample_rejects_invalid_contracts(
    values: object,
    maximum: object,
    seed: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        deterministic_subsample(values, maximum, seed)  # type: ignore[arg-type]


def test_energy_and_fid_are_finite_for_one_sample() -> None:
    generated = np.asarray([[1.0, 2.0]])
    reference = np.asarray([[4.0, 6.0]])

    assert multivariate_energy_distance(generated, reference) == pytest.approx(10.0)
    assert oracle_feature_fid(generated, reference) == pytest.approx(25.0)


def test_validity_rejects_empty_or_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_validity(
            predicted_class=np.asarray([], dtype=np.int64),
            requested_class=np.asarray([], dtype=np.int64),
            render_residual=np.asarray([], dtype=np.float64),
            residual_threshold=0.1,
        )
    with pytest.raises(ValueError, match="matching one-dimensional shapes"):
        summarize_validity(
            predicted_class=np.asarray([0, 1]),
            requested_class=np.asarray([0]),
            render_residual=np.asarray([0.0, 0.0]),
            residual_threshold=0.1,
        )


def test_evaluator_uses_real_oracle_and_renderer_and_preserves_invalid_mass(
    tmp_path: Path,
) -> None:
    config = _config()
    oracle = train_factor_oracle(config, tmp_path / "oracle", "cpu")
    generated_root = tmp_path / "generated"
    _write_real_generated_samples(config, generated_root, count_per_class=2)

    result = evaluate_generated_distribution(
        generated_root=generated_root,
        oracle_checkpoint=oracle["checkpoint_path"],
        oracle_gate=oracle["gate_path"],
        output_dir=tmp_path / "evaluation",
        device="cpu",
        samples_per_class=2,
        reference_samples_per_class=2,
        seed=53,
        required_gate_profile="fixture_only",
        source_revision="test-revision",
    )

    assert (tmp_path / "evaluation").is_symlink()
    assert (tmp_path / "evaluation" / "factor_metrics.json").is_file()
    assert (tmp_path / "evaluation" / "factor_metrics_by_class.csv").is_file()
    persisted = json.loads(
        (tmp_path / "evaluation" / "factor_metrics.json").read_text(encoding="utf-8")
    )
    assert persisted == result
    assert result["provenance"]["source_revision"] == "test-revision"
    assert len(result["provenance"]["oracle_checkpoint_digest"]) == 64
    assert len(result["provenance"]["generated_samples_sha256"]) == 64
    assert result["provenance"]["reference_source"] == (
        "independently_sampled_high_dimensional_renderer"
    )
    assert result["requested_labels"] == [0, 1, 2]
    assert result["samples_per_requested_class"] == 2
    assert len(result["classes"]) == 3
    for class_result in result["classes"]:
        assert set(class_result["validity"]) == {
            "class_leakage_rate",
            "off_renderer_rate",
            "joint_valid_rate",
        }
        assert class_result["all_requested"]["sample_count"] == 2
        assert class_result["all_requested"]["status"] == "ok"
        assert class_result["joint_valid"]["sample_count"] <= 2
        if class_result["joint_valid"]["sample_count"] == 0:
            assert class_result["joint_valid"]["status"] == "empty_joint_valid_subset"
            assert class_result["joint_valid"]["metrics"] is None
    with (tmp_path / "evaluation" / "factor_metrics_by_class.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(row["source_revision"] == "test-revision" for row in rows)
    assert all(row["oracle_checkpoint_digest"] for row in rows)


def test_evaluator_rejects_wrong_counts_before_loading_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_root = tmp_path / "generated"
    samples_dir = generated_root / "samples"
    samples_dir.mkdir(parents=True)
    np.save(samples_dir / "euler_nfe64.npy", np.zeros((5, 3 * 8 * 8), dtype=np.float32))
    np.save(samples_dir / "generated_labels.npy", np.asarray([0, 0, 1, 1, 2]))

    def unexpected_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("invalid generated counts must fail before loading the oracle")

    monkeypatch.setattr(metrics_module, "load_factor_oracle", unexpected_load)
    with pytest.raises(ValueError, match="exactly 2 samples for requested class 2"):
        evaluate_generated_distribution(
            generated_root=generated_root,
            oracle_checkpoint=tmp_path / "missing.pt",
            oracle_gate=tmp_path / "missing.json",
            output_dir=tmp_path / "evaluation",
            device="cpu",
            samples_per_class=2,
            reference_samples_per_class=2,
            required_gate_profile="fixture_only",
        )


def test_evaluator_refuses_broken_output_symlink_before_reading_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "evaluation"
    output_dir.symlink_to(tmp_path / "missing", target_is_directory=True)
    original_target = os.readlink(output_dir)

    def unexpected_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("existing destination must fail before reading generated arrays")

    monkeypatch.setattr(metrics_module.np, "load", unexpected_load)
    with pytest.raises(FileExistsError, match="Evaluation destination already exists"):
        evaluate_generated_distribution(
            generated_root=tmp_path / "generated",
            oracle_checkpoint=tmp_path / "missing.pt",
            oracle_gate=tmp_path / "missing.json",
            output_dir=output_dir,
            device="cpu",
            samples_per_class=2,
            reference_samples_per_class=2,
            required_gate_profile="fixture_only",
        )
    assert output_dir.is_symlink()
    assert os.readlink(output_dir) == original_target


def test_metric_controls_render_real_images_and_have_strict_factor_ordering(
    tmp_path: Path,
) -> None:
    config = _config()
    oracle = train_factor_oracle(config, tmp_path / "oracle", "cpu")

    result = calibrate_metric_controls(
        oracle_checkpoint=oracle["checkpoint_path"],
        oracle_gate=oracle["gate_path"],
        output_dir=tmp_path / "controls",
        device="cpu",
        samples_per_class=3,
        seed=79,
        required_gate_profile="fixture_only",
        source_revision="test-revision",
    )

    assert (tmp_path / "controls").is_symlink()
    assert result["control_ordering"]["passed"] is True
    assert result["rendered_samples_per_control"] == 9
    for factor_name in ("tx", "ty", "tz", "azimuth", "elevation"):
        errors = [
            result["controls"][name]["factor_truth"][factor_name]["normalized_wasserstein"]
            for name in ("full", "half", "collapsed")
        ]
        coverage = [
            result["controls"][name]["factor_truth"][factor_name]["central_range_ratio"]
            for name in ("full", "half", "collapsed")
        ]
        assert errors[0] < errors[1] < errors[2]
        assert coverage[0] > coverage[1] > coverage[2]


def test_evaluator_refuses_publication_race_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    oracle = train_factor_oracle(config, tmp_path / "oracle", "cpu")
    generated_root = tmp_path / "generated"
    _write_real_generated_samples(config, generated_root, count_per_class=1)
    output_dir = tmp_path / "evaluation"
    original_symlink = metrics_module.os.symlink

    def race_destination(
        target: str,
        link_name: Path,
        *,
        target_is_directory: bool,
    ) -> None:
        output_dir.mkdir()
        (output_dir / "sentinel.txt").write_text("preserve me", encoding="utf-8")
        original_symlink(target, link_name, target_is_directory=target_is_directory)

    monkeypatch.setattr(metrics_module.os, "symlink", race_destination)
    with pytest.raises(FileExistsError):
        evaluate_generated_distribution(
            generated_root=generated_root,
            oracle_checkpoint=oracle["checkpoint_path"],
            oracle_gate=oracle["gate_path"],
            output_dir=output_dir,
            device="cpu",
            samples_per_class=1,
            reference_samples_per_class=1,
            seed=53,
            required_gate_profile="fixture_only",
        )
    assert (output_dir / "sentinel.txt").read_text(encoding="utf-8") == "preserve me"
    assert not (output_dir / "factor_metrics.json").exists()
    assert not list(tmp_path.glob(".evaluation-*"))
