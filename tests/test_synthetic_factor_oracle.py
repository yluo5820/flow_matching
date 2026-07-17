from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from fm_lab.geometry_explorer.synthetic_factor_oracle import (
    FACTOR_NAMES,
    SyntheticFactorOracle,
    circular_vector_error,
    load_factor_oracle,
    oracle_gate_metrics,
    oracle_loss,
    train_factor_oracle,
)


def _config(**oracle_overrides: Any) -> dict[str, Any]:
    oracle = {
        "training_samples_per_object": 2,
        "validation_samples_per_object": 1,
        "batch_size": 2,
        "steps": 1,
        "learning_rate": 1.0e-3,
        "min_object_accuracy": 0.0,
        "max_normalized_factor_mae": 2.0,
    }
    oracle.update(oracle_overrides)
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
        "oracle": oracle,
    }


def test_oracle_output_shapes() -> None:
    model = SyntheticFactorOracle(num_classes=3)
    prediction = model(torch.zeros(4, 3, 32, 32))
    assert prediction.class_logits.shape == (4, 3)
    assert prediction.translation.shape == (4, 3)
    assert prediction.view.shape == (4, 3)
    assert prediction.features.shape == (4, 256)
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction.view[:, :2], dim=1),
        torch.ones(4),
    )


@pytest.mark.parametrize(
    "images, message",
    [
        (torch.zeros(3, 32, 32), "NCHW"),
        (torch.zeros(2, 1, 32, 32), "three channels"),
        (torch.full((2, 3, 32, 32), -0.01), r"\[0, 1\]"),
        (torch.full((2, 3, 32, 32), float("nan")), "finite"),
    ],
)
def test_oracle_rejects_invalid_inputs(images: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SyntheticFactorOracle()(images)


def test_oracle_circular_error_wraps_at_pi() -> None:
    predicted = torch.tensor([[math.sin(-math.pi + 0.01), math.cos(-math.pi + 0.01)]])
    target = torch.tensor([[math.sin(math.pi - 0.01), math.cos(math.pi - 0.01)]])
    assert float(circular_vector_error(predicted, target)) < 0.03


def test_oracle_circular_error_rejects_non_circular_targets() -> None:
    with pytest.raises(ValueError, match="nonzero circular vectors"):
        circular_vector_error(torch.ones(1, 2), torch.zeros(1, 2))


def test_oracle_loss_uses_all_four_unit_weight_terms() -> None:
    prediction = SyntheticFactorOracle(num_classes=3)(torch.zeros(2, 3, 8, 8))
    targets = {
        "class_ids": torch.tensor([0, 1]),
        "translation": torch.zeros(2, 3),
        "view": torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
    }
    losses = oracle_loss(prediction, **targets)
    assert set(losses) == {"loss", "classification", "translation", "azimuth", "elevation"}
    torch.testing.assert_close(
        losses["loss"],
        losses["classification"] + losses["translation"] + losses["azimuth"] + losses["elevation"],
    )


def test_oracle_gate_rejects_one_bad_factor() -> None:
    metrics = oracle_gate_metrics(
        object_accuracy=0.999,
        factor_mae={
            "tx": 0.01,
            "ty": 0.01,
            "tz": 0.01,
            "azimuth": 0.03,
            "elevation": 0.01,
        },
        min_accuracy=0.99,
        max_factor_mae=0.02,
    )
    assert metrics["passed"] is False
    assert metrics["failed_factors"] == ["azimuth"]
    assert metrics["failure_reasons"] == ["factor_mae:azimuth"]


def test_oracle_gate_reports_accuracy_and_factor_failures_together() -> None:
    metrics = oracle_gate_metrics(
        object_accuracy=0.5,
        factor_mae={name: (0.2 if name == "tz" else 0.0) for name in FACTOR_NAMES},
        min_accuracy=0.9,
        max_factor_mae=0.1,
    )
    assert metrics["failure_reasons"] == ["object_accuracy", "factor_mae:tz"]


def test_tiny_training_writes_reproducible_checkpoint_and_complete_gate(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "oracle"
    result = train_factor_oracle(_config(), output_dir, "cpu")

    checkpoint_path = output_dir / "factor_oracle.pt"
    gate_path = output_dir / "oracle_gate.json"
    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["gate_path"] == str(gate_path)
    assert checkpoint_path.is_file()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert set(gate["factor_mae"]) == set(FACTOR_NAMES)
    assert gate["validation_rerender_pixel_mae_q995"] >= 0.0
    assert gate["off_renderer_threshold"] == gate["validation_rerender_pixel_mae_q995"]
    assert isinstance(gate["failure_reasons"], list)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["architecture"] == {
        "name": "SyntheticFactorOracle",
        "num_classes": 3,
        "input_channels": 3,
        "feature_dim": 256,
    }
    assert payload["seed"] == 17
    assert len(payload["renderer_config_hash"]) == 64
    assert payload["data_provenance"]["master_pool_reads"] == 0
    assert payload["data_provenance"]["training_samples_per_object"] == 2
    assert payload["data_provenance"]["validation_samples_per_object"] == 1
    assert (
        payload["data_provenance"]["training_seed"] != payload["data_provenance"]["validation_seed"]
    )
    assert tuple(payload["factor_normalization"]) == FACTOR_NAMES
    assert payload["factor_normalization"]["azimuth"]["representation"] == "sin_cos"
    assert payload["factor_normalization"]["elevation"]["world_range_radians"] == pytest.approx(
        [-math.pi / 6.0, math.pi / 6.0]
    )

    restored = load_factor_oracle(checkpoint_path, "cpu")
    expected = restored(torch.zeros(1, 3, 8, 8))
    assert expected.class_logits.shape == (1, 3)


def test_training_is_deterministic_and_independent_of_master_pool_files(
    tmp_path: Path,
) -> None:
    fake_pool = tmp_path / "master_pool.npy"
    fake_pool.write_bytes(b"must never be read")
    first = train_factor_oracle(_config(), tmp_path / "first", "cpu")
    fake_pool.write_bytes(b"changed but still must never be read")
    second = train_factor_oracle(_config(), tmp_path / "second", "cpu")

    left = load_factor_oracle(first["checkpoint_path"], "cpu").state_dict()
    right = load_factor_oracle(second["checkpoint_path"], "cpu").state_dict()
    assert all(torch.equal(left[name], right[name]) for name in left)
    assert first["metrics"] == second["metrics"]


def test_training_refuses_overwrite_without_mutating_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "oracle"
    train_factor_oracle(_config(), output_dir, "cpu")
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    with pytest.raises(FileExistsError, match="Oracle destination already exists"):
        train_factor_oracle(_config(), output_dir, "cpu")

    assert {path.name: path.read_bytes() for path in output_dir.iterdir()} == before


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"training_samples_per_object": True}, "training_samples_per_object"),
        ({"validation_samples_per_object": 0}, "validation_samples_per_object"),
        ({"batch_size": 0}, "batch_size"),
        ({"steps": -1}, "steps"),
        ({"learning_rate": float("inf")}, "learning_rate"),
        ({"min_object_accuracy": 1.1}, "min_object_accuracy"),
        ({"max_normalized_factor_mae": -0.1}, "max_normalized_factor_mae"),
    ],
)
def test_training_validates_oracle_config(
    tmp_path: Path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        train_factor_oracle(_config(**overrides), tmp_path / "oracle", "cpu")


def test_training_rejects_invalid_elevation_ranges_and_unavailable_device(
    tmp_path: Path,
) -> None:
    config = _config()
    config["render"]["elevation_bounds_degrees"] = [20.0, -20.0]
    with pytest.raises(ValueError, match="elevation_bounds_degrees"):
        train_factor_oracle(config, tmp_path / "bad-range", "cpu")

    unavailable = "cuda" if not torch.cuda.is_available() else "mps"
    if unavailable == "mps" and torch.backends.mps.is_available():
        pytest.skip("Both optional accelerators are available")
    with pytest.raises(ValueError, match="unavailable"):
        train_factor_oracle(_config(), tmp_path / "bad-device", unavailable)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("background", [1.0, 1.0]),
        ("background", [1.0, 1.0, float("nan")]),
        ("camera_distance", 0.0),
        ("supersample", True),
        ("render_batch_size", 1.5),
    ],
)
def test_training_strictly_validates_renderer_config(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    config = _config()
    config["render"][field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        train_factor_oracle(config, tmp_path / "oracle", "cpu")


def test_load_rejects_checkpoint_schema_and_renderer_hash_mismatch(tmp_path: Path) -> None:
    result = train_factor_oracle(_config(), tmp_path / "oracle", "cpu")
    payload = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)

    malformed = dict(payload)
    malformed.pop("factor_normalization")
    malformed_path = tmp_path / "malformed.pt"
    torch.save(malformed, malformed_path)
    with pytest.raises(ValueError, match="factor_normalization"):
        load_factor_oracle(malformed_path, "cpu")

    bad_hash = dict(payload)
    bad_hash["renderer_config_hash"] = "0" * 64
    bad_hash_path = tmp_path / "bad-hash.pt"
    torch.save(bad_hash, bad_hash_path)
    with pytest.raises(ValueError, match="renderer config hash"):
        load_factor_oracle(bad_hash_path, "cpu")
