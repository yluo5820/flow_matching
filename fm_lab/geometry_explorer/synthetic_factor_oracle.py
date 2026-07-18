"""Independent factor oracle for the synthetic long-tail geometry study."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from fm_lab.geometry_explorer.latent_factors import (
    BoundedLookAtView,
    BoundedTranslation,
    ProductFactorSpace,
    sample_values,
)
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    OBJECT_IDS,
    _object_configs,
    _render_map,
    canonical_factor_rows,
)
from fm_lab.utils.logging import write_json

FACTOR_NAMES = ("tx", "ty", "tz", "azimuth", "elevation")
_CHECKPOINT_FILENAME = "factor_oracle.pt"
_GATE_FILENAME = "oracle_gate.json"
_CHECKPOINT_SCHEMA_VERSION = 2
_TRANSLATION_RANGES = ((-0.25, 0.25), (-0.25, 0.25), (-0.75, 0.75))


@dataclass(frozen=True)
class OraclePrediction:
    class_logits: torch.Tensor
    translation: torch.Tensor
    view: torch.Tensor
    features: torch.Tensor


class SyntheticFactorOracle(nn.Module):
    """Compact image encoder predicting object identity and five latent factors."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer.")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.class_head = nn.Linear(256, num_classes)
        self.translation_head = nn.Linear(256, 3)
        self.view_head = nn.Linear(256, 3)

    def forward(self, images: torch.Tensor) -> OraclePrediction:
        _validate_model_input(images, self)
        features = self.encoder(images)
        raw_view = self.view_head(features)
        azimuth = _unit_azimuth(raw_view[:, :2])
        return OraclePrediction(
            class_logits=self.class_head(features),
            translation=torch.tanh(self.translation_head(features)),
            view=torch.cat([azimuth, torch.tanh(raw_view[:, 2:3])], dim=1),
            features=features,
        )


def circular_vector_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean wrapped angular error for vectors stored in ``(sin, cos)`` order."""

    if predicted.ndim != 2 or predicted.shape[1] != 2 or predicted.shape != target.shape:
        raise ValueError("predicted and target must have matching shape (N, 2).")
    if not torch.is_floating_point(predicted) or not torch.is_floating_point(target):
        raise TypeError("circular vectors must be floating-point tensors.")
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("circular vectors must be finite.")
    predicted_norm = torch.linalg.vector_norm(predicted, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)
    if bool((predicted_norm <= 1.0e-8).any()) or bool((target_norm <= 1.0e-8).any()):
        raise ValueError("predicted and target must contain nonzero circular vectors.")
    predicted_angle = torch.atan2(predicted[:, 0], predicted[:, 1])
    target_angle = torch.atan2(target[:, 0], target[:, 1])
    delta = predicted_angle - target_angle
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs().mean()


def oracle_loss(
    prediction: OraclePrediction,
    *,
    class_ids: torch.Tensor,
    translation: torch.Tensor,
    view: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the four unit-weight oracle training terms."""

    batch = prediction.class_logits.shape[0]
    if class_ids.shape != (batch,):
        raise ValueError("class_ids must have shape (N,).")
    if translation.shape != (batch, 3):
        raise ValueError("translation targets must have shape (N, 3).")
    if view.shape != (batch, 3):
        raise ValueError("view targets must have shape (N, 3).")
    if class_ids.dtype != torch.long:
        raise TypeError("class_ids must use torch.long dtype.")
    if not bool(torch.isfinite(translation).all()) or not bool(torch.isfinite(view).all()):
        raise ValueError("factor targets must be finite.")
    azimuth_norm = torch.linalg.vector_norm(view[:, :2], dim=1)
    if bool((torch.abs(azimuth_norm - 1.0) > 1.0e-4).any()):
        raise ValueError("azimuth targets must be unit (sin, cos) vectors.")
    if bool((translation.abs() > 1.0 + 1.0e-6).any()) or bool(
        (view[:, 2].abs() > 1.0 + 1.0e-6).any()
    ):
        raise ValueError("normalized factor targets must lie in [-1, 1].")
    terms = {
        "classification": F.cross_entropy(prediction.class_logits, class_ids),
        "translation": F.mse_loss(prediction.translation, translation),
        "azimuth": F.mse_loss(prediction.view[:, :2], view[:, :2]),
        "elevation": F.mse_loss(prediction.view[:, 2], view[:, 2]),
    }
    terms["loss"] = sum(terms.values())
    return {
        "loss": terms["loss"],
        **{name: terms[name] for name in ("classification", "translation", "azimuth", "elevation")},
    }


def oracle_gate_metrics(
    *,
    object_accuracy: float,
    factor_mae: Mapping[str, float],
    min_accuracy: float,
    max_factor_mae: float,
    rerender_pixel_mae_q995: float | None = None,
) -> dict[str, Any]:
    """Apply every coordinate gate independently and report exact failure reasons."""

    _validate_unit_interval(object_accuracy, "object_accuracy")
    _validate_unit_interval(min_accuracy, "min_accuracy")
    _validate_nonnegative_finite(max_factor_mae, "max_factor_mae")
    if set(factor_mae) != set(FACTOR_NAMES) or len(factor_mae) != len(FACTOR_NAMES):
        raise ValueError(f"factor_mae must contain exactly: {', '.join(FACTOR_NAMES)}")
    normalized_mae = {}
    for name in FACTOR_NAMES:
        value = float(factor_mae[name])
        _validate_nonnegative_finite(value, f"factor_mae.{name}")
        normalized_mae[name] = value
    failed = sorted(name for name, value in normalized_mae.items() if value > max_factor_mae)
    reasons = []
    if object_accuracy < min_accuracy:
        reasons.append("object_accuracy")
    reasons.extend(f"factor_mae:{name}" for name in failed)
    result: dict[str, Any] = {
        "passed": not reasons,
        "checks": {
            "object_accuracy": object_accuracy >= min_accuracy,
            **{
                f"factor_mae:{name}": normalized_mae[name] <= max_factor_mae
                for name in FACTOR_NAMES
            },
        },
        "object_accuracy": float(object_accuracy),
        "factor_mae": normalized_mae,
        "failed_factors": failed,
        "failure_reasons": reasons,
        "thresholds": {
            "min_object_accuracy": float(min_accuracy),
            "max_normalized_factor_mae": float(max_factor_mae),
        },
    }
    if rerender_pixel_mae_q995 is not None:
        _validate_nonnegative_finite(rerender_pixel_mae_q995, "rerender_pixel_mae_q995")
        result["validation_rerender_pixel_mae_q995"] = float(rerender_pixel_mae_q995)
        result["off_renderer_threshold"] = float(rerender_pixel_mae_q995)
    return result


@dataclass(frozen=True)
class _OracleConfig:
    seed: int
    image_size: int
    train_per_object: int
    validation_per_object: int
    batch_size: int
    steps: int
    learning_rate: float
    min_accuracy: float
    max_factor_mae: float
    render_batch_size: int
    elevation_bounds: tuple[float, float]


@dataclass(frozen=True)
class _RenderedDataset:
    images: np.ndarray
    class_ids: np.ndarray
    translation: np.ndarray
    view: np.ndarray


def train_factor_oracle(
    config: dict[str, Any],
    output_dir: str | Path,
    device: str | torch.device,
) -> dict[str, Any]:
    """Train and validate on independently sampled high-dimensional renders."""

    requested_destination = Path(output_dir).expanduser()
    destination = requested_destination.parent.resolve() / requested_destination.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Oracle destination already exists: {destination}")
    parsed = _parse_config(config)
    resolved_device = _resolve_device(device)
    with _deterministic_training_context(resolved_device) as determinism:
        return _train_factor_oracle_impl(
            config,
            destination=destination,
            parsed=parsed,
            device=resolved_device,
            determinism=determinism,
        )


def _train_factor_oracle_impl(
    config: dict[str, Any],
    *,
    destination: Path,
    parsed: _OracleConfig,
    device: torch.device,
    determinism: dict[str, Any],
) -> dict[str, Any]:
    object_configs = _object_configs(config)
    factor = _factor_space(parsed.elevation_bounds)
    normalization = _factor_normalization(parsed.elevation_bounds)
    renderer_config = _renderer_config_payload(config, object_configs, normalization)
    renderer_hash = _stable_hash(renderer_config)
    train_seed = parsed.seed + 10_000_000
    validation_seed = parsed.seed + 20_000_000

    torch.manual_seed(parsed.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(parsed.seed)
    train_data = _render_independent_dataset(
        config,
        object_configs=object_configs,
        factor=factor,
        count_per_object=parsed.train_per_object,
        split_seed=train_seed,
        parsed=parsed,
    )
    validation_data = _render_independent_dataset(
        config,
        object_configs=object_configs,
        factor=factor,
        count_per_object=parsed.validation_per_object,
        split_seed=validation_seed,
        parsed=parsed,
    )
    model = SyntheticFactorOracle(num_classes=len(OBJECT_IDS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parsed.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(parsed.seed + 30_000_000)
    _train_steps(
        model,
        optimizer,
        train_data,
        steps=parsed.steps,
        batch_size=parsed.batch_size,
        device=device,
        generator=generator,
    )
    metrics = _validate_oracle(
        model,
        validation_data,
        config=config,
        object_configs=object_configs,
        factor=factor,
        parsed=parsed,
        device=device,
    )
    gate = oracle_gate_metrics(
        object_accuracy=metrics["object_accuracy"],
        factor_mae=metrics["factor_mae"],
        min_accuracy=parsed.min_accuracy,
        max_factor_mae=parsed.max_factor_mae,
        rerender_pixel_mae_q995=metrics["validation_rerender_pixel_mae_q995"],
    )
    configured_gate_passed = bool(gate["passed"])
    configured_failure_reasons = list(gate["failure_reasons"])
    gate_profile = "production" if _is_production_profile(config, parsed) else "fixture_only"
    production_qualified = gate_profile == "production" and configured_gate_passed
    qualification_reasons = list(configured_failure_reasons)
    if gate_profile != "production":
        qualification_reasons.append("fixture_only_profile")
    gate.update(
        {
            "gate_profile": gate_profile,
            "configured_gate_passed": configured_gate_passed,
            "configured_failure_reasons": configured_failure_reasons,
            "production_qualified": production_qualified,
            "passed": production_qualified,
            "failure_reasons": qualification_reasons,
        }
    )
    provenance = {
        "source": "independently_sampled_high_dimensional_renderer",
        "factor_space": "translation_xyz_bounded_view",
        "master_pool_reads": 0,
        "training_samples_per_object": parsed.train_per_object,
        "validation_samples_per_object": parsed.validation_per_object,
        "training_seed": train_seed,
        "validation_seed": validation_seed,
        "object_cell_seeds": {
            "training": {
                object_id: train_seed + index * 1_000 for index, object_id in enumerate(OBJECT_IDS)
            },
            "validation": {
                object_id: validation_seed + index * 1_000
                for index, object_id in enumerate(OBJECT_IDS)
            },
        },
    }
    gate.update(
        {
            "renderer_config_hash": renderer_hash,
            "seed": parsed.seed,
            "data_provenance": provenance,
        }
    )
    checkpoint = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "architecture": _architecture_payload(len(OBJECT_IDS)),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "renderer_config_hash": renderer_hash,
        "renderer_config": renderer_config,
        "seed": parsed.seed,
        "factor_normalization": normalization,
        "object_ids": list(OBJECT_IDS),
        "training_config": dict(config.get("oracle", {})),
        "config": config,
        "data_provenance": provenance,
        "determinism": determinism,
        "metrics": gate,
    }
    checkpoint["artifact_digest"] = _checkpoint_artifact_digest(checkpoint)
    published_gate = dict(gate)
    published_gate["checkpoint_artifact_digest"] = checkpoint["artifact_digest"]
    checkpoint_path, gate_path = _publish_artifacts(
        destination,
        checkpoint=checkpoint,
        gate=published_gate,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "gate_path": str(gate_path),
        "metrics": published_gate,
    }


def requalify_factor_oracle(
    checkpoint: str | Path,
    gate_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    scientific_basis: str,
) -> dict[str, Any]:
    """Republish unchanged oracle weights after a threshold-only gate revision.

    The source checkpoint and gate must form an intact artifact pair, and the new
    configuration may differ from the training configuration only in the maximum
    normalized factor MAE. This makes a post-training gate correction explicit
    without pretending that the model was retrained.
    """

    requested_destination = Path(output_dir).expanduser()
    destination = requested_destination.parent.resolve() / requested_destination.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Oracle destination already exists: {destination}")
    basis = str(scientific_basis).strip()
    if not basis:
        raise ValueError("scientific_basis must be a nonempty string.")

    source_checkpoint = Path(checkpoint).expanduser().resolve()
    source_gate_path = Path(gate_path).expanduser().resolve()
    load_factor_oracle(source_checkpoint, "cpu")
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    try:
        source_gate_bytes = source_gate_path.read_bytes()
        source_gate = json.loads(source_gate_bytes.decode("utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Oracle gate does not exist: {source_gate_path}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Oracle gate is not valid JSON.") from exc
    if not isinstance(source_gate, dict):
        raise ValueError("Oracle gate payload must be a dictionary.")
    expected_source_gate = dict(payload["metrics"])
    expected_source_gate["checkpoint_artifact_digest"] = payload["artifact_digest"]
    if source_gate != expected_source_gate:
        raise ValueError("Oracle gate must exactly match checkpoint metrics plus its digest.")

    source_config = payload.get("config")
    source_training_config = payload.get("training_config")
    if not isinstance(source_config, dict) or not isinstance(source_training_config, dict):
        raise ValueError("Oracle checkpoint configuration is malformed.")
    if source_training_config != source_config.get("oracle"):
        raise ValueError("Oracle checkpoint training_config does not match config.oracle.")
    source_comparison, prior_threshold = _config_without_oracle_threshold(source_config)
    revised_comparison, revised_threshold = _config_without_oracle_threshold(config)
    if source_comparison != revised_comparison:
        raise ValueError(
            "Requalification config may differ only in oracle.max_normalized_factor_mae."
        )
    source_gate_threshold = source_gate.get("thresholds", {}).get("max_normalized_factor_mae")
    if source_gate_threshold != prior_threshold:
        raise ValueError("Source oracle gate threshold does not match its training config.")
    if revised_threshold == prior_threshold:
        raise ValueError("Requalification requires a changed factor-MAE threshold.")

    parsed = _parse_config(config)
    object_configs = _object_configs(config)
    normalization = _factor_normalization(parsed.elevation_bounds)
    renderer_config = _renderer_config_payload(config, object_configs, normalization)
    renderer_hash = _stable_hash(renderer_config)
    if renderer_hash != payload["renderer_config_hash"]:
        raise ValueError("Requalification config changes the trained renderer contract.")

    gate = oracle_gate_metrics(
        object_accuracy=float(source_gate["object_accuracy"]),
        factor_mae=source_gate["factor_mae"],
        min_accuracy=parsed.min_accuracy,
        max_factor_mae=parsed.max_factor_mae,
        rerender_pixel_mae_q995=float(source_gate["validation_rerender_pixel_mae_q995"]),
    )
    configured_gate_passed = bool(gate["passed"])
    configured_failure_reasons = list(gate["failure_reasons"])
    gate_profile = "production" if _is_production_profile(config, parsed) else "fixture_only"
    production_qualified = gate_profile == "production" and configured_gate_passed
    qualification_reasons = list(configured_failure_reasons)
    if gate_profile != "production":
        qualification_reasons.append("fixture_only_profile")
    qualification_provenance = {
        "method": "threshold_only_requalification_without_retraining",
        "scientific_basis": basis,
        "source_checkpoint_artifact_digest": str(payload["artifact_digest"]),
        "source_gate_file_sha256": hashlib.sha256(source_gate_bytes).hexdigest(),
        "prior_max_normalized_factor_mae": float(prior_threshold),
        "revised_max_normalized_factor_mae": float(revised_threshold),
        "model_state_dict_unchanged": True,
    }
    gate.update(
        {
            "gate_profile": gate_profile,
            "configured_gate_passed": configured_gate_passed,
            "configured_failure_reasons": configured_failure_reasons,
            "production_qualified": production_qualified,
            "passed": production_qualified,
            "failure_reasons": qualification_reasons,
            "renderer_config_hash": renderer_hash,
            "seed": parsed.seed,
            "data_provenance": payload["data_provenance"],
            "qualification_provenance": qualification_provenance,
        }
    )

    revised_checkpoint = copy.deepcopy(payload)
    revised_checkpoint["training_config"] = copy.deepcopy(config["oracle"])
    revised_checkpoint["config"] = copy.deepcopy(config)
    revised_checkpoint["metrics"] = gate
    revised_checkpoint["qualification_provenance"] = qualification_provenance
    revised_checkpoint["artifact_digest"] = _checkpoint_artifact_digest(revised_checkpoint)
    published_gate = dict(gate)
    published_gate["checkpoint_artifact_digest"] = revised_checkpoint["artifact_digest"]
    checkpoint_path, published_gate_path = _publish_artifacts(
        destination,
        checkpoint=revised_checkpoint,
        gate=published_gate,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "gate_path": str(published_gate_path),
        "metrics": published_gate,
    }


def load_factor_oracle(
    checkpoint: str | Path,
    device: str | torch.device,
) -> SyntheticFactorOracle:
    """Load an oracle only after validating its complete reproducibility contract."""

    resolved_device = _resolve_device(device)
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Oracle checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Oracle checkpoint payload must be a dictionary.")
    stored_digest = payload.get("artifact_digest")
    if not isinstance(stored_digest, str) or len(stored_digest) != 64:
        raise ValueError("Oracle checkpoint artifact digest is missing or malformed.")
    try:
        computed_digest = _checkpoint_artifact_digest(payload)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Oracle checkpoint artifact digest cannot be recomputed.") from exc
    if not hmac.compare_digest(stored_digest, computed_digest):
        raise ValueError("Oracle checkpoint artifact digest mismatch.")
    required = {
        "artifact_digest",
        "schema_version",
        "architecture",
        "model_state_dict",
        "renderer_config_hash",
        "renderer_config",
        "seed",
        "factor_normalization",
        "object_ids",
        "training_config",
        "config",
        "data_provenance",
        "determinism",
        "metrics",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Oracle checkpoint missing required fields: {', '.join(missing)}")
    if payload["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported oracle checkpoint schema_version.")
    architecture = payload["architecture"]
    if not isinstance(architecture, dict) or architecture.get("name") != "SyntheticFactorOracle":
        raise ValueError("Oracle checkpoint has an incompatible architecture.")
    expected_architecture = _architecture_payload(int(architecture.get("num_classes", 0)))
    if architecture != expected_architecture:
        raise ValueError("Oracle checkpoint architecture metadata is incompatible.")
    if list(payload["object_ids"]) != list(OBJECT_IDS):
        raise ValueError("Oracle checkpoint object_ids are incompatible.")
    _validate_factor_normalization(payload["factor_normalization"])
    if _stable_hash(payload["renderer_config"]) != payload["renderer_config_hash"]:
        raise ValueError("Oracle checkpoint renderer config hash does not match its payload.")
    model = SyntheticFactorOracle(num_classes=architecture["num_classes"])
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("Oracle checkpoint model_state_dict is incompatible.") from exc
    model.to(resolved_device)
    model.eval()
    return model


def _parse_config(config: dict[str, Any]) -> _OracleConfig:
    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary.")
    seed = _required_int(config.get("seed"), "seed", minimum=0)
    image_size = _required_int(config.get("image_size"), "image_size", minimum=8)
    oracle = config.get("oracle", {})
    render = config.get("render", {})
    if not isinstance(oracle, Mapping):
        raise TypeError("oracle must be a mapping.")
    if not isinstance(render, Mapping):
        raise TypeError("render must be a mapping.")
    train_per_object = _required_int(
        oracle.get("training_samples_per_object", 30_000),
        "oracle.training_samples_per_object",
        minimum=1,
    )
    validation_per_object = _required_int(
        oracle.get("validation_samples_per_object", 5_000),
        "oracle.validation_samples_per_object",
        minimum=1,
    )
    batch_size = _required_int(oracle.get("batch_size", 256), "oracle.batch_size", minimum=1)
    steps = _required_int(oracle.get("steps", 20_000), "oracle.steps", minimum=1)
    learning_rate = _positive_finite(oracle.get("learning_rate", 1.0e-3), "oracle.learning_rate")
    min_accuracy = _number(oracle.get("min_object_accuracy", 0.99), "oracle.min_object_accuracy")
    _validate_unit_interval(min_accuracy, "oracle.min_object_accuracy")
    max_factor_mae = _number(
        oracle.get("max_normalized_factor_mae", 0.02),
        "oracle.max_normalized_factor_mae",
    )
    _validate_nonnegative_finite(max_factor_mae, "oracle.max_normalized_factor_mae")
    render_batch_size = _required_int(
        render.get("render_batch_size", 128), "render.render_batch_size", minimum=1
    )
    background = render.get("background", (1.0, 1.0, 1.0))
    if not isinstance(background, (list, tuple)) or len(background) != 3:
        raise ValueError("render.background must contain three numeric channels.")
    for channel in background:
        channel_value = _number(channel, "render.background")
        if not 0.0 <= channel_value <= 1.0:
            raise ValueError("render.background channels must lie in [0, 1].")
    _positive_finite(render.get("camera_distance", 4.0), "render.camera_distance")
    _required_int(render.get("supersample", 3), "render.supersample", minimum=1)
    bounds_raw = render.get("elevation_bounds_degrees", (-30.0, 30.0))
    if not isinstance(bounds_raw, (list, tuple)) or len(bounds_raw) != 2:
        raise ValueError("render.elevation_bounds_degrees must contain two values.")
    bounds_degrees = tuple(
        _number(value, "render.elevation_bounds_degrees") for value in bounds_raw
    )
    if not -90.0 < bounds_degrees[0] < bounds_degrees[1] < 90.0:
        raise ValueError("render.elevation_bounds_degrees must increase inside (-90, 90).")
    return _OracleConfig(
        seed=seed,
        image_size=image_size,
        train_per_object=train_per_object,
        validation_per_object=validation_per_object,
        batch_size=batch_size,
        steps=steps,
        learning_rate=learning_rate,
        min_accuracy=min_accuracy,
        max_factor_mae=max_factor_mae,
        render_batch_size=render_batch_size,
        elevation_bounds=tuple(math.radians(value) for value in bounds_degrees),
    )


def _factor_space(elevation_bounds: tuple[float, float]) -> ProductFactorSpace:
    return ProductFactorSpace(
        [
            BoundedTranslation(
                dim=3,
                bounds=_TRANSLATION_RANGES,
                name="translation_xyz",
            ),
            BoundedLookAtView(elevation_bounds=elevation_bounds),
        ],
        name="translation_xyz_bounded_view",
    )


def _factor_normalization(
    elevation_bounds: tuple[float, float],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, bounds in zip(("tx", "ty", "tz"), _TRANSLATION_RANGES, strict=True):
        midpoint = (bounds[0] + bounds[1]) / 2.0
        half_range = (bounds[1] - bounds[0]) / 2.0
        result[name] = {
            "world_range": [float(bounds[0]), float(bounds[1])],
            "midpoint": midpoint,
            "half_range": half_range,
            "normalized_range": [-1.0, 1.0],
        }
    result["azimuth"] = {
        "world_range_radians": [-math.pi, math.pi],
        "representation": "sin_cos",
        "mae_normalizer_radians": math.pi,
    }
    result["elevation"] = {
        "world_range_radians": [float(elevation_bounds[0]), float(elevation_bounds[1])],
        "midpoint_radians": float(sum(elevation_bounds) / 2.0),
        "half_range_radians": float((elevation_bounds[1] - elevation_bounds[0]) / 2.0),
        "normalized_range": [-1.0, 1.0],
    }
    return result


def _renderer_config_payload(
    config: dict[str, Any],
    object_configs: Mapping[str, dict[str, Any]],
    normalization: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    render = config.get("render", {})
    return {
        "backend": "analytic_RenderMap",
        "image_size": int(config["image_size"]),
        "background": [float(value) for value in render.get("background", (1.0, 1.0, 1.0))],
        "camera_distance": float(render.get("camera_distance", 4.0)),
        "supersample": int(render.get("supersample", 3)),
        "translation_target": "camera_plane",
        "objects": {name: dict(object_configs[name]) for name in OBJECT_IDS},
        "factor_normalization": normalization,
    }


def _render_independent_dataset(
    config: dict[str, Any],
    *,
    object_configs: Mapping[str, dict[str, Any]],
    factor: ProductFactorSpace,
    count_per_object: int,
    split_seed: int,
    parsed: _OracleConfig,
) -> _RenderedDataset:
    total = count_per_object * len(OBJECT_IDS)
    images = np.empty((total, 3, parsed.image_size, parsed.image_size), dtype=np.uint8)
    class_ids = np.empty(total, dtype=np.int64)
    translation = np.empty((total, 3), dtype=np.float32)
    view = np.empty((total, 3), dtype=np.float32)
    elevation_midpoint = sum(parsed.elevation_bounds) / 2.0
    elevation_half_range = (parsed.elevation_bounds[1] - parsed.elevation_bounds[0]) / 2.0
    for class_id, object_id in enumerate(OBJECT_IDS):
        start = class_id * count_per_object
        stop = start + count_per_object
        seed = split_seed + class_id * 1_000
        values = sample_values(factor.sample(count_per_object, seed=seed))
        render_map = _render_map(config, object_configs[object_id], factor)
        for batch_start in range(0, count_per_object, parsed.render_batch_size):
            batch_stop = min(count_per_object, batch_start + parsed.render_batch_size)
            rendered = np.asarray(
                render_map.render_batch(
                    values[batch_start:batch_stop],
                    batch_size=parsed.render_batch_size,
                ),
                dtype=np.float32,
            ).reshape(-1, parsed.image_size, parsed.image_size, 3)
            images[start + batch_start : start + batch_stop] = np.rint(
                np.clip(rendered.transpose(0, 3, 1, 2), 0.0, 1.0) * 255.0
            ).astype(np.uint8)
        rows = canonical_factor_rows(factor, values)
        if not np.all(np.isfinite(rows)):
            raise RuntimeError("High-dimensional oracle factor samples must define all factors.")
        class_ids[start:stop] = class_id
        translation[start:stop] = _normalize_translation(rows[:, :3])
        view[start:stop, 0] = np.sin(rows[:, 3])
        view[start:stop, 1] = np.cos(rows[:, 3])
        view[start:stop, 2] = (rows[:, 4] - elevation_midpoint) / elevation_half_range
    return _RenderedDataset(images, class_ids, translation, view)


def _train_steps(
    model: SyntheticFactorOracle,
    optimizer: torch.optim.Optimizer,
    data: _RenderedDataset,
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> None:
    model.train()
    order = torch.randperm(len(data.class_ids), generator=generator)
    position = 0
    for _ in range(steps):
        if position + batch_size > len(order):
            order = torch.randperm(len(data.class_ids), generator=generator)
            position = 0
        indices = order[position : min(position + batch_size, len(order))]
        position += len(indices)
        index_array = indices.numpy()
        images = (
            torch.from_numpy(data.images[index_array]).to(device=device, dtype=torch.float32)
            / 255.0
        )
        class_ids = torch.from_numpy(data.class_ids[index_array]).to(device=device)
        translation = torch.from_numpy(data.translation[index_array]).to(device=device)
        view = torch.from_numpy(data.view[index_array]).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        losses = oracle_loss(
            model(images),
            class_ids=class_ids,
            translation=translation,
            view=view,
        )
        losses["loss"].backward()
        optimizer.step()


def _validate_oracle(
    model: SyntheticFactorOracle,
    data: _RenderedDataset,
    *,
    config: dict[str, Any],
    object_configs: Mapping[str, dict[str, Any]],
    factor: ProductFactorSpace,
    parsed: _OracleConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    class_logits = []
    translations = []
    views = []
    with torch.no_grad():
        for start in range(0, len(data.class_ids), parsed.batch_size):
            stop = min(len(data.class_ids), start + parsed.batch_size)
            images = (
                torch.from_numpy(data.images[start:stop]).to(device=device, dtype=torch.float32)
                / 255.0
            )
            prediction = model(images)
            class_logits.append(prediction.class_logits.cpu())
            translations.append(prediction.translation.cpu())
            views.append(prediction.view.cpu())
    logits = torch.cat(class_logits)
    predicted_translation = torch.cat(translations)
    predicted_view = torch.cat(views)
    target_translation = torch.from_numpy(data.translation)
    target_view = torch.from_numpy(data.view)
    object_accuracy = float(
        (logits.argmax(dim=1) == torch.from_numpy(data.class_ids)).float().mean()
    )
    factor_mae = {
        "tx": float((predicted_translation[:, 0] - target_translation[:, 0]).abs().mean()),
        "ty": float((predicted_translation[:, 1] - target_translation[:, 1]).abs().mean()),
        "tz": float((predicted_translation[:, 2] - target_translation[:, 2]).abs().mean()),
        "azimuth": float(
            circular_vector_error(predicted_view[:, :2], target_view[:, :2]) / math.pi
        ),
        "elevation": float((predicted_view[:, 2] - target_view[:, 2]).abs().mean()),
    }
    rerender_residuals = _rerender_residuals(
        data.images,
        predicted_classes=logits.argmax(dim=1).numpy(),
        predicted_translation=predicted_translation.numpy(),
        predicted_view=predicted_view.numpy(),
        config=config,
        object_configs=object_configs,
        factor=factor,
        parsed=parsed,
    )
    return {
        "object_accuracy": object_accuracy,
        "factor_mae": factor_mae,
        "validation_rerender_pixel_mae_q995": float(np.quantile(rerender_residuals, 0.995)),
    }


def _rerender_residuals(
    target_images: np.ndarray,
    *,
    predicted_classes: np.ndarray,
    predicted_translation: np.ndarray,
    predicted_view: np.ndarray,
    config: dict[str, Any],
    object_configs: Mapping[str, dict[str, Any]],
    factor: ProductFactorSpace,
    parsed: _OracleConfig,
) -> np.ndarray:
    world_translation = _denormalize_translation(predicted_translation)
    elevation_midpoint = sum(parsed.elevation_bounds) / 2.0
    elevation_half_range = (parsed.elevation_bounds[1] - parsed.elevation_bounds[0]) / 2.0
    elevations = np.clip(
        elevation_midpoint + predicted_view[:, 2] * elevation_half_range,
        parsed.elevation_bounds[0],
        parsed.elevation_bounds[1],
    )
    azimuths = np.arctan2(predicted_view[:, 0], predicted_view[:, 1])
    render_maps = {
        class_id: _render_map(config, object_configs[object_id], factor)
        for class_id, object_id in enumerate(OBJECT_IDS)
    }
    residuals = np.empty(len(target_images), dtype=np.float64)
    translation_key, view_key = factor.factor_keys
    for index in range(len(target_images)):
        state = {
            translation_key: world_translation[index].astype(np.float32),
            view_key: np.asarray(
                [azimuths[index], math.sin(float(elevations[index]))],
                dtype=np.float32,
            ),
        }
        class_id = int(predicted_classes[index])
        rendered = render_maps[class_id].render(state).transpose(2, 0, 1)
        target = target_images[index].astype(np.float32) / 255.0
        residuals[index] = float(np.mean(np.abs(rendered - target)))
    return residuals


def _publish_artifacts(
    destination: Path,
    *,
    checkpoint: dict[str, Any],
    gate: dict[str, Any],
) -> tuple[Path, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    published = False
    try:
        torch.save(checkpoint, staging / _CHECKPOINT_FILENAME)
        write_json(gate, staging / _GATE_FILENAME)
        os.symlink(
            os.path.relpath(staging, start=destination.parent),
            destination,
            target_is_directory=True,
        )
        published = True
    except BaseException:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / _CHECKPOINT_FILENAME, destination / _GATE_FILENAME


def _checkpoint_artifact_digest(payload: Mapping[str, Any]) -> str:
    """Hash the complete local checkpoint contract, excluding only this digest.

    This detects accidental or internally inconsistent local artifacts. It is not
    a signature and does not establish an external publisher identity.
    """

    hasher = hashlib.sha256()
    _update_digest_blob(hasher, b"synthetic-factor-oracle-checkpoint-v2")
    if any(not isinstance(name, str) for name in payload):
        raise TypeError("Checkpoint field names must be strings.")
    metadata_items = sorted(
        (str(name), name) for name in payload if name not in {"artifact_digest", "model_state_dict"}
    )
    for key_label, original_key in metadata_items:
        _update_digest_blob(hasher, key_label.encode("utf-8"))
        _update_digest_blob(hasher, _canonical_json_bytes(payload[original_key]))
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("model_state_dict must be a mapping for artifact digesting.")
    if any(not isinstance(name, str) for name in state):
        raise TypeError("model_state_dict field names must be strings.")
    _update_digest_blob(hasher, b"model_state_dict")
    state_items = sorted((str(name), name) for name in state)
    for name_label, original_name in state_items:
        tensor = state[original_name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model_state_dict.{name_label} must be a tensor.")
        values = tensor.detach().cpu().contiguous()
        _update_digest_blob(hasher, name_label.encode("utf-8"))
        _update_digest_blob(hasher, str(values.dtype).encode("ascii"))
        _update_digest_blob(hasher, _canonical_json_bytes(list(values.shape)))
        raw = values.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")
        _update_digest_blob(hasher, raw)
    return hasher.hexdigest()


def _config_without_oracle_threshold(config: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    if not isinstance(config, Mapping):
        raise TypeError("Oracle configuration must be a mapping.")
    comparison = copy.deepcopy(dict(config))
    oracle = comparison.get("oracle")
    if not isinstance(oracle, dict):
        raise ValueError("Oracle configuration must contain an oracle mapping.")
    threshold = oracle.pop("max_normalized_factor_mae", None)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("oracle.max_normalized_factor_mae must be numeric.")
    _validate_nonnegative_finite(float(threshold), "oracle.max_normalized_factor_mae")
    return comparison, float(threshold)


def _update_digest_blob(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_digest_json_default,
    ).encode("utf-8")


def _digest_json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported checkpoint metadata type: {type(value).__name__}")


@contextmanager
def _deterministic_training_context(device: torch.device) -> Iterator[dict[str, Any]]:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cpu_rng = torch.get_rng_state()
    previous_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    previous_mps_rng = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    settings = _determinism_payload(device)
    try:
        yield settings
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.set_rng_state(previous_cpu_rng)
        if previous_cuda_rng is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng)
        if previous_mps_rng is not None:
            torch.mps.set_rng_state(previous_mps_rng)


def _determinism_payload(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    selected_cuda_name = None
    if device.type == "cuda":
        selected_cuda_name = torch.cuda.get_device_name(device)
    return {
        "resolved_device": str(device),
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": True,
        "warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "backend_context": {
            "cpu": {
                "available": True,
                "threads": int(torch.get_num_threads()),
                "interop_threads": int(torch.get_num_interop_threads()),
            },
            "cuda": {
                "available": cuda_available,
                "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
                "selected_device_name": selected_cuda_name,
                "runtime_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            },
            "mps": {
                "available": mps_available,
                "built": bool(torch.backends.mps.is_built()),
            },
        },
    }


def _is_production_profile(config: dict[str, Any], parsed: _OracleConfig) -> bool:
    profile = str(config.get("renderer_profile", "original_v1"))
    approved_objects = {
        "original_v1": [
            ("stepped_monument", 25.0, 1.0, 0.70, 0.12),
            ("crooked_arch", 145.0, 1.0, 0.70, 0.12),
            ("three_arm_vane", 265.0, 1.0, 0.70, 0.12),
        ],
        "calibrated_v2": [
            ("stepped_monument", 61.9380951, 1.02712272, 0.75872787, 0.12976866),
            ("crooked_arch", 143.54619021, 1.03138909, 0.74009267, 0.14495007),
            ("three_arm_vane", 225.49885742, 1.49318782, 0.76711206, 0.13290756),
        ],
    }
    if profile not in approved_objects:
        return False
    raw_objects = config.get("objects")
    if not isinstance(raw_objects, list) or len(raw_objects) != len(approved_objects[profile]):
        return False
    material = config.get("material", {})
    observed_objects = []
    try:
        common_lightness = float(material.get("oklch_lightness"))
        common_chroma = float(material.get("oklch_chroma"))
        for item in raw_objects:
            observed_objects.append(
                (
                    str(item["id"]),
                    float(item["hue_degrees"]),
                    float(item.get("scale", 1.0)),
                    float(item.get("oklch_lightness", common_lightness)),
                    float(item.get("oklch_chroma", common_chroma)),
                )
            )
    except (KeyError, TypeError, ValueError):
        return False
    render = config.get("render", {})
    try:
        canonical_material = common_lightness == 0.70 and common_chroma == 0.12
        expected_camera = 4.0 if profile == "original_v1" else 3.9
        expected_ambient = 0.35 if profile == "original_v1" else 0.6
        expected_diffuse = 0.70 if profile == "original_v1" else 0.4
        canonical_render = (
            [float(value) for value in render.get("background", [])] == [1.0, 1.0, 1.0]
            and float(render.get("camera_distance")) == expected_camera
            and [float(value) for value in render.get("elevation_bounds_degrees", [])]
            == [-30.0, 30.0]
            and int(render.get("supersample")) == 3
            and int(render.get("render_batch_size")) == 128
            and float(render.get("ambient", 0.35)) == expected_ambient
            and float(render.get("diffuse", 0.70)) == expected_diffuse
        )
    except (TypeError, ValueError):
        return False
    calibrated_v2_gate = True
    if profile == "calibrated_v2":
        calibration = config.get("calibration", {})
        try:
            calibrated_v2_gate = (
                float(calibration.get("finite_difference_epsilon", -1.0)) == 0.02
                and int(calibration.get("renderer_seed_offset", -1)) == 9_200_003
                and float(calibration.get("max_pullback_norm_ratio", -1.0)) == 4.25
                and list(calibration.get("blocking_checks", []))
                == ["object_separability", "renderer_rank", "factor_visibility"]
            )
        except (TypeError, ValueError):
            return False
    return (
        int(config.get("image_size", -1)) == 32
        and observed_objects == approved_objects[profile]
        and canonical_material
        and canonical_render
        and calibrated_v2_gate
        and parsed.train_per_object == 30_000
        and parsed.validation_per_object == 5_000
        and parsed.batch_size == 256
        and parsed.steps == 20_000
        and parsed.learning_rate == 1.0e-3
        and parsed.min_accuracy == 0.99
        and parsed.max_factor_mae == (0.02 if profile == "original_v1" else 0.08)
    )


def _validate_model_input(images: torch.Tensor, model: nn.Module) -> None:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor.")
    if images.ndim != 4:
        raise ValueError("images must have NCHW shape.")
    if images.shape[0] <= 0 or images.shape[1] != 3:
        raise ValueError("images must have a nonempty batch and three channels.")
    if images.shape[2] < 8 or images.shape[3] < 8:
        raise ValueError("image height and width must be at least 8.")
    if not torch.is_floating_point(images):
        raise TypeError("images must be floating point in [0, 1].")
    if images.device != next(model.parameters()).device:
        raise ValueError("images and oracle parameters must be on the same device.")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("images must be finite.")
    if bool((images < 0.0).any()) or bool((images > 1.0).any()):
        raise ValueError("images must lie in [0, 1].")


def _unit_azimuth(raw: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    fallback = torch.zeros_like(raw)
    fallback[:, 1] = 1.0
    return torch.where(norm > 1.0e-8, raw / norm.clamp_min(1.0e-8), fallback)


def _normalize_translation(values: np.ndarray) -> np.ndarray:
    lows = np.asarray([item[0] for item in _TRANSLATION_RANGES], dtype=np.float32)
    highs = np.asarray([item[1] for item in _TRANSLATION_RANGES], dtype=np.float32)
    centered = np.asarray(values, dtype=np.float32) - (lows + highs) / 2.0
    return (centered / ((highs - lows) / 2.0)).astype(np.float32)


def _denormalize_translation(values: np.ndarray) -> np.ndarray:
    lows = np.asarray([item[0] for item in _TRANSLATION_RANGES], dtype=np.float32)
    highs = np.asarray([item[1] for item in _TRANSLATION_RANGES], dtype=np.float32)
    normalized = np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)
    return normalized * ((highs - lows) / 2.0) + (lows + highs) / 2.0


def _architecture_payload(num_classes: int) -> dict[str, Any]:
    return {
        "name": "SyntheticFactorOracle",
        "num_classes": int(num_classes),
        "input_channels": 3,
        "feature_dim": 256,
    }


def _validate_factor_normalization(value: Any) -> None:
    if not isinstance(value, dict) or tuple(value) != FACTOR_NAMES:
        raise ValueError("Oracle factor_normalization is incompatible.")
    for name in FACTOR_NAMES:
        if not isinstance(value[name], dict):
            raise ValueError(f"Oracle factor_normalization.{name} is incompatible.")
    if value["azimuth"].get("representation") != "sin_cos":
        raise ValueError("Oracle factor_normalization azimuth representation is incompatible.")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str) and device.strip().lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid oracle device: {device!r}") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable.")
    if resolved.type == "cuda" and resolved.index is not None:
        if resolved.index < 0 or resolved.index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device index is unavailable: {resolved}")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable.")
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported oracle device type: {resolved.type}")
    if resolved.type in {"cpu", "mps"} and resolved.index not in {None, 0}:
        raise ValueError(f"Unsupported oracle device index: {resolved}")
    return resolved


def _required_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_finite(value: Any, name: str) -> float:
    result = _number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_unit_interval(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
