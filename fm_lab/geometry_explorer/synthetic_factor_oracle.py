"""Independent factor oracle for the synthetic long-tail geometry study."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
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
_CHECKPOINT_SCHEMA_VERSION = 1
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

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Oracle destination already exists: {destination}")
    parsed = _parse_config(config)
    resolved_device = _resolve_device(device)
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
    model = SyntheticFactorOracle(num_classes=len(OBJECT_IDS)).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parsed.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(parsed.seed + 30_000_000)
    _train_steps(
        model,
        optimizer,
        train_data,
        steps=parsed.steps,
        batch_size=parsed.batch_size,
        device=resolved_device,
        generator=generator,
    )
    metrics = _validate_oracle(
        model,
        validation_data,
        config=config,
        object_configs=object_configs,
        factor=factor,
        parsed=parsed,
        device=resolved_device,
    )
    gate = oracle_gate_metrics(
        object_accuracy=metrics["object_accuracy"],
        factor_mae=metrics["factor_mae"],
        min_accuracy=parsed.min_accuracy,
        max_factor_mae=parsed.max_factor_mae,
        rerender_pixel_mae_q995=metrics["validation_rerender_pixel_mae_q995"],
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
        "metrics": gate,
    }
    checkpoint_path, gate_path = _publish_artifacts(
        destination,
        checkpoint=checkpoint,
        gate=gate,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "gate_path": str(gate_path),
        "metrics": gate,
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
    required = {
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
    try:
        torch.save(checkpoint, staging / _CHECKPOINT_FILENAME)
        write_json(gate, staging / _GATE_FILENAME)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / _CHECKPOINT_FILENAME, destination / _GATE_FILENAME


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
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid oracle device: {device!r}") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable.")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable.")
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported oracle device type: {resolved.type}")
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
