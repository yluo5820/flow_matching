"""Distribution and quality metrics for the synthetic long-tail study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.linalg import LinAlgWarning, sqrtm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from fm_lab.geometry_explorer.latent_factors import sample_values
from fm_lab.geometry_explorer.synthetic_factor_oracle import (
    FACTOR_NAMES,
    _factor_space,
    _normalize_translation,
    _object_configs,
    _parse_config,
    _render_independent_dataset,
    _rerender_residuals,
    load_factor_oracle,
)
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    OBJECT_IDS,
    _render_map,
    canonical_factor_rows,
)
from fm_lab.utils.logging import write_json

_FACTOR_METRICS_FILENAME = "factor_metrics.json"
_CLASS_METRICS_FILENAME = "factor_metrics_by_class.csv"
_CONTROL_METRICS_FILENAME = "metric_controls.json"
_CONTROL_CLASS_FILENAME = "metric_controls_by_class.csv"
_CLASS_CSV_FIELDS = (
    "requested_class",
    "object_id",
    "requested_samples",
    "joint_valid_samples",
    "class_leakage_rate",
    "off_renderer_rate",
    "joint_valid_rate",
    "all_requested_status",
    "joint_valid_status",
    "reference_seed",
    "generated_seed",
    "source_revision",
    "generated_samples_sha256",
    "generated_labels_sha256",
    "oracle_checkpoint_digest",
    "renderer_config_hash",
    *(f"all_{name}_normalized_wasserstein" for name in FACTOR_NAMES),
    *(f"all_{name}_central_range_ratio" for name in FACTOR_NAMES),
    *(f"joint_{name}_normalized_wasserstein" for name in FACTOR_NAMES),
    *(f"joint_{name}_central_range_ratio" for name in FACTOR_NAMES),
    "all_multivariate_energy_distance",
    "joint_multivariate_energy_distance",
    "all_oracle_feature_fid",
    "joint_oracle_feature_fid",
)


def normalized_wasserstein(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    value_range: float,
) -> float:
    """Return scalar Wasserstein-1 distance normalized by a known factor range."""

    generated_values = _finite_vector(generated, "generated")
    reference_values = _finite_vector(reference, "reference")
    if not np.isfinite(value_range) or value_range <= 0.0:
        raise ValueError("value_range must be positive and finite.")
    return float(wasserstein_distance(generated_values, reference_values) / value_range)


def central_range_ratio(generated: np.ndarray, reference: np.ndarray) -> float:
    """Return the generated/reference ratio of central 90% factor widths."""

    generated_values = _finite_vector(generated, "generated")
    reference_values = _finite_vector(reference, "reference")
    generated_width = float(
        np.quantile(generated_values, 0.95) - np.quantile(generated_values, 0.05)
    )
    reference_width = float(
        np.quantile(reference_values, 0.95) - np.quantile(reference_values, 0.05)
    )
    return generated_width / max(reference_width, np.finfo(np.float64).eps)


def deterministic_subsample(
    values: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    """Select at most ``maximum`` rows deterministically, without replacement."""

    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a numpy array.")
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension.")
    if len(values) == 0:
        raise ValueError("values must not be empty.")
    if any(size == 0 for size in values.shape[1:]):
        raise ValueError("values must have non-empty feature dimensions.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("values must have a numeric dtype.")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("values must be finite.")
    if isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, (int, np.integer)):
        raise TypeError("maximum must be an integer.")
    if maximum <= 0:
        raise ValueError("maximum must be positive.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if len(values) <= maximum:
        return values
    indices = np.random.default_rng(int(seed)).choice(len(values), size=int(maximum), replace=False)
    return values[np.sort(indices)]


def multivariate_energy_distance(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    max_samples: int = 1_000,
    seed: int = 0,
) -> float:
    """Return the deterministic multivariate energy distance V-statistic."""

    generated_values, reference_values = _feature_matrices(generated, reference)
    generated_values = deterministic_subsample(generated_values, max_samples, seed)
    reference_values = deterministic_subsample(reference_values, max_samples, seed + 1)
    cross = float(cdist(generated_values, reference_values).mean())
    within_generated = float(cdist(generated_values, generated_values).mean())
    within_reference = float(cdist(reference_values, reference_values).mean())
    value = 2.0 * cross - within_generated - within_reference
    return float(max(value, 0.0))


def oracle_feature_fid(generated: np.ndarray, reference: np.ndarray) -> float:
    """Compute finite Fréchet distance with stable one-sample covariance handling."""

    generated_values, reference_values = _feature_matrices(generated, reference)
    generated_mean = generated_values.mean(axis=0)
    reference_mean = reference_values.mean(axis=0)
    generated_covariance = _covariance(generated_values)
    reference_covariance = _covariance(reference_values)
    generated_root = _real_sqrtm(generated_covariance)
    covariance_middle = generated_root @ reference_covariance @ generated_root
    covariance_middle = (covariance_middle + covariance_middle.T) / 2.0
    covariance_root = _real_sqrtm(covariance_middle)
    mean_term = float(np.sum(np.square(generated_mean - reference_mean)))
    covariance_term = float(
        np.trace(generated_covariance)
        + np.trace(reference_covariance)
        - 2.0 * np.trace(covariance_root)
    )
    value = mean_term + covariance_term
    if not np.isfinite(value):
        raise ValueError("oracle feature FID is not finite.")
    return float(max(value, 0.0))


def summarize_validity(
    *,
    predicted_class: np.ndarray,
    requested_class: np.ndarray,
    render_residual: np.ndarray,
    residual_threshold: float,
) -> dict[str, float]:
    """Report label leakage and renderer validity without discarding invalid mass."""

    predicted = _class_vector(predicted_class, "predicted_class")
    requested = _class_vector(requested_class, "requested_class")
    residual = _finite_vector(render_residual, "render_residual")
    if predicted.shape != requested.shape or predicted.shape != residual.shape:
        raise ValueError("validity inputs must have matching one-dimensional shapes.")
    if not np.isfinite(residual_threshold) or residual_threshold < 0.0:
        raise ValueError("residual_threshold must be non-negative and finite.")
    class_valid = predicted == requested
    renderer_valid = residual <= float(residual_threshold)
    return {
        "class_leakage_rate": float(1.0 - class_valid.mean()),
        "off_renderer_rate": float(1.0 - renderer_valid.mean()),
        "joint_valid_rate": float(np.mean(class_valid & renderer_valid)),
    }


def evaluate_generated_distribution(
    *,
    generated_root: str | Path,
    oracle_checkpoint: str | Path,
    oracle_gate: str | Path,
    output_dir: str | Path,
    device: str | torch.device,
    samples_per_class: int = 5_000,
    reference_samples_per_class: int | None = None,
    seed: int = 19_072_026,
    generated_seed: int = 19_072_026,
    required_gate_profile: str = "production",
    source_revision: str = "unknown",
    sample_value_range: tuple[float, float] = (-1.0, 1.0),
    inference_batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate generated samples against independent renderer references.

    The default enforces the frozen production count of 5,000 samples for each
    requested class. Tests and pilot checks may explicitly override it.
    """

    destination = _destination(output_dir, "Evaluation")
    count = _positive_integer(samples_per_class, "samples_per_class")
    reference_count = _positive_integer(
        reference_samples_per_class if reference_samples_per_class is not None else count,
        "reference_samples_per_class",
    )
    reference_seed = _nonnegative_integer(seed, "seed")
    sample_seed = _nonnegative_integer(generated_seed, "generated_seed")
    batch_size = _positive_integer(inference_batch_size, "inference_batch_size")
    revision = _nonempty_string(source_revision, "source_revision")
    samples_path, labels_path = _generated_paths(generated_root)
    samples = _load_numeric_array(samples_path, "generated samples")
    requested_labels = _load_requested_labels(labels_path, count)
    if len(samples) != len(requested_labels):
        raise ValueError("generated samples and requested labels must have matching lengths.")

    payload, gate, model = _validated_oracle(
        oracle_checkpoint,
        oracle_gate,
        device=device,
        required_gate_profile=required_gate_profile,
    )
    parsed = _parse_config(payload["config"])
    images = _normalize_generated_images(
        samples,
        image_size=parsed.image_size,
        value_range=sample_value_range,
    )
    generated_prediction = _infer(model, images, device=device, batch_size=batch_size)
    factor = _factor_space(parsed.elevation_bounds)
    object_configs = _object_configs(payload["config"])
    generated_uint8 = np.rint(images * 255.0).astype(np.uint8)
    render_residual = _rerender_residuals(
        generated_uint8,
        predicted_classes=generated_prediction["predicted_class"],
        predicted_translation=generated_prediction["translation"],
        predicted_view=generated_prediction["view"],
        config=payload["config"],
        object_configs=object_configs,
        factor=factor,
        parsed=parsed,
    )
    threshold = float(gate["off_renderer_threshold"])
    reference_data = _render_independent_dataset(
        payload["config"],
        object_configs=object_configs,
        factor=factor,
        count_per_object=reference_count,
        split_seed=reference_seed,
        parsed=parsed,
    )
    reference_images = reference_data.images.astype(np.float32) / 255.0
    reference_prediction = _infer(
        model,
        reference_images,
        device=device,
        batch_size=batch_size,
    )
    reference_class = np.asarray(reference_data.class_ids, dtype=np.int64)
    class_results = []
    for class_id, object_id in enumerate(OBJECT_IDS):
        generated_mask = requested_labels == class_id
        reference_mask = reference_class == class_id
        class_valid = generated_prediction["predicted_class"] == requested_labels
        renderer_valid = render_residual <= threshold
        joint_mask = generated_mask & class_valid & renderer_valid
        validity = summarize_validity(
            predicted_class=generated_prediction["predicted_class"][generated_mask],
            requested_class=requested_labels[generated_mask],
            render_residual=render_residual[generated_mask],
            residual_threshold=threshold,
        )
        all_metrics = _metric_bundle(
            generated_prediction,
            generated_mask,
            reference_prediction,
            reference_mask,
            seed=reference_seed + class_id * 100,
        )
        joint_count = int(np.count_nonzero(joint_mask))
        joint_result = (
            {
                "sample_count": 0,
                "status": "empty_joint_valid_subset",
                "metrics": None,
            }
            if joint_count == 0
            else {
                "sample_count": joint_count,
                "status": "ok",
                "metrics": _metric_bundle(
                    generated_prediction,
                    joint_mask,
                    reference_prediction,
                    reference_mask,
                    seed=reference_seed + class_id * 100 + 1,
                ),
            }
        )
        class_results.append(
            {
                "requested_class": class_id,
                "object_id": object_id,
                "validity": validity,
                "all_requested": {
                    "sample_count": int(np.count_nonzero(generated_mask)),
                    "status": "ok",
                    "metrics": all_metrics,
                },
                "joint_valid": joint_result,
            }
        )

    provenance = {
        "generated_seed": sample_seed,
        "reference_seed": reference_seed,
        "source_revision": revision,
        "generated_samples_path": str(samples_path),
        "generated_labels_path": str(labels_path),
        "generated_samples_sha256": _sha256_file(samples_path),
        "generated_labels_sha256": _sha256_file(labels_path),
        "oracle_checkpoint_path": str(Path(oracle_checkpoint).expanduser().resolve()),
        "oracle_checkpoint_digest": str(payload["artifact_digest"]),
        "oracle_gate_profile": str(gate["gate_profile"]),
        "renderer_config_hash": str(payload["renderer_config_hash"]),
        "reference_source": "independently_sampled_high_dimensional_renderer",
        "master_pool_reads": 0,
        "sample_value_range": [float(sample_value_range[0]), float(sample_value_range[1])],
        "oracle_input_range": [0.0, 1.0],
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "requested_labels": list(range(len(OBJECT_IDS))),
        "samples_per_requested_class": count,
        "reference_samples_per_class": reference_count,
        "off_renderer_threshold": threshold,
        "provenance": provenance,
        "classes": class_results,
        "artifacts": {
            "factor_metrics": _FACTOR_METRICS_FILENAME,
            "class_metrics": _CLASS_METRICS_FILENAME,
        },
    }
    rows = [_class_csv_row(item, provenance) for item in class_results]
    _publish_result(
        destination,
        json_name=_FACTOR_METRICS_FILENAME,
        csv_name=_CLASS_METRICS_FILENAME,
        result=result,
        rows=rows,
        fields=_CLASS_CSV_FIELDS,
    )
    return result


def calibrate_metric_controls(
    *,
    oracle_checkpoint: str | Path,
    oracle_gate: str | Path,
    output_dir: str | Path,
    device: str | torch.device,
    samples_per_class: int = 5_000,
    seed: int = 29_072_026,
    required_gate_profile: str = "production",
    source_revision: str = "unknown",
    inference_batch_size: int = 256,
) -> dict[str, Any]:
    """Render full, half-range, and collapsed positive distribution controls."""

    destination = _destination(output_dir, "Metric-control")
    count = _positive_integer(samples_per_class, "samples_per_class")
    control_seed = _nonnegative_integer(seed, "seed")
    batch_size = _positive_integer(inference_batch_size, "inference_batch_size")
    revision = _nonempty_string(source_revision, "source_revision")
    payload, gate, model = _validated_oracle(
        oracle_checkpoint,
        oracle_gate,
        device=device,
        required_gate_profile=required_gate_profile,
    )
    parsed = _parse_config(payload["config"])
    factor = _factor_space(parsed.elevation_bounds)
    object_configs = _object_configs(payload["config"])
    base_states: list[tuple[int, Any]] = []
    for class_id in range(len(OBJECT_IDS)):
        states = sample_values(factor.sample(count, seed=control_seed + class_id * 1_000))
        base_states.extend((class_id, state) for state in states)
    controls = {}
    rendered_by_control: dict[str, dict[str, np.ndarray]] = {}
    reference_truth = _control_truth(
        [state for _, state in base_states], elevation_bounds=parsed.elevation_bounds
    )
    for control_name, scale in (("full", 1.0), ("half", 0.5), ("collapsed", 0.0)):
        states = [_scaled_state(state, factor=factor, scale=scale) for _, state in base_states]
        images = _render_states(
            states,
            class_ids=np.asarray([class_id for class_id, _ in base_states], dtype=np.int64),
            config=payload["config"],
            object_configs=object_configs,
            factor=factor,
        )
        prediction = _infer(model, images, device=device, batch_size=batch_size)
        truth = _control_truth(states, elevation_bounds=parsed.elevation_bounds)
        rendered_by_control[control_name] = prediction
        controls[control_name] = {
            "factor_truth": _factor_metrics(truth, reference_truth),
            "oracle_inferred": None,
        }
    reference_prediction = rendered_by_control["full"]
    full_mask = np.ones(len(base_states), dtype=bool)
    for index, control_name in enumerate(("full", "half", "collapsed")):
        prediction = rendered_by_control[control_name]
        controls[control_name]["oracle_inferred"] = _metric_bundle(
            prediction,
            full_mask,
            reference_prediction,
            full_mask,
            seed=control_seed + index,
        )
    ordering = _control_ordering(controls)
    provenance = {
        "control_seed": control_seed,
        "source_revision": revision,
        "oracle_checkpoint_digest": str(payload["artifact_digest"]),
        "oracle_gate_profile": str(gate["gate_profile"]),
        "renderer_config_hash": str(payload["renderer_config_hash"]),
        "reference_source": "independently_sampled_high_dimensional_renderer",
        "master_pool_reads": 0,
    }
    result = {
        "schema_version": 1,
        "rendered_samples_per_control": len(base_states),
        "samples_per_class": count,
        "provenance": provenance,
        "controls": controls,
        "control_ordering": ordering,
        "artifacts": {
            "metric_controls": _CONTROL_METRICS_FILENAME,
            "class_metrics": _CONTROL_CLASS_FILENAME,
        },
    }
    rows = _control_csv_rows(controls, provenance, count)
    fields = tuple(rows[0])
    _publish_result(
        destination,
        json_name=_CONTROL_METRICS_FILENAME,
        csv_name=_CONTROL_CLASS_FILENAME,
        result=result,
        rows=rows,
        fields=fields,
    )
    return result


def _destination(output_dir: str | Path, label: str) -> Path:
    requested = Path(output_dir).expanduser()
    destination = requested.parent.resolve() / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"{label} destination already exists: {destination}")
    return destination


def _generated_paths(root: str | Path) -> tuple[Path, Path]:
    base = Path(root).expanduser().resolve()
    samples_dir = base / "samples" if (base / "samples").is_dir() else base
    samples_path = samples_dir / "euler_nfe64.npy"
    labels_path = samples_dir / "generated_labels.npy"
    if not samples_path.is_file():
        raise FileNotFoundError(f"Generated samples do not exist: {samples_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Generated labels do not exist: {labels_path}")
    return samples_path, labels_path


def _load_numeric_array(path: Path, label: str) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load {label}: {path}") from exc
    if not isinstance(values, np.ndarray) or values.ndim < 1 or len(values) == 0:
        raise ValueError(f"{label} must be a non-empty numpy array.")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{label} must have a numeric dtype.")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{label} must contain only finite values.")
    return values


def _load_requested_labels(path: Path, samples_per_class: int) -> np.ndarray:
    labels = _load_numeric_array(path, "generated labels")
    if labels.ndim != 1:
        raise ValueError("generated labels must have shape (N,).")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("generated labels must have an integer dtype.")
    result = labels.astype(np.int64, copy=False)
    expected = set(range(len(OBJECT_IDS)))
    if set(result.tolist()) != expected:
        raise ValueError(
            f"generated labels must contain exactly requested labels {sorted(expected)}."
        )
    for class_id in range(len(OBJECT_IDS)):
        actual = int(np.count_nonzero(result == class_id))
        if actual != samples_per_class:
            raise ValueError(
                f"generated labels must contain exactly {samples_per_class} samples "
                f"for requested class {class_id}; found {actual}."
            )
    return result


def _validated_oracle(
    checkpoint: str | Path,
    gate_path: str | Path,
    *,
    device: str | torch.device,
    required_gate_profile: str,
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module]:
    profile = _nonempty_string(required_gate_profile, "required_gate_profile")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    gate_file = Path(gate_path).expanduser().resolve()
    if not gate_file.is_file():
        raise FileNotFoundError(f"Oracle gate does not exist: {gate_file}")
    try:
        gate = json.loads(gate_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Oracle gate is not valid JSON: {gate_file}") from exc
    if not isinstance(gate, dict):
        raise ValueError("Oracle gate payload must be a dictionary.")
    model = load_factor_oracle(checkpoint_path, device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    digest = payload.get("artifact_digest")
    if gate.get("checkpoint_artifact_digest") != digest:
        raise ValueError("Oracle gate checkpoint digest does not match the checkpoint.")
    if gate.get("renderer_config_hash") != payload.get("renderer_config_hash"):
        raise ValueError("Oracle gate renderer config hash does not match the checkpoint.")
    if gate.get("gate_profile") != profile:
        raise ValueError(
            f"Oracle gate_profile must be {profile!r}; found {gate.get('gate_profile')!r}."
        )
    if profile == "production":
        if gate.get("production_qualified") is not True or gate.get("passed") is not True:
            raise ValueError("Oracle is not production-qualified.")
    elif gate.get("configured_gate_passed") is not True:
        raise ValueError("Oracle did not pass its configured fixture gate.")
    threshold = gate.get("off_renderer_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Oracle gate off_renderer_threshold is missing or invalid.")
    if not np.isfinite(threshold) or float(threshold) < 0.0:
        raise ValueError("Oracle gate off_renderer_threshold must be finite and non-negative.")
    return payload, gate, model


def _normalize_generated_images(
    samples: np.ndarray,
    *,
    image_size: int,
    value_range: tuple[float, float],
) -> np.ndarray:
    if not isinstance(value_range, tuple) or len(value_range) != 2:
        raise TypeError("sample_value_range must be a two-item tuple.")
    low, high = (float(value_range[0]), float(value_range[1]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("sample_value_range must be finite and increasing.")
    expected_flat = 3 * image_size * image_size
    if samples.ndim == 2 and samples.shape[1] == expected_flat:
        images = samples.reshape(len(samples), 3, image_size, image_size)
    elif samples.ndim == 4 and samples.shape[1:] == (3, image_size, image_size):
        images = samples
    elif samples.ndim == 4 and samples.shape[1:] == (image_size, image_size, 3):
        images = samples.transpose(0, 3, 1, 2)
    else:
        raise ValueError(
            "generated samples must have shape (N, 3*H*W), (N, 3, H, W), or (N, H, W, 3)."
        )
    converted = images.astype(np.float32, copy=False)
    tolerance = 1.0e-6
    if float(converted.min()) < low - tolerance or float(converted.max()) > high + tolerance:
        raise ValueError("generated samples fall outside sample_value_range.")
    normalized = (np.clip(converted, low, high) - low) / (high - low)
    if not bool(np.all(np.isfinite(normalized))):
        raise ValueError("normalized generated samples must be finite.")
    return normalized.astype(np.float32, copy=False)


def _infer(
    model: torch.nn.Module,
    images: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    resolved = next(model.parameters()).device
    requested = None if str(device).strip().lower() == "auto" else torch.device(device)
    if requested is not None and requested != resolved:
        raise ValueError("Oracle model device does not match requested inference device.")
    logits = []
    translation = []
    view = []
    features = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            tensor = torch.from_numpy(images[start : start + batch_size]).to(
                device=resolved, dtype=torch.float32
            )
            prediction = model(tensor)
            logits.append(prediction.class_logits.detach().cpu().numpy())
            translation.append(prediction.translation.detach().cpu().numpy())
            view.append(prediction.view.detach().cpu().numpy())
            features.append(prediction.features.detach().cpu().numpy())
    class_logits = np.concatenate(logits).astype(np.float64)
    translations = np.concatenate(translation).astype(np.float64)
    views = np.concatenate(view).astype(np.float64)
    feature_values = np.concatenate(features).astype(np.float64)
    for values, name in (
        (class_logits, "class logits"),
        (translations, "translation predictions"),
        (views, "view predictions"),
        (feature_values, "oracle features"),
    ):
        if not bool(np.all(np.isfinite(values))):
            raise ValueError(f"Oracle {name} must be finite.")
    if translations.shape != (len(images), 3) or views.shape != (len(images), 3):
        raise ValueError("Oracle factor prediction shapes are incompatible.")
    if feature_values.ndim != 2 or feature_values.shape[0] != len(images):
        raise ValueError("Oracle feature shape is incompatible.")
    azimuth_norm = np.linalg.norm(views[:, :2], axis=1)
    if not np.allclose(azimuth_norm, 1.0, atol=1.0e-4, rtol=0.0):
        raise ValueError("Oracle azimuth predictions must be unit circular vectors.")
    if np.any(np.abs(translations) > 1.0 + 1.0e-5) or np.any(np.abs(views[:, 2]) > 1.0 + 1.0e-5):
        raise ValueError("Oracle normalized factor predictions must lie in [-1, 1].")
    return {
        "predicted_class": class_logits.argmax(axis=1).astype(np.int64),
        "translation": translations,
        "view": views,
        "features": feature_values,
    }


def _factor_arrays(prediction: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    translation = prediction["translation"]
    view = prediction["view"]
    return {
        "tx": translation[:, 0],
        "ty": translation[:, 1],
        "tz": translation[:, 2],
        "azimuth": view[:, :2],
        "elevation": view[:, 2],
    }


def _factor_metrics(
    generated: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    result = {}
    for name in FACTOR_NAMES:
        if name == "azimuth":
            generated_angles = _circular_angles(generated[name])
            reference_angles = _circular_angles(reference[name])
            error = _circular_wasserstein(generated_angles, reference_angles) / (2.0 * math.pi)
            coverage = _circular_central_width(generated_angles) / max(
                _circular_central_width(reference_angles), np.finfo(np.float64).eps
            )
        else:
            error = normalized_wasserstein(
                np.asarray(generated[name]), np.asarray(reference[name]), value_range=2.0
            )
            coverage = central_range_ratio(np.asarray(generated[name]), np.asarray(reference[name]))
        result[name] = {
            "normalized_wasserstein": float(error),
            "central_range_ratio": float(coverage),
        }
    return result


def _metric_bundle(
    generated: Mapping[str, np.ndarray],
    generated_mask: np.ndarray,
    reference: Mapping[str, np.ndarray],
    reference_mask: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    generated_factors = {
        name: values[generated_mask] for name, values in _factor_arrays(generated).items()
    }
    reference_factors = {
        name: values[reference_mask] for name, values in _factor_arrays(reference).items()
    }
    generated_joint = _joint_factor_matrix(generated_factors)
    reference_joint = _joint_factor_matrix(reference_factors)
    return {
        "factor_metrics": _factor_metrics(generated_factors, reference_factors),
        "multivariate_energy_distance": multivariate_energy_distance(
            generated_joint, reference_joint, seed=seed
        ),
        "oracle_feature_fid": oracle_feature_fid(
            generated["features"][generated_mask], reference["features"][reference_mask]
        ),
    }


def _joint_factor_matrix(factors: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            factors["tx"],
            factors["ty"],
            factors["tz"],
            factors["azimuth"][:, 0],
            factors["azimuth"][:, 1],
            factors["elevation"],
        ]
    ).astype(np.float64)


def _circular_angles(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) == 0:
        raise ValueError("azimuth values must have non-empty shape (N, 2).")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1.0e-12) or not bool(np.all(np.isfinite(values))):
        raise ValueError("azimuth values must be finite nonzero circular vectors.")
    return np.mod(np.arctan2(values[:, 0], values[:, 1]), 2.0 * math.pi)


def _circular_wasserstein(generated: np.ndarray, reference: np.ndarray) -> float:
    points = np.unique(np.concatenate([generated, reference, [0.0, 2.0 * math.pi]]))
    points.sort()
    if len(points) < 2:
        return 0.0
    intervals = np.diff(points)
    left = points[:-1]
    generated_cdf = np.searchsorted(np.sort(generated), left, side="right") / len(generated)
    reference_cdf = np.searchsorted(np.sort(reference), left, side="right") / len(reference)
    difference = generated_cdf - reference_cdf
    offset = _weighted_median(difference, intervals)
    return float(np.sum(np.abs(difference - offset) * intervals))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    midpoint = float(ordered_weights.sum()) / 2.0
    index = int(np.searchsorted(np.cumsum(ordered_weights), midpoint, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _circular_central_width(angles: np.ndarray, mass: float = 0.9) -> float:
    ordered = np.sort(np.asarray(angles, dtype=np.float64))
    if len(ordered) == 1:
        return 0.0
    keep = max(1, int(math.ceil(mass * len(ordered))))
    extended = np.concatenate([ordered, ordered + 2.0 * math.pi])
    widths = extended[np.arange(len(ordered)) + keep - 1] - ordered
    return float(np.min(widths))


def _class_csv_row(item: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    all_metrics = item["all_requested"]["metrics"]
    joint_metrics = item["joint_valid"]["metrics"]
    row = {
        "requested_class": item["requested_class"],
        "object_id": item["object_id"],
        "requested_samples": item["all_requested"]["sample_count"],
        "joint_valid_samples": item["joint_valid"]["sample_count"],
        **item["validity"],
        "all_requested_status": item["all_requested"]["status"],
        "joint_valid_status": item["joint_valid"]["status"],
        "reference_seed": provenance["reference_seed"],
        "generated_seed": provenance["generated_seed"],
        "source_revision": provenance["source_revision"],
        "generated_samples_sha256": provenance["generated_samples_sha256"],
        "generated_labels_sha256": provenance["generated_labels_sha256"],
        "oracle_checkpoint_digest": provenance["oracle_checkpoint_digest"],
        "renderer_config_hash": provenance["renderer_config_hash"],
        "all_multivariate_energy_distance": all_metrics["multivariate_energy_distance"],
        "joint_multivariate_energy_distance": (
            "" if joint_metrics is None else joint_metrics["multivariate_energy_distance"]
        ),
        "all_oracle_feature_fid": all_metrics["oracle_feature_fid"],
        "joint_oracle_feature_fid": (
            "" if joint_metrics is None else joint_metrics["oracle_feature_fid"]
        ),
    }
    for prefix, metrics in (("all", all_metrics), ("joint", joint_metrics)):
        for name in FACTOR_NAMES:
            values = None if metrics is None else metrics["factor_metrics"][name]
            row[f"{prefix}_{name}_normalized_wasserstein"] = (
                "" if values is None else values["normalized_wasserstein"]
            )
            row[f"{prefix}_{name}_central_range_ratio"] = (
                "" if values is None else values["central_range_ratio"]
            )
    return row


def _scaled_state(state: Any, *, factor: Any, scale: float) -> dict[str, np.ndarray]:
    translation_key, view_key = factor.factor_keys
    translation = np.asarray(state[translation_key], dtype=np.float32) * scale
    original_view = np.asarray(state[view_key], dtype=np.float64)
    view = np.asarray(
        [
            original_view[0] * scale,
            math.sin(math.asin(float(np.clip(original_view[1], -1.0, 1.0))) * scale),
        ],
        dtype=np.float32,
    )
    return {translation_key: translation, view_key: view}


def _render_states(
    states: Sequence[Any],
    *,
    class_ids: np.ndarray,
    config: dict[str, Any],
    object_configs: Mapping[str, dict[str, Any]],
    factor: Any,
) -> np.ndarray:
    render_maps = {
        class_id: _render_map(config, object_configs[object_id], factor)
        for class_id, object_id in enumerate(OBJECT_IDS)
    }
    images = [
        np.asarray(render_maps[int(class_id)].render(state), dtype=np.float32).transpose(2, 0, 1)
        for state, class_id in zip(states, class_ids, strict=True)
    ]
    result = np.asarray(images, dtype=np.float32)
    if not bool(np.all(np.isfinite(result))) or np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("Control renderer produced invalid image values.")
    return result


def _control_truth(
    states: Sequence[Any],
    *,
    elevation_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    factor = _factor_space(elevation_bounds)
    rows = canonical_factor_rows(factor, states).astype(np.float64)
    translation = _normalize_translation(rows[:, :3]).astype(np.float64)
    midpoint = sum(elevation_bounds) / 2.0
    half_range = (elevation_bounds[1] - elevation_bounds[0]) / 2.0
    return {
        "tx": translation[:, 0],
        "ty": translation[:, 1],
        "tz": translation[:, 2],
        "azimuth": np.column_stack([np.sin(rows[:, 3]), np.cos(rows[:, 3])]),
        "elevation": (rows[:, 4] - midpoint) / half_range,
    }


def _control_ordering(controls: Mapping[str, Any]) -> dict[str, Any]:
    checks = {}
    for name in FACTOR_NAMES:
        errors = [
            controls[control]["factor_truth"][name]["normalized_wasserstein"]
            for control in ("full", "half", "collapsed")
        ]
        coverage = [
            controls[control]["factor_truth"][name]["central_range_ratio"]
            for control in ("full", "half", "collapsed")
        ]
        checks[name] = {
            "error_full_lt_half_lt_collapsed": errors[0] < errors[1] < errors[2],
            "coverage_full_gt_half_gt_collapsed": coverage[0] > coverage[1] > coverage[2],
        }
    return {
        "passed": all(all(values.values()) for values in checks.values()),
        "checks": checks,
    }


def _control_csv_rows(
    controls: Mapping[str, Any],
    provenance: Mapping[str, Any],
    samples_per_class: int,
) -> list[dict[str, Any]]:
    rows = []
    for control_name in ("full", "half", "collapsed"):
        row = {
            "control": control_name,
            "samples_per_class": samples_per_class,
            "control_seed": provenance["control_seed"],
            "source_revision": provenance["source_revision"],
            "oracle_checkpoint_digest": provenance["oracle_checkpoint_digest"],
            "renderer_config_hash": provenance["renderer_config_hash"],
        }
        for name in FACTOR_NAMES:
            values = controls[control_name]["factor_truth"][name]
            row[f"{name}_normalized_wasserstein"] = values["normalized_wasserstein"]
            row[f"{name}_central_range_ratio"] = values["central_range_ratio"]
        oracle_metrics = controls[control_name]["oracle_inferred"]
        row["oracle_multivariate_energy_distance"] = oracle_metrics["multivariate_energy_distance"]
        row["oracle_feature_fid"] = oracle_metrics["oracle_feature_fid"]
        rows.append(row)
    return rows


def _publish_result(
    destination: Path,
    *,
    json_name: str,
    csv_name: str,
    result: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    published = False
    try:
        write_json(result, staging / json_name)
        with (staging / csv_name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return int(value)


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype.")
    converted = array.astype(np.float64, copy=False)
    if not bool(np.all(np.isfinite(converted))):
        raise ValueError(f"{name} must be finite.")
    return converted


def _class_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("validity inputs must have matching one-dimensional shapes.")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    return array.astype(np.int64, copy=False)


def _feature_matrices(
    generated: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    for values, name in ((generated, "generated"), (reference, "reference")):
        if not isinstance(values, np.ndarray):
            raise TypeError(f"{name} must be a numpy array.")
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(f"{name} must have non-empty shape (N, D).")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"{name} must have a numeric dtype.")
        converted = values.astype(np.float64, copy=False)
        if not bool(np.all(np.isfinite(converted))):
            raise ValueError(f"{name} must be finite.")
        matrices.append(converted)
    if matrices[0].shape[1] != matrices[1].shape[1]:
        raise ValueError("generated and reference feature dimensions must match.")
    return matrices[0], matrices[1]


def _covariance(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    covariance = np.asarray(np.cov(values, rowvar=False), dtype=np.float64)
    return covariance.reshape(values.shape[1], values.shape[1])


def _real_sqrtm(matrix: np.ndarray) -> np.ndarray:
    if not bool(np.any(matrix)):
        return np.zeros_like(matrix, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LinAlgWarning)
        root = np.asarray(sqrtm(matrix))
    if np.iscomplexobj(root):
        residue = float(np.max(np.abs(root.imag)))
        if residue >= 1.0e-6:
            raise ValueError(f"matrix square root has non-negligible imaginary residue: {residue}")
        root = root.real
    result = np.asarray(root, dtype=np.float64)
    if not bool(np.all(np.isfinite(result))):
        raise ValueError("matrix square root is not finite.")
    return result


__all__ = [
    "calibrate_metric_controls",
    "central_range_ratio",
    "deterministic_subsample",
    "evaluate_generated_distribution",
    "multivariate_energy_distance",
    "normalized_wasserstein",
    "oracle_feature_fid",
    "summarize_validity",
]
