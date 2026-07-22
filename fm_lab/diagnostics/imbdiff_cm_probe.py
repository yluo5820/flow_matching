"""Mechanistic probes for the vendored official ImbDiff-CM implementation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from fm_lab.evaluation.groups import frequency_ranked_groups
from fm_lab.experiments.factory import build_model
from fm_lab.integrations.official_imbdiff_cm import (
    OfficialImbDiffCMProbeTerms,
    OfficialImbDiffObjective,
)
from fm_lab.training.losses import build_objective
from fm_lab.utils.checkpoints import load_checkpoint

_SCHEMA_VERSION = 1
_SPECTRAL_BANDS = (
    ("low", 0.0, 0.25),
    ("mid_low", 0.25, 0.5),
    ("mid_high", 0.5, 0.75),
    ("high", 0.75, math.nextafter(1.0, 2.0)),
)
_GRADIENT_COMPONENTS = ("base", "consistency", "diversity", "cm", "total")


@dataclass(frozen=True)
class ImbDiffCMProbeManifest:
    """Balanced held-out rows and all randomness used by a CM probe."""

    dataset_positions: np.ndarray
    original_indices: np.ndarray
    labels: np.ndarray
    timesteps: tuple[int, ...]
    noise_seeds: np.ndarray
    transfer_seeds: np.ndarray
    samples_per_class: int
    seed: int
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        positions = np.asarray(self.dataset_positions, dtype=np.int64)
        original = np.asarray(self.original_indices, dtype=np.int64)
        labels = np.asarray(self.labels, dtype=np.int64)
        timesteps = tuple(int(value) for value in self.timesteps)
        noise_seeds = np.asarray(self.noise_seeds, dtype=np.int64)
        transfer_seeds = np.asarray(self.transfer_seeds, dtype=np.int64)
        if int(self.schema_version) != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported ImbDiff-CM probe manifest version: {self.schema_version}."
            )
        if int(self.samples_per_class) < 1:
            raise ValueError("samples_per_class must be positive.")
        if positions.ndim != 1 or len(positions) < 3:
            raise ValueError("Probe dataset_positions must be a non-empty vector.")
        if original.shape != positions.shape or labels.shape != positions.shape:
            raise ValueError("Probe positions, original indices, and labels must align.")
        if len(np.unique(positions)) != len(positions):
            raise ValueError("Probe dataset positions must be unique.")
        if not timesteps or len(set(timesteps)) != len(timesteps):
            raise ValueError("Probe timesteps must be a non-empty unique sequence.")
        if any(value < 0 for value in timesteps):
            raise ValueError("Probe timesteps must be non-negative.")
        if noise_seeds.shape != (len(timesteps), len(positions)):
            raise ValueError("Probe noise_seeds must have shape [timesteps, rows].")
        if transfer_seeds.shape != (len(timesteps),):
            raise ValueError("Probe transfer_seeds must have one value per timestep.")
        classes, counts = np.unique(labels, return_counts=True)
        if len(classes) < 3 or not np.all(counts == int(self.samples_per_class)):
            raise ValueError("Probe rows must contain samples_per_class rows for every class.")
        for name, values in (
            ("dataset_positions", positions),
            ("original_indices", original),
            ("labels", labels),
            ("noise_seeds", noise_seeds),
            ("transfer_seeds", transfer_seeds),
        ):
            values = np.ascontiguousarray(values).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "samples_per_class", int(self.samples_per_class))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def num_rows(self) -> int:
        return int(len(self.labels))

    @property
    def digest(self) -> str:
        payload = self.to_dict(include_digest=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "samples_per_class": self.samples_per_class,
            "seed": self.seed,
            "timesteps": list(self.timesteps),
            "dataset_positions": self.dataset_positions.tolist(),
            "original_indices": self.original_indices.tolist(),
            "labels": self.labels.tolist(),
            "noise_seeds": self.noise_seeds.tolist(),
            "transfer_seeds": self.transfer_seeds.tolist(),
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> ImbDiffCMProbeManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stored_digest = str(payload.pop("digest"))
        manifest = cls(**payload)
        if manifest.digest != stored_digest:
            raise ValueError("ImbDiff-CM probe manifest digest does not match its contents.")
        return manifest


@dataclass(frozen=True)
class RestoredImbDiffCMCheckpoint:
    """Official CM model, objective, and checkpoint provenance."""

    model: nn.Module
    objective: OfficialImbDiffObjective
    config: dict[str, Any]
    checkpoint_path: Path
    checkpoint_step: int
    method: str
    weights: str


def build_imbdiff_cm_probe_manifest(
    labels: np.ndarray | torch.Tensor,
    original_indices: Sequence[int | str],
    *,
    timesteps: Sequence[int],
    samples_per_class: int,
    seed: int,
) -> ImbDiffCMProbeManifest:
    """Select a balanced held-out probe set and freeze all random draws."""

    labels_array = np.asarray(labels, dtype=np.int64)
    original_array = np.asarray(original_indices, dtype=np.int64)
    if labels_array.ndim != 1 or original_array.shape != labels_array.shape:
        raise ValueError("Probe labels and original_indices must be aligned vectors.")
    if int(samples_per_class) < 1:
        raise ValueError("samples_per_class must be positive.")
    normalized_timesteps = tuple(int(value) for value in timesteps)
    if not normalized_timesteps:
        raise ValueError("At least one probe timestep is required.")
    rng = np.random.RandomState(int(seed))
    positions: list[int] = []
    for class_id in np.unique(labels_array):
        candidates = np.flatnonzero(labels_array == class_id)
        if len(candidates) < int(samples_per_class):
            raise ValueError(f"Class {class_id} lacks enough held-out probe rows.")
        positions.extend(
            int(value)
            for value in rng.permutation(candidates)[: int(samples_per_class)]
        )
    selected = np.asarray(positions, dtype=np.int64)
    max_seed = np.iinfo(np.int64).max
    noise_seeds = rng.randint(
        0,
        max_seed,
        size=(len(normalized_timesteps), len(selected)),
        dtype=np.int64,
    )
    transfer_seeds = rng.randint(
        0,
        max_seed,
        size=len(normalized_timesteps),
        dtype=np.int64,
    )
    return ImbDiffCMProbeManifest(
        dataset_positions=selected,
        original_indices=original_array[selected],
        labels=labels_array[selected],
        timesteps=normalized_timesteps,
        noise_seeds=noise_seeds,
        transfer_seeds=transfer_seeds,
        samples_per_class=int(samples_per_class),
        seed=int(seed),
    )


def materialize_probe_noise(
    seeds: Sequence[int] | np.ndarray,
    image_shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate device-independent noise rows from fixed CPU generators."""

    shape = tuple(int(value) for value in image_shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("image_shape must contain three positive dimensions.")
    rows: list[torch.Tensor] = []
    for seed in np.asarray(seeds, dtype=np.int64):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        rows.append(torch.randn(shape, generator=generator, dtype=dtype))
    if not rows:
        raise ValueError("At least one noise seed is required.")
    return torch.stack(rows)


def restore_imbdiff_cm_probe_checkpoint(
    checkpoint_path: str | Path,
    *,
    class_counts: Sequence[int],
    device: torch.device,
    weights: str = "ema",
    channels_last: bool = False,
    checkpoint_payload: dict[str, Any] | None = None,
) -> RestoredImbDiffCMCheckpoint:
    """Restore a raw or EMA official CM checkpoint for mechanism probing."""

    path = Path(checkpoint_path)
    payload = (
        load_checkpoint(path, map_location="cpu")
        if checkpoint_payload is None
        else checkpoint_payload
    )
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("CM probe checkpoint is missing its experiment config.")
    source_dim = int((config.get("source", {}) or {}).get("dim", 0))
    if source_dim < 1:
        raise ValueError("CM probe checkpoint config is missing source.dim.")
    model = build_model(config, dim=source_dim)
    if not bool(getattr(model, "is_official_imbdiff_cm", False)):
        raise ValueError("CM mechanism probes require an official ImbDiff-CM checkpoint.")
    normalized_weights = str(weights).lower()
    state_key = {
        "raw": "model_state_dict",
        "ema": "ema_model_state_dict",
    }.get(normalized_weights)
    if state_key is None:
        raise ValueError("weights must be 'raw' or 'ema'.")
    state_dict = payload.get(state_key)
    if not isinstance(state_dict, dict):
        raise ValueError(f"CM probe checkpoint is missing {state_key}.")
    model.load_state_dict(state_dict)
    model.to(device)
    if channels_last:
        model.to(memory_format=torch.channels_last)
    model.eval()
    objective = build_objective(
        config.get("objective", {}),
        diffusion_config=config.get("diffusion", {}),
        class_counts=class_counts,
    )
    if not isinstance(objective, OfficialImbDiffObjective) or not objective.uses_capacity_model:
        raise ValueError("CM mechanism probes require an official capacity objective.")
    return RestoredImbDiffCMCheckpoint(
        model=model,
        objective=objective,
        config=config,
        checkpoint_path=path,
        checkpoint_step=int(payload.get("step", -1)),
        method=objective.method,
        weights=normalized_weights,
    )


def radial_spectral_fractions(delta: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-sample radial Fourier-energy fractions for NCHW deltas."""

    if delta.ndim != 4:
        raise ValueError("Spectral CM deltas must be NCHW tensors.")
    height, width = delta.shape[-2:]
    fy = torch.fft.fftfreq(height, device=delta.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(width, device=delta.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    radius = radius / math.sqrt(0.5**2 + 0.5**2)
    power = torch.fft.fft2(delta.float(), norm="ortho").abs().square()
    total = power.flatten(1).sum(1).clamp_min(torch.finfo(power.dtype).tiny)
    fractions: dict[str, torch.Tensor] = {}
    for name, low, high in _SPECTRAL_BANDS:
        mask = (radius >= low) & (radius <= high) if name == "high" else (
            (radius >= low) & (radius < high)
        )
        band_energy = power[..., mask].sum(dim=(1, 2))
        fractions[name] = band_energy / total
    return fractions


def probe_imbdiff_cm_checkpoint(
    restored: RestoredImbDiffCMCheckpoint,
    *,
    clean_images: torch.Tensor,
    manifest: ImbDiffCMProbeManifest,
    class_counts: Sequence[int],
    mixed_precision: str = "off",
    compute_gradients: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run paired functional, spectral, and gradient probes for one checkpoint."""

    model = restored.model
    objective = restored.objective
    device = next(model.parameters()).device
    image_shape = tuple(int(value) for value in objective.image_shape)
    if clean_images.shape != (manifest.num_rows, *image_shape):
        raise ValueError("clean_images do not match the manifest and objective image shape.")
    clean = clean_images.to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(manifest.labels.copy()).to(device=device, dtype=torch.long)
    groups = frequency_ranked_groups(class_counts)
    group_by_class = {
        class_id: group_name
        for group_name, class_ids in groups.items()
        for class_id in class_ids
    }
    checkpoint_results: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    for timestep_index, timestep in enumerate(manifest.timesteps):
        if timestep >= objective.timesteps:
            raise ValueError(
                f"Probe timestep {timestep} exceeds schedule length {objective.timesteps}."
            )
        noise = materialize_probe_noise(
            manifest.noise_seeds[timestep_index],
            image_shape,
        ).to(device)
        discrete_t = torch.full(
            (manifest.num_rows,),
            int(timestep),
            device=device,
            dtype=torch.long,
        )
        with _autocast_context(device, mixed_precision):
            terms = objective.probe_terms(
                model=model,
                clean=clean,
                labels=labels,
                timesteps=discrete_t,
                noise=noise,
                transfer_seed=int(manifest.transfer_seeds[timestep_index]),
            )
        functional = _functional_probe_summary(
            terms,
            labels=labels,
            groups=groups,
            objective=objective,
            timestep=timestep,
        )
        gradient_summary = None
        if compute_gradients:
            gradient_summary = _gradient_probe_summary(
                model,
                terms,
                labels=labels,
                groups=groups,
                consistency_weight=objective.consistency_weight,
                diversity_weight=objective.diversity_weight,
            )
        checkpoint_results.append(
            {
                "timestep": int(timestep),
                "functional": functional,
                "gradients": gradient_summary,
            }
        )
        row_records.extend(
            _functional_row_records(
                terms,
                manifest=manifest,
                timestep_index=timestep_index,
                timestep=timestep,
                group_by_class=group_by_class,
                objective=objective,
            )
        )
        del terms
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint": {
            "path": str(restored.checkpoint_path),
            "step": restored.checkpoint_step,
            "method": restored.method,
            "weights": restored.weights,
        },
        "manifest_digest": manifest.digest,
        "class_counts": [int(value) for value in class_counts],
        "frequency_groups": groups,
        "mixed_precision": str(mixed_precision),
        "compute_gradients": bool(compute_gradients),
        "cm_weights": {
            "consistency": objective.consistency_weight,
            "diversity": objective.diversity_weight,
        },
        "timesteps": checkpoint_results,
    }
    return summary, row_records


def _functional_probe_summary(
    terms: OfficialImbDiffCMProbeTerms,
    *,
    labels: torch.Tensor,
    groups: Mapping[str, Sequence[int]],
    objective: OfficialImbDiffObjective,
    timestep: int,
) -> dict[str, dict[str, Any]]:
    delta = terms.capacity_on - terms.capacity_off
    spectral = radial_spectral_fractions(delta)
    off_mse = (terms.capacity_off - terms.target).square().flatten(1).mean(1)
    alpha_bar = objective._sqrt_alpha_bars[int(timestep)].square().to(
        device=delta.device,
        dtype=delta.dtype,
    )
    x0_scale_sq = (1.0 - alpha_bar) / alpha_bar.clamp_min(
        torch.finfo(delta.dtype).tiny
    )
    x0_delta_mse = x0_scale_sq * terms.distance_per_sample
    result: dict[str, dict[str, Any]] = {}
    masks = {"all": torch.ones_like(labels, dtype=torch.bool)}
    masks.update(
        {
            group_name: _class_mask(labels, class_ids)
            for group_name, class_ids in groups.items()
        }
    )
    for group_name, mask in masks.items():
        result[group_name] = {
            "num_rows": int(mask.sum()),
            "capacity_distance": _masked_mean(terms.distance_per_sample, mask),
            "epsilon_delta_rms": _masked_mean(
                terms.distance_per_sample.clamp_min(0).sqrt(),
                mask,
            ),
            "x0_delta_mse": _masked_mean(x0_delta_mse, mask),
            "base_mse_on": _masked_mean(terms.base_per_sample, mask),
            "base_mse_off": _masked_mean(off_mse, mask),
            "expert_mse_gain": _masked_mean(off_mse - terms.base_per_sample, mask),
            "consistency_raw": _masked_mean(terms.consistency_per_sample, mask),
            "diversity_raw": _masked_mean(terms.diversity_per_sample, mask),
            "cm_training_contribution": _masked_mean(
                objective.consistency_weight * terms.consistency_per_sample
                + objective.diversity_weight * terms.diversity_per_sample,
                mask,
            ),
            "total_loss": _masked_mean(terms.total_per_sample, mask),
            "spectral_energy_fraction": {
                name: _masked_mean(values, mask) for name, values in spectral.items()
            },
        }
    return result


def _functional_row_records(
    terms: OfficialImbDiffCMProbeTerms,
    *,
    manifest: ImbDiffCMProbeManifest,
    timestep_index: int,
    timestep: int,
    group_by_class: Mapping[int, str],
    objective: OfficialImbDiffObjective,
) -> list[dict[str, Any]]:
    delta = terms.capacity_on - terms.capacity_off
    spectral = radial_spectral_fractions(delta)
    off_mse = (terms.capacity_off - terms.target).square().flatten(1).mean(1)
    alpha_bar = float(objective._sqrt_alpha_bars[int(timestep)].square())
    x0_scale_sq = (1.0 - alpha_bar) / max(alpha_bar, np.finfo(np.float64).tiny)
    fields = {
        "capacity_distance": terms.distance_per_sample,
        "base_mse_on": terms.base_per_sample,
        "base_mse_off": off_mse,
        "expert_mse_gain": off_mse - terms.base_per_sample,
        "consistency_raw": terms.consistency_per_sample,
        "diversity_raw": terms.diversity_per_sample,
        "total_loss": terms.total_per_sample,
    }
    detached = {
        name: values.detach().float().cpu().numpy() for name, values in fields.items()
    }
    detached_spectral = {
        name: values.detach().float().cpu().numpy() for name, values in spectral.items()
    }
    records: list[dict[str, Any]] = []
    for row in range(manifest.num_rows):
        class_id = int(manifest.labels[row])
        record: dict[str, Any] = {
            "timestep": int(timestep),
            "class_id": class_id,
            "frequency_group": group_by_class[class_id],
            "dataset_position": int(manifest.dataset_positions[row]),
            "original_index": int(manifest.original_indices[row]),
            "noise_seed": int(manifest.noise_seeds[timestep_index, row]),
            "transfer_seed": int(manifest.transfer_seeds[timestep_index]),
            "x0_delta_mse": float(
                x0_scale_sq * detached["capacity_distance"][row]
            ),
        }
        record.update({name: float(values[row]) for name, values in detached.items()})
        record.update(
            {
                f"spectral_{name}": float(values[row])
                for name, values in detached_spectral.items()
            }
        )
        records.append(record)
    return records


def _gradient_probe_summary(
    model: nn.Module,
    terms: OfficialImbDiffCMProbeTerms,
    *,
    labels: torch.Tensor,
    groups: Mapping[str, Sequence[int]],
    consistency_weight: float,
    diversity_weight: float,
) -> dict[str, Any]:
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    parameters = tuple(parameter for _, parameter in named_parameters)
    if not parameters:
        raise ValueError("CM gradient probe requires trainable parameters.")
    group_names = tuple(groups)
    masks = tuple(
        _class_mask(labels, groups[group_name]) for group_name in group_names
    )
    loss_vectors = (
        torch.stack([terms.base_per_sample[mask].mean() for mask in masks]),
        torch.stack([terms.consistency_per_sample[mask].mean() for mask in masks]),
        torch.stack([terms.diversity_per_sample[mask].mean() for mask in masks]),
    )
    grad_outputs = torch.eye(
        len(group_names),
        device=labels.device,
        dtype=loss_vectors[0].dtype,
    )
    batched_gradients = tuple(
        torch.autograd.grad(
            loss_vector,
            parameters,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
            is_grads_batched=True,
        )
        for loss_vector in loss_vectors
    )
    result: dict[str, Any] = {}
    for group_index, group_name in enumerate(group_names):
        component_gradients = tuple(
            tuple(
                gradient[group_index] if gradient is not None else None
                for gradient in component
            )
            for component in batched_gradients
        )
        result[group_name] = summarize_gradient_components(
            named_parameters,
            component_gradients,
            consistency_weight=consistency_weight,
            diversity_weight=diversity_weight,
        )
    return result


def summarize_gradient_components(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    component_gradients: Sequence[Sequence[torch.Tensor | None]],
    *,
    consistency_weight: float,
    diversity_weight: float,
) -> dict[str, Any]:
    """Summarize exact base/CM gradients by general and expert parameters."""

    if len(component_gradients) != 3:
        raise ValueError("Expected base, consistency, and diversity gradients.")
    if any(len(gradients) != len(named_parameters) for gradients in component_gradients):
        raise ValueError("Gradient tuples must align with named_parameters.")
    coefficients = {
        "base": (1.0, 0.0, 0.0),
        "consistency": (0.0, 1.0, 0.0),
        "diversity": (0.0, 0.0, 1.0),
        "cm": (0.0, float(consistency_weight), float(diversity_weight)),
        "total": (1.0, float(consistency_weight), float(diversity_weight)),
    }
    grouped_indices: dict[str, list[int]] = {"general": [], "expert": []}
    expert_layers: dict[str, list[int]] = {}
    for index, (name, _) in enumerate(named_parameters):
        if _is_expert_parameter(name):
            grouped_indices["expert"].append(index)
            expert_layers.setdefault(_expert_layer_name(name), []).append(index)
        else:
            grouped_indices["general"].append(index)
    if not grouped_indices["expert"]:
        raise ValueError("CM gradient probe found no LoRA expert parameters.")
    group_summaries = {
        name: _gradient_group_summary(
            named_parameters,
            component_gradients,
            indices,
            coefficients,
        )
        for name, indices in grouped_indices.items()
    }
    layer_summaries = {
        name: _gradient_group_summary(
            named_parameters,
            component_gradients,
            indices,
            coefficients,
        )
        for name, indices in sorted(expert_layers.items())
    }
    expert_fraction: dict[str, float] = {}
    for component in _GRADIENT_COMPONENTS:
        general_norm = group_summaries["general"]["components"][component]["norm"]
        expert_norm = group_summaries["expert"]["components"][component]["norm"]
        denominator = general_norm**2 + expert_norm**2
        expert_fraction[component] = (
            expert_norm**2 / denominator if denominator > 0 else 0.0
        )
    return {
        "groups": group_summaries,
        "expert_layers": layer_summaries,
        "expert_gradient_energy_fraction": expert_fraction,
    }


def _gradient_group_summary(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    component_gradients: Sequence[Sequence[torch.Tensor | None]],
    indices: Sequence[int],
    coefficients: Mapping[str, tuple[float, float, float]],
) -> dict[str, Any]:
    parameter_count = sum(named_parameters[index][1].numel() for index in indices)
    if parameter_count < 1:
        return {
            "num_parameters": 0,
            "components": {
                name: {"norm": 0.0, "rms": 0.0} for name in coefficients
            },
            "cosines": {},
        }
    norm_squares = {name: 0.0 for name in coefficients}
    dots = {
        (left, right): 0.0
        for left_index, left in enumerate(coefficients)
        for right in tuple(coefficients)[left_index + 1 :]
    }
    for index in indices:
        parameter = named_parameters[index][1]
        base_vectors = tuple(
            gradients[index]
            if gradients[index] is not None
            else torch.zeros_like(parameter)
            for gradients in component_gradients
        )
        derived = {
            name: sum(
                coefficient * gradient
                for coefficient, gradient in zip(weights, base_vectors, strict=True)
            )
            for name, weights in coefficients.items()
        }
        for name, gradient in derived.items():
            norm_squares[name] += float(gradient.detach().float().square().sum().cpu())
        for pair in dots:
            dots[pair] += float(
                (derived[pair[0]].detach().float() * derived[pair[1]].detach().float())
                .sum()
                .cpu()
            )
    norms = {name: math.sqrt(max(value, 0.0)) for name, value in norm_squares.items()}
    cosines = {}
    for (left, right), dot in dots.items():
        denominator = norms[left] * norms[right]
        cosines[f"{left}__{right}"] = dot / denominator if denominator > 0 else 0.0
    return {
        "num_parameters": int(parameter_count),
        "components": {
            name: {
                "norm": norms[name],
                "rms": norms[name] / math.sqrt(parameter_count),
            }
            for name in coefficients
        },
        "cosines": cosines,
    }


def _autocast_context(device: torch.device, mode: str):
    normalized = str(mode).lower()
    if normalized == "auto":
        normalized = "bf16" if device.type == "cuda" else "off"
    if normalized == "off":
        return nullcontext()
    if normalized not in {"bf16", "fp16"}:
        raise ValueError("mixed_precision must be auto, off, bf16, or fp16.")
    if device.type != "cuda":
        raise ValueError("CM probe mixed precision is only supported on CUDA.")
    dtype = torch.bfloat16 if normalized == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _class_mask(labels: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for class_id in class_ids:
        mask |= labels == int(class_id)
    return mask


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if not len(selected):
        raise ValueError("CM probe group mask selected no rows.")
    return float(selected.detach().float().mean().cpu())


def _is_expert_parameter(name: str) -> bool:
    return name.endswith(".lora_A") or name.endswith(".lora_B")


def _expert_layer_name(name: str) -> str:
    return name.rsplit(".", 1)[0]
