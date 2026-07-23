"""Matched end-to-end sampling interventions for official ImbDiff-CM."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_intervention import (
    active_lora_modules,
    intervention_repeat_seed,
    radial_spectral_fractions,
    reversible_expert_intervention,
)
from fm_lab.diagnostics.imbdiff_cm_probe import RestoredImbDiffCMCheckpoint
from fm_lab.evaluation.groups import frequency_ranked_groups
from fm_lab.integrations.official_imbdiff_cm import sample_official_imbdiff

_SCHEMA_VERSION = 1


def matched_sampling_inputs(
    *,
    num_classes: int,
    samples_per_class: int,
    image_shape: Sequence[int],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one balanced label vector and one shared CPU noise tensor."""

    if int(num_classes) < 2:
        raise ValueError("Matched sampling requires at least two classes.")
    if int(samples_per_class) < 1:
        raise ValueError("samples_per_class must be positive.")
    shape = tuple(int(value) for value in image_shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("image_shape must contain three positive dimensions.")
    labels = torch.arange(int(num_classes), dtype=torch.long).repeat_interleave(
        int(samples_per_class)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn((len(labels), *shape), generator=generator, dtype=torch.float32)
    return labels, noise


def sample_matched_cm_interventions(
    restored: RestoredImbDiffCMCheckpoint,
    *,
    samples_per_class: int,
    batch_size: int,
    random_repeats: int,
    seed: int,
    input_seed: int | None = None,
    response_scales: Mapping[int, float] | None = None,
    mixed_precision: str = "off",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Sample learned, general-only, and randomized experts from identical noise."""

    if int(batch_size) < 1:
        raise ValueError("Matched sampling batch_size must be positive.")
    if int(random_repeats) < 1:
        raise ValueError("random_repeats must be positive.")
    objective = restored.objective
    model = restored.model
    class_counts = tuple(int(value) for value in objective.class_counts)
    image_shape = tuple(int(value) for value in objective.image_shape)
    resolved_input_seed = int(seed if input_seed is None else input_seed)
    labels, initial_noise = matched_sampling_inputs(
        num_classes=len(class_counts),
        samples_per_class=samples_per_class,
        image_shape=image_shape,
        seed=resolved_input_seed,
    )
    settings = _sampling_settings(restored)
    device = next(model.parameters()).device
    original_digest = _factor_digest(active_lora_modules(model))
    conditions: dict[str, torch.Tensor] = {}
    intervention_rows: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    try:
        conditions["learned"] = _sample_condition(
            model=model,
            labels=labels,
            initial_noise=initial_noise,
            batch_size=batch_size,
            device=device,
            settings=settings,
            mixed_precision=mixed_precision,
        )
        with reversible_expert_intervention(model, mode="zero", seed=seed) as manifest:
            conditions["general"] = _sample_condition(
                model=model,
                labels=labels,
                initial_noise=initial_noise,
                batch_size=batch_size,
                device=device,
                settings=settings,
                mixed_precision=mixed_precision,
            )
        intervention_rows.append(
            {
                "condition": "general",
                "mode": "zero",
                "seed": int(seed),
                "response_scale": 0.0,
                "restoration_verified": bool(manifest["restoration_verified"]),
            }
        )

        normalized_scales = {
            int(repeat): float(scale) for repeat, scale in (response_scales or {}).items()
        }
        for repeat in range(int(random_repeats)):
            condition = f"random_{repeat:02d}"
            repeat_seed = intervention_repeat_seed(seed, repeat)
            with reversible_expert_intervention(
                model,
                mode="spectrum_random",
                seed=repeat_seed,
            ) as manifest:
                response_scale = normalized_scales.get(repeat, 1.0)
                _scale_active_expert(model, response_scale)
                conditions[condition] = _sample_condition(
                    model=model,
                    labels=labels,
                    initial_noise=initial_noise,
                    batch_size=batch_size,
                    device=device,
                    settings=settings,
                    mixed_precision=mixed_precision,
                )
            intervention_rows.append(
                {
                    "condition": condition,
                    "mode": "spectrum_random",
                    "seed": int(repeat_seed),
                    "response_scale": float(response_scale),
                    "response_scale_source": (
                        "local_intervention_probe" if repeat in normalized_scales else "identity"
                    ),
                    "max_unscaled_spectrum_relative_error": max(
                        float(layer["spectrum_max_relative_error"]) for layer in manifest["layers"]
                    ),
                    "restoration_verified": bool(manifest["restoration_verified"]),
                }
            )
    finally:
        model.train(was_training)

    restored_digest = _factor_digest(active_lora_modules(model))
    if restored_digest != original_digest:
        raise RuntimeError("Matched sampling left the expert factors mutated.")
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint": {
            "path": str(restored.checkpoint_path),
            "step": int(restored.checkpoint_step),
            "method": restored.method,
            "weights": restored.weights,
        },
        "num_classes": len(class_counts),
        "samples_per_class": int(samples_per_class),
        "num_samples": int(len(labels)),
        "batch_size": int(batch_size),
        "random_repeats": int(random_repeats),
        "intervention_seed": int(seed),
        "input_seed": resolved_input_seed,
        "labels_sha256": _tensor_digest(labels),
        "initial_noise_sha256": _tensor_digest(initial_noise),
        "sampling": settings,
        "mixed_precision": str(mixed_precision),
        "conditions": intervention_rows,
        "original_factor_digest": original_digest,
        "restored_factor_digest": restored_digest,
        "restoration_verified": restored_digest == original_digest,
    }
    conditions["labels"] = labels
    conditions["initial_noise"] = initial_noise
    return conditions, manifest


def endpoint_response_scales(
    condition_samples: Mapping[str, torch.Tensor],
    *,
    base_scales: Mapping[int, float],
) -> tuple[dict[int, float], dict[str, Any]]:
    """Calibrate fixed random experts to the learned trajectory endpoint RMS."""

    learned = condition_samples["learned"].float()
    general = condition_samples["general"].float()
    learned_rms = float((learned - general).square().mean().sqrt())
    if not np.isfinite(learned_rms) or learned_rms <= 0.0:
        raise ValueError("Learned endpoint response RMS must be finite and positive.")
    calibrated = {}
    rows = []
    for repeat, base_scale in sorted(
        (int(key), float(value)) for key, value in base_scales.items()
    ):
        condition = f"random_{repeat:02d}"
        if condition not in condition_samples:
            raise ValueError(f"Endpoint calibration is missing condition {condition}.")
        random_samples = condition_samples[condition].float()
        if random_samples.shape != general.shape:
            raise ValueError("Endpoint calibration samples must have matching shapes.")
        random_rms = float((random_samples - general).square().mean().sqrt())
        if not np.isfinite(random_rms) or random_rms <= 0.0:
            raise ValueError("Random endpoint response RMS must be finite and positive.")
        endpoint_ratio = learned_rms / random_rms
        calibrated_scale = base_scale * endpoint_ratio
        calibrated[repeat] = calibrated_scale
        rows.append(
            {
                "random_repeat": repeat,
                "base_scale": base_scale,
                "learned_endpoint_rms": learned_rms,
                "random_endpoint_rms": random_rms,
                "endpoint_ratio": endpoint_ratio,
                "calibrated_scale": calibrated_scale,
            }
        )
    return calibrated, {
        "method": (
            "One independent matched DDIM pilot applies each locally calibrated random "
            "expert, then multiplies its fixed expert-weight scale by learned endpoint "
            "RMS divided by random endpoint RMS."
        ),
        "learned_endpoint_rms": learned_rms,
        "conditions": rows,
    }


def paired_sampling_effects(
    condition_samples: Mapping[str, torch.Tensor],
    *,
    labels: torch.Tensor,
    class_counts: Sequence[int],
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize paired output changes with class-cluster uncertainty."""

    learned = condition_samples["learned"].float()
    general = condition_samples["general"].float()
    if learned.shape != general.shape or len(learned) != len(labels):
        raise ValueError("Learned/general samples and labels must align.")
    random_names = sorted(name for name in condition_samples if name.startswith("random_"))
    if not random_names:
        raise ValueError("At least one random expert condition is required.")
    groups = frequency_ranked_groups(class_counts)
    group_by_class = {
        int(class_id): group_name
        for group_name, class_ids in groups.items()
        for class_id in class_ids
    }
    learned_delta = learned - general
    learned_rms = _per_sample_rms(learned_delta)
    learned_spectral = radial_spectral_fractions(learned_delta)
    random_rms = torch.stack(
        [_per_sample_rms(condition_samples[name].float() - general) for name in random_names]
    )
    learned_random_rms = torch.stack(
        [_per_sample_rms(learned - condition_samples[name].float()) for name in random_names]
    )
    rows: list[dict[str, Any]] = []
    for index, class_id_tensor in enumerate(labels):
        class_id = int(class_id_tensor)
        rows.append(
            {
                "sample_index": index,
                "class_id": class_id,
                "frequency_group": group_by_class[class_id],
                "learned_vs_general_rms": float(learned_rms[index]),
                "random_vs_general_rms_mean": float(random_rms[:, index].mean()),
                "learned_vs_random_rms_mean": float(learned_random_rms[:, index].mean()),
                **{
                    f"learned_vs_general_spectral_{name}": float(values[index])
                    for name, values in learned_spectral.items()
                },
            }
        )
    summaries = _paired_group_summaries(
        rows,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    return rows, summaries


def quality_contrasts(
    condition_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare learned quality with general and randomized expert conditions."""

    learned = condition_metrics["learned"]
    general = condition_metrics["general"]
    random_names = sorted(name for name in condition_metrics if name.startswith("random_"))
    if not random_names:
        raise ValueError("Quality contrasts require random expert conditions.")
    result: dict[str, Any] = {"overall": {}, "groups": {}}
    for metric in ("kid", "fid"):
        if metric not in learned or metric not in general:
            continue
        learned_value = float(learned[metric])
        general_value = float(general[metric])
        random_values = [
            float(condition_metrics[name][metric])
            for name in random_names
            if metric in condition_metrics[name]
        ]
        result["overall"][metric] = _lower_is_better_contrast(
            learned_value,
            general_value,
            random_values,
        )
    learned_groups = learned.get("groups", {})
    general_groups = general.get("groups", {})
    for group_name in ("many", "medium", "few"):
        if group_name not in learned_groups or group_name not in general_groups:
            continue
        result["groups"][group_name] = {}
        for metric in ("kid", "fid"):
            if metric not in learned_groups[group_name]:
                continue
            random_values = [
                float(condition_metrics[name]["groups"][group_name][metric])
                for name in random_names
                if metric in condition_metrics[name].get("groups", {}).get(group_name, {})
            ]
            result["groups"][group_name][metric] = _lower_is_better_contrast(
                float(learned_groups[group_name][metric]),
                float(general_groups[group_name][metric]),
                random_values,
            )
    if {"many", "few"}.issubset(result["groups"]):
        result["tail_selectivity"] = {}
        for metric in ("kid", "fid"):
            if metric not in result["groups"]["many"] or metric not in result["groups"]["few"]:
                continue
            result["tail_selectivity"][metric] = {
                "learned_gain_vs_general_few_minus_many": (
                    result["groups"]["few"][metric]["learned_gain_vs_general"]
                    - result["groups"]["many"][metric]["learned_gain_vs_general"]
                ),
                "learned_advantage_vs_random_few_minus_many": (
                    result["groups"]["few"][metric]["learned_advantage_vs_random_mean"]
                    - result["groups"]["many"][metric]["learned_advantage_vs_random_mean"]
                ),
            }
    result["paired_kid_subset_uncertainty"] = _paired_kid_uncertainty(condition_metrics)
    return result


def _sampling_settings(restored: RestoredImbDiffCMCheckpoint) -> dict[str, Any]:
    config = restored.config
    sampling = config.get("sampling", {}) or {}
    diffusion = config.get("diffusion", {}) or {}
    guidance = sampling.get("classifier_free_guidance", {}) or {}
    method = str(sampling.get("sampler", "ddim")).lower()
    if method != "ddim":
        raise ValueError("Matched CM intervention currently requires deterministic DDIM.")
    omega = float(guidance.get("paper_omega", guidance.get("omega", 1.5)))
    if not bool(guidance.get("enabled", True)):
        omega = 0.0
    return {
        "method": method,
        "ddim_skip": int(sampling.get("ddim_skip", 20)),
        "timesteps": int(diffusion.get("timesteps", 1000)),
        "beta_start": float(diffusion.get("beta_start", 1e-4)),
        "beta_end": float(diffusion.get("beta_end", 2e-2)),
        "variance": str(diffusion.get("variance", "fixed_large")).replace("_", ""),
        "omega": omega,
        "image_shape": [int(value) for value in restored.objective.image_shape],
        "sampler_family": str(getattr(restored.objective, "sampler_family", "cm")),
    }


def _sample_condition(
    *,
    model: torch.nn.Module,
    labels: torch.Tensor,
    initial_noise: torch.Tensor,
    batch_size: int,
    device: torch.device,
    settings: Mapping[str, Any],
    mixed_precision: str,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, len(labels), int(batch_size)):
        stop = min(start + int(batch_size), len(labels))
        with _autocast_context(device, mixed_precision):
            samples = sample_official_imbdiff(
                model=model,
                initial_noise=initial_noise[start:stop].to(device),
                class_labels=labels[start:stop].to(device),
                timesteps=int(settings["timesteps"]),
                beta_start=float(settings["beta_start"]),
                beta_end=float(settings["beta_end"]),
                variance=str(settings["variance"]),
                omega=float(settings["omega"]),
                method=str(settings["method"]),
                ddim_skip=int(settings["ddim_skip"]),
                image_shape=settings["image_shape"],
                sampler_family=str(settings["sampler_family"]),
            )
        chunks.append(samples.detach().float().cpu())
    return torch.cat(chunks)


def _scale_active_expert(model: torch.nn.Module, scale: float) -> None:
    if not np.isfinite(scale) or float(scale) < 0.0:
        raise ValueError("Expert response scale must be finite and non-negative.")
    with torch.no_grad():
        for _, module in active_lora_modules(model):
            module.lora_B.mul_(float(scale))


def _paired_group_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = (
        "learned_vs_general_rms",
        "random_vs_general_rms_mean",
        "learned_vs_random_rms_mean",
    )
    result = []
    for group_name in ("all", "many", "medium", "few"):
        selected = (
            list(rows)
            if group_name == "all"
            else [row for row in rows if row["frequency_group"] == group_name]
        )
        record: dict[str, Any] = {
            "frequency_group": group_name,
            "num_samples": len(selected),
            "num_classes": len({int(row["class_id"]) for row in selected}),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            record[f"{metric}_mean"] = float(values.mean())
            low, high = _class_bootstrap_mean(
                selected,
                metric=metric,
                repeats=bootstrap_repeats,
                seed=_stable_seed(seed, group_name, metric),
            )
            record[f"{metric}_class_bootstrap_low"] = low
            record[f"{metric}_class_bootstrap_high"] = high
        result.append(record)
    return result


def _class_bootstrap_mean(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    if int(repeats) < 1:
        raise ValueError("bootstrap_repeats must be positive.")
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(float(row[metric]))
    class_means = np.asarray([np.mean(values) for values in grouped.values()])
    rng = np.random.RandomState(int(seed) % np.iinfo(np.int32).max)
    draws = class_means[
        rng.randint(0, len(class_means), size=(int(repeats), len(class_means)))
    ].mean(1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _lower_is_better_contrast(
    learned: float,
    general: float,
    random_values: Sequence[float],
) -> dict[str, float]:
    random_mean = float(np.mean(random_values))
    return {
        "learned": float(learned),
        "general": float(general),
        "random_mean": random_mean,
        "learned_gain_vs_general": float(general - learned),
        "learned_advantage_vs_random_mean": float(random_mean - learned),
    }


def _paired_kid_uncertainty(
    condition_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    learned = condition_metrics["learned"]
    general = condition_metrics["general"]
    random_names = sorted(name for name in condition_metrics if name.startswith("random_"))
    result = {
        "interpretation": (
            "These intervals describe Monte Carlo variation across the fixed KID subset "
            "draws, paired by subset seed. They are not confidence intervals over training "
            "replicates, checkpoints, or random expert orientations."
        ),
        "overall": {},
        "groups": {},
    }
    if "kid_subset_estimates" in learned and "kid_subset_estimates" in general:
        result["overall"] = _kid_subset_contrast(
            learned["kid_subset_estimates"],
            general["kid_subset_estimates"],
            [condition_metrics[name]["kid_subset_estimates"] for name in random_names],
        )
    for group_name in ("many", "medium", "few"):
        learned_group = learned.get("groups", {}).get(group_name, {})
        general_group = general.get("groups", {}).get(group_name, {})
        if (
            "kid_subset_estimates" not in learned_group
            or "kid_subset_estimates" not in general_group
        ):
            continue
        result["groups"][group_name] = _kid_subset_contrast(
            learned_group["kid_subset_estimates"],
            general_group["kid_subset_estimates"],
            [
                condition_metrics[name]["groups"][group_name]["kid_subset_estimates"]
                for name in random_names
            ],
        )
    return result


def _kid_subset_contrast(
    learned: Sequence[float],
    general: Sequence[float],
    random: Sequence[Sequence[float]],
) -> dict[str, Any]:
    learned_values = np.asarray(learned, dtype=np.float64)
    general_values = np.asarray(general, dtype=np.float64)
    random_values = np.asarray(random, dtype=np.float64)
    if (
        learned_values.ndim != 1
        or general_values.shape != learned_values.shape
        or random_values.ndim != 2
        or random_values.shape[1:] != learned_values.shape
    ):
        raise ValueError("Paired KID subset estimates must have aligned shapes.")
    return {
        "learned_gain_vs_general": _difference_summary(general_values - learned_values),
        "learned_advantage_vs_random_mean": _difference_summary(
            random_values.mean(0) - learned_values
        ),
    }


def _difference_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "low": float(np.quantile(values, 0.025)),
        "high": float(np.quantile(values, 0.975)),
        "fraction_positive": float(np.mean(values > 0.0)),
        "num_subset_draws": int(len(values)),
    }


def _per_sample_rms(values: torch.Tensor) -> torch.Tensor:
    return values.square().flatten(1).mean(1).sqrt()


def _factor_digest(modules: Sequence[tuple[str, torch.nn.Module]]) -> str:
    digest = hashlib.sha256()
    for layer_name, module in modules:
        digest.update(layer_name.encode())
        for factor_name in ("lora_A", "lora_B"):
            factor = getattr(module, factor_name).detach().cpu().contiguous()
            digest.update(factor_name.encode())
            digest.update(str(tuple(factor.shape)).encode())
            digest.update(factor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _autocast_context(device: torch.device, mode: str):
    normalized = str(mode).lower()
    if normalized == "auto":
        normalized = "bf16" if device.type == "cuda" else "off"
    if normalized == "off":
        return nullcontext()
    if normalized not in {"bf16", "fp16"}:
        raise ValueError("mixed_precision must be auto, off, bf16, or fp16.")
    if device.type != "cuda":
        raise ValueError("CM sampling mixed precision is only supported on CUDA.")
    dtype = torch.bfloat16 if normalized == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
