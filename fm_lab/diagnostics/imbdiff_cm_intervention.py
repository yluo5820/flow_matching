"""Causal expert-branch interventions for official ImbDiff-CM checkpoints."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from fm_lab.diagnostics.imbdiff_cm_knowledge import ImbDiffCMKnowledgeManifest
from fm_lab.diagnostics.imbdiff_cm_probe import (
    RestoredImbDiffCMCheckpoint,
    materialize_probe_noise,
    radial_spectral_fractions,
)
from fm_lab.evaluation.groups import frequency_ranked_groups

_SCHEMA_VERSION = 1
_SPECTRAL_NAMES = ("low", "mid_low", "mid_high", "high")


@dataclass(frozen=True)
class _Stimulus:
    timestep_index: int
    timestep: int
    noisy: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor


def active_lora_modules(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    """Return every active official CM LoRA convolution in model order."""

    modules = tuple(
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Conv2d_LoRA" and int(getattr(module, "r", 0)) > 0
    )
    if not modules:
        raise ValueError("CM intervention found no active Conv2d_LoRA layers.")
    return modules


@contextmanager
def reversible_expert_intervention(
    model: nn.Module,
    *,
    mode: str,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Temporarily replace expert factors and restore them bit-exactly."""

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"zero", "spectrum_random"}:
        raise ValueError("Expert intervention mode must be zero or spectrum_random.")
    modules = active_lora_modules(model)
    originals = {
        name: (
            module.lora_A.detach().clone(),
            module.lora_B.detach().clone(),
        )
        for name, module in modules
    }
    original_digest = _factor_digest(modules)
    layer_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for layer_index, (layer_name, module) in enumerate(modules):
            if normalized_mode == "zero":
                module.lora_B.zero_()
                layer_rows.append(
                    {
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "factor_rank": int(module.lora_A.shape[0]),
                        "spectrum_max_relative_error": 0.0,
                    }
                )
                continue
            random_a, random_b, audit = _spectrum_matched_random_factors(
                module,
                seed=_stable_seed(seed, normalized_mode, layer_name),
            )
            module.lora_A.copy_(random_a)
            module.lora_B.copy_(random_b)
            layer_rows.append(
                {
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    **audit,
                }
            )
    manifest: dict[str, Any] = {
        "mode": normalized_mode,
        "seed": int(seed),
        "original_factor_digest": original_digest,
        "intervened_factor_digest": _factor_digest(modules),
        "layers": layer_rows,
        "restoration_verified": False,
    }
    try:
        yield manifest
    finally:
        with torch.no_grad():
            for layer_name, module in modules:
                original_a, original_b = originals[layer_name]
                module.lora_A.copy_(original_a)
                module.lora_B.copy_(original_b)
        restored_digest = _factor_digest(modules)
        manifest["restored_factor_digest"] = restored_digest
        manifest["restoration_verified"] = restored_digest == original_digest
        if not manifest["restoration_verified"]:
            raise RuntimeError("CM expert intervention failed bit-exact restoration.")


def probe_imbdiff_cm_intervention(
    restored: RestoredImbDiffCMCheckpoint,
    *,
    clean_images: torch.Tensor,
    manifest: ImbDiffCMKnowledgeManifest,
    class_counts: Sequence[int],
    batch_size: int,
    random_repeats: int,
    bootstrap_repeats: int,
    seed: int,
    mixed_precision: str = "off",
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Compare learned experts with general-only and spectrum-random controls."""

    if int(batch_size) < 1:
        raise ValueError("Intervention batch_size must be positive.")
    if int(random_repeats) < 1:
        raise ValueError("random_repeats must be positive.")
    if int(bootstrap_repeats) < 1:
        raise ValueError("bootstrap_repeats must be positive.")
    model = restored.model
    objective = restored.objective
    device = next(model.parameters()).device
    image_shape = tuple(int(value) for value in objective.image_shape)
    probe = manifest.probe
    if clean_images.shape != (probe.num_rows, *image_shape):
        raise ValueError("Intervention images do not match the knowledge manifest.")
    normalized_counts = tuple(int(value) for value in class_counts)
    if normalized_counts != tuple(objective.class_counts):
        raise ValueError("Intervention class counts do not match the objective.")

    groups = frequency_ranked_groups(normalized_counts)
    group_by_class = {
        int(class_id): group_name
        for group_name, class_ids in groups.items()
        for class_id in class_ids
    }
    clean = clean_images.to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(probe.labels.copy()).to(device=device, dtype=torch.long)
    stimuli = _materialize_stimuli(
        restored,
        clean=clean,
        labels=labels,
        manifest=manifest,
    )
    was_training = model.training
    model.eval()
    original_factor_digest = _factor_digest(active_lora_modules(model))
    try:
        with torch.no_grad():
            learned_predictions = {
                stimulus.timestep: _predict_batched(
                    model,
                    stimulus,
                    labels=labels,
                    use_capacity=True,
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                )
                for stimulus in stimuli
            }
            general_predictions = {
                stimulus.timestep: _predict_batched(
                    model,
                    stimulus,
                    labels=labels,
                    use_capacity=False,
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                )
                for stimulus in stimuli
            }
            with reversible_expert_intervention(
                model,
                mode="zero",
                seed=seed,
            ) as zero_manifest:
                validation_stimulus = stimuli[0]
                validation_rows = min(int(batch_size), probe.num_rows)
                zero_prediction = _predict_batched(
                    model,
                    _slice_stimulus(validation_stimulus, validation_rows),
                    labels=labels[:validation_rows],
                    use_capacity=True,
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                )
                zero_validation_max_abs = float(
                    (
                        zero_prediction
                        - general_predictions[validation_stimulus.timestep][:validation_rows]
                    )
                    .abs()
                    .max()
                    .cpu()
                )

            random_predictions: dict[int, list[torch.Tensor]] = {
                stimulus.timestep: [] for stimulus in stimuli
            }
            random_manifests: list[dict[str, Any]] = []
            for repeat in range(int(random_repeats)):
                repeat_seed = _stable_seed(seed, "repeat", repeat)
                with reversible_expert_intervention(
                    model,
                    mode="spectrum_random",
                    seed=repeat_seed,
                ) as random_manifest:
                    random_manifest["repeat"] = repeat
                    for stimulus in stimuli:
                        random_predictions[stimulus.timestep].append(
                            _predict_batched(
                                model,
                                stimulus,
                                labels=labels,
                                use_capacity=True,
                                batch_size=batch_size,
                                mixed_precision=mixed_precision,
                            )
                        )
                random_manifests.append(random_manifest)
    finally:
        model.train(was_training)

    restored_factor_digest = _factor_digest(active_lora_modules(model))
    if restored_factor_digest != original_factor_digest:
        raise RuntimeError("Intervention probe left the model expert factors mutated.")

    effect_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        timestep_effects, timestep_random = _effect_rows(
            stimulus,
            manifest=manifest,
            group_by_class=group_by_class,
            learned=learned_predictions[stimulus.timestep],
            general=general_predictions[stimulus.timestep],
            random_predictions=random_predictions[stimulus.timestep],
        )
        effect_rows.extend(timestep_effects)
        random_rows.extend(timestep_random)

    group_rows = _group_summary_rows(
        effect_rows,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    class_rows = _class_summary_rows(effect_rows)
    tail_rows = _tail_selectivity_rows(
        effect_rows,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    intervention_manifest = {
        "schema_version": _SCHEMA_VERSION,
        "original_factor_digest": original_factor_digest,
        "restored_factor_digest": restored_factor_digest,
        "restoration_verified": restored_factor_digest == original_factor_digest,
        "zero_validation_max_abs": zero_validation_max_abs,
        "zero_intervention": zero_manifest,
        "random_interventions": random_manifests,
    }
    max_spectrum_error = max(
        float(layer["spectrum_max_relative_error"])
        for item in random_manifests
        for layer in item["layers"]
    )
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint": {
            "path": str(restored.checkpoint_path),
            "step": int(restored.checkpoint_step),
            "method": restored.method,
            "weights": restored.weights,
        },
        "manifest_digest": manifest.digest,
        "timesteps": [int(value) for value in probe.timesteps],
        "num_rows_per_timestep": int(probe.num_rows),
        "random_repeats": int(random_repeats),
        "bootstrap_repeats": int(bootstrap_repeats),
        "mixed_precision": str(mixed_precision),
        "frequency_groups": groups,
        "zero_validation_max_abs": zero_validation_max_abs,
        "max_random_spectrum_relative_error": max_spectrum_error,
        "restoration_verified": intervention_manifest["restoration_verified"],
        "group_summary": group_rows,
        "tail_selectivity": tail_rows,
        "interpretation_boundary": (
            "These are paired causal effects on the released CM prediction and "
            "transferred training target at fixed noisy inputs. They do not by "
            "themselves establish end-to-end sampling quality or FID effects."
        ),
    }
    return (
        summary,
        effect_rows,
        random_rows,
        group_rows,
        class_rows,
        intervention_manifest,
    )


def _materialize_stimuli(
    restored: RestoredImbDiffCMCheckpoint,
    *,
    clean: torch.Tensor,
    labels: torch.Tensor,
    manifest: ImbDiffCMKnowledgeManifest,
) -> tuple[_Stimulus, ...]:
    objective = restored.objective
    probe = manifest.probe
    image_shape = tuple(int(value) for value in objective.image_shape)
    stimuli: list[_Stimulus] = []
    with torch.no_grad():
        for timestep_index, timestep in enumerate(probe.timesteps):
            if timestep >= objective.timesteps:
                raise ValueError(
                    f"Intervention timestep {timestep} exceeds the diffusion schedule."
                )
            noise = materialize_probe_noise(
                probe.noise_seeds[timestep_index],
                image_shape,
            ).to(device=clean.device, dtype=clean.dtype)
            timesteps = torch.full(
                (probe.num_rows,),
                int(timestep),
                device=clean.device,
                dtype=torch.long,
            )
            noisy, target = objective.probe_inputs(
                model=restored.model,
                clean=clean,
                labels=labels,
                timesteps=timesteps,
                noise=noise,
                transfer_seed=int(probe.transfer_seeds[timestep_index]),
            )
            stimuli.append(
                _Stimulus(
                    timestep_index=timestep_index,
                    timestep=int(timestep),
                    noisy=noisy,
                    target=target,
                    timesteps=timesteps,
                )
            )
    return tuple(stimuli)


def _predict_batched(
    model: nn.Module,
    stimulus: _Stimulus,
    *,
    labels: torch.Tensor,
    use_capacity: bool,
    batch_size: int,
    mixed_precision: str,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    device = stimulus.noisy.device
    for start in range(0, len(labels), int(batch_size)):
        stop = min(start + int(batch_size), len(labels))
        with _autocast_context(device, mixed_precision):
            output = model(
                stimulus.noisy[start:stop],
                stimulus.timesteps[start:stop],
                y=labels[start:stop],
                use_cm=bool(use_capacity),
            )
        outputs.append(output.detach().float())
    return torch.cat(outputs, dim=0)


def _slice_stimulus(stimulus: _Stimulus, stop: int) -> _Stimulus:
    return _Stimulus(
        timestep_index=stimulus.timestep_index,
        timestep=stimulus.timestep,
        noisy=stimulus.noisy[:stop],
        target=stimulus.target[:stop],
        timesteps=stimulus.timesteps[:stop],
    )


def _effect_rows(
    stimulus: _Stimulus,
    *,
    manifest: ImbDiffCMKnowledgeManifest,
    group_by_class: Mapping[int, str],
    learned: torch.Tensor,
    general: torch.Tensor,
    random_predictions: Sequence[torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    learned_delta = learned - general
    learned_mse = _per_sample_mse(learned, stimulus.target)
    general_mse = _per_sample_mse(general, stimulus.target)
    learned_delta_rms = _per_sample_rms(learned_delta)
    learned_spectral = radial_spectral_fractions(learned_delta)
    random_mse = torch.stack(
        [_per_sample_mse(prediction, stimulus.target) for prediction in random_predictions]
    )
    random_delta = torch.stack([prediction - general for prediction in random_predictions])
    random_delta_rms = random_delta.flatten(2).square().mean(2).sqrt()
    learned_flat = learned_delta.flatten(1)
    random_flat = random_delta.flatten(2)
    cosine = (random_flat * learned_flat[None, :, :]).sum(2) / (
        random_flat.norm(dim=2) * learned_flat.norm(dim=1)[None, :]
    ).clamp_min(torch.finfo(random_flat.dtype).tiny)
    random_spectral = {
        name: torch.stack([radial_spectral_fractions(delta)[name] for delta in random_delta])
        for name in _SPECTRAL_NAMES
    }

    random_rows: list[dict[str, Any]] = []
    for repeat in range(len(random_predictions)):
        for row_index in range(manifest.probe.num_rows):
            class_id = int(manifest.probe.labels[row_index])
            random_rows.append(
                {
                    **_row_identity(
                        manifest,
                        stimulus=stimulus,
                        row_index=row_index,
                        class_id=class_id,
                        group_by_class=group_by_class,
                    ),
                    "random_repeat": repeat,
                    "random_mse": float(random_mse[repeat, row_index].cpu()),
                    "random_gain_vs_general": float(
                        (general_mse[row_index] - random_mse[repeat, row_index]).cpu()
                    ),
                    "random_delta_rms": float(random_delta_rms[repeat, row_index].cpu()),
                    "random_delta_cosine_to_learned": float(cosine[repeat, row_index].cpu()),
                    **{
                        f"random_spectral_{name}": float(
                            random_spectral[name][repeat, row_index].cpu()
                        )
                        for name in _SPECTRAL_NAMES
                    },
                }
            )

    effect_rows: list[dict[str, Any]] = []
    for row_index in range(manifest.probe.num_rows):
        class_id = int(manifest.probe.labels[row_index])
        random_mse_mean = random_mse[:, row_index].mean()
        effect_rows.append(
            {
                **_row_identity(
                    manifest,
                    stimulus=stimulus,
                    row_index=row_index,
                    class_id=class_id,
                    group_by_class=group_by_class,
                ),
                "learned_mse": float(learned_mse[row_index].cpu()),
                "general_mse": float(general_mse[row_index].cpu()),
                "random_mse_mean": float(random_mse_mean.cpu()),
                "random_mse_std": float(random_mse[:, row_index].std(unbiased=False).cpu()),
                "learned_gain_vs_general": float(
                    (general_mse[row_index] - learned_mse[row_index]).cpu()
                ),
                "random_gain_vs_general": float((general_mse[row_index] - random_mse_mean).cpu()),
                "learned_advantage_vs_random": float(
                    (random_mse_mean - learned_mse[row_index]).cpu()
                ),
                "learned_delta_rms": float(learned_delta_rms[row_index].cpu()),
                "random_delta_rms_mean": float(random_delta_rms[:, row_index].mean().cpu()),
                "random_delta_cosine_to_learned_mean": float(cosine[:, row_index].mean().cpu()),
                **{
                    f"learned_spectral_{name}": float(learned_spectral[name][row_index].cpu())
                    for name in _SPECTRAL_NAMES
                },
                **{
                    f"random_spectral_{name}_mean": float(
                        random_spectral[name][:, row_index].mean().cpu()
                    )
                    for name in _SPECTRAL_NAMES
                },
            }
        )
    return effect_rows, random_rows


def _row_identity(
    manifest: ImbDiffCMKnowledgeManifest,
    *,
    stimulus: _Stimulus,
    row_index: int,
    class_id: int,
    group_by_class: Mapping[int, str],
) -> dict[str, Any]:
    return {
        "timestep": int(stimulus.timestep),
        "class_id": class_id,
        "coarse_id": int(manifest.coarse_labels[row_index]),
        "frequency_group": group_by_class[class_id],
        "dataset_position": int(manifest.probe.dataset_positions[row_index]),
        "original_index": int(manifest.probe.original_indices[row_index]),
        "noise_seed": int(manifest.probe.noise_seeds[stimulus.timestep_index, row_index]),
        "transfer_seed": int(manifest.probe.transfer_seeds[stimulus.timestep_index]),
    }


def _group_summary_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = (
        "learned_mse",
        "general_mse",
        "random_mse_mean",
        "learned_gain_vs_general",
        "random_gain_vs_general",
        "learned_advantage_vs_random",
        "learned_delta_rms",
        "random_delta_rms_mean",
        "random_delta_cosine_to_learned_mean",
    )
    timesteps = (-1, *sorted({int(row["timestep"]) for row in rows}))
    result: list[dict[str, Any]] = []
    for timestep in timesteps:
        timestep_rows = (
            list(rows)
            if timestep == -1
            else [row for row in rows if int(row["timestep"]) == timestep]
        )
        for group_name in ("all", "many", "medium", "few"):
            selected = (
                timestep_rows
                if group_name == "all"
                else [row for row in timestep_rows if row["frequency_group"] == group_name]
            )
            record: dict[str, Any] = {
                "timestep": timestep,
                "frequency_group": group_name,
                "num_rows": len(selected),
                "num_classes": len({int(row["class_id"]) for row in selected}),
            }
            record.update(
                {
                    f"{metric}_mean": float(np.mean([float(row[metric]) for row in selected]))
                    for metric in metrics
                }
            )
            for metric in (
                "learned_gain_vs_general",
                "learned_advantage_vs_random",
            ):
                interval = _cluster_bootstrap_interval(
                    selected,
                    metric=metric,
                    repeats=bootstrap_repeats,
                    seed=_stable_seed(seed, "group", timestep, group_name, metric),
                )
                record.update(
                    {
                        f"{metric}_class_bootstrap_low": interval["low"],
                        f"{metric}_class_bootstrap_high": interval["high"],
                        f"{metric}_class_bootstrap_probability_positive": interval[
                            "probability_positive"
                        ],
                    }
                )
            result.append(record)
    return result


def _class_summary_rows(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(row)
    result: list[dict[str, Any]] = []
    for class_id, selected in sorted(grouped.items()):
        result.append(
            {
                "class_id": class_id,
                "coarse_id": int(selected[0]["coarse_id"]),
                "frequency_group": selected[0]["frequency_group"],
                "num_rows": len(selected),
                "learned_gain_vs_general_mean": float(
                    np.mean([float(row["learned_gain_vs_general"]) for row in selected])
                ),
                "learned_advantage_vs_random_mean": float(
                    np.mean([float(row["learned_advantage_vs_random"]) for row in selected])
                ),
            }
        )
    return result


def _tail_selectivity_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    timesteps = (-1, *sorted({int(row["timestep"]) for row in rows}))
    result: list[dict[str, Any]] = []
    for timestep in timesteps:
        timestep_rows = (
            list(rows)
            if timestep == -1
            else [row for row in rows if int(row["timestep"]) == timestep]
        )
        for metric in (
            "learned_gain_vs_general",
            "learned_advantage_vs_random",
        ):
            interval = _cluster_bootstrap_group_difference(
                timestep_rows,
                metric=metric,
                positive_group="few",
                negative_group="many",
                repeats=bootstrap_repeats,
                seed=_stable_seed(seed, "tail", timestep, metric),
            )
            result.append(
                {
                    "timestep": timestep,
                    "metric": metric,
                    "contrast": "few_minus_many",
                    **interval,
                }
            )
    return result


def _cluster_bootstrap_interval(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    class_means = _class_means(rows, metric)
    values = np.asarray(list(class_means.values()), dtype=np.float64)
    rng = np.random.RandomState(int(seed) % np.iinfo(np.int32).max)
    draws = values[rng.randint(0, len(values), size=(int(repeats), len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0.0)),
    }


def _cluster_bootstrap_group_difference(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    positive_group: str,
    negative_group: str,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    positive = _class_means(
        [row for row in rows if row["frequency_group"] == positive_group],
        metric,
    )
    negative = _class_means(
        [row for row in rows if row["frequency_group"] == negative_group],
        metric,
    )
    positive_values = np.asarray(list(positive.values()), dtype=np.float64)
    negative_values = np.asarray(list(negative.values()), dtype=np.float64)
    rng = np.random.RandomState(int(seed) % np.iinfo(np.int32).max)
    positive_draws = positive_values[
        rng.randint(
            0,
            len(positive_values),
            size=(int(repeats), len(positive_values)),
        )
    ].mean(axis=1)
    negative_draws = negative_values[
        rng.randint(
            0,
            len(negative_values),
            size=(int(repeats), len(negative_values)),
        )
    ].mean(axis=1)
    draws = positive_draws - negative_draws
    return {
        "estimate": float(positive_values.mean() - negative_values.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0.0)),
    }


def _class_means(
    rows: Sequence[dict[str, Any]],
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(float(row[metric]))
    if not grouped:
        raise ValueError("Cluster bootstrap received no classes.")
    return {class_id: float(np.mean(values)) for class_id, values in grouped.items()}


def _spectrum_matched_random_factors(
    module: nn.Module,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    trained_left = module.lora_B.detach().float().cpu()
    trained_right = module.lora_A.detach().float().cpu()
    target_singular_values = _compact_product_singular_values(
        trained_left,
        trained_right,
    )
    factor_rank = int(target_singular_values.numel())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_left_basis, _ = torch.linalg.qr(
        torch.randn(
            trained_left.shape[0],
            factor_rank,
            generator=generator,
            dtype=torch.float32,
        ),
        mode="reduced",
    )
    random_right_basis, _ = torch.linalg.qr(
        torch.randn(
            trained_right.shape[1],
            factor_rank,
            generator=generator,
            dtype=torch.float32,
        ),
        mode="reduced",
    )
    singular_root = target_singular_values.clamp_min(0.0).sqrt()
    random_left = random_left_basis * singular_root[None, :]
    random_right = singular_root[:, None] * random_right_basis.T
    achieved_singular_values = _compact_product_singular_values(
        random_left,
        random_right,
    )
    spectral_scale = target_singular_values.max().clamp_min(
        torch.finfo(target_singular_values.dtype).tiny
    )
    spectrum_error = float(
        (achieved_singular_values - target_singular_values).abs().max() / spectral_scale
    )
    audit = {
        "factor_rank": factor_rank,
        "target_stable_rank": _stable_rank(target_singular_values),
        "random_stable_rank": _stable_rank(achieved_singular_values),
        "spectrum_max_relative_error": spectrum_error,
    }
    return (
        random_right.to(
            device=module.lora_A.device,
            dtype=module.lora_A.dtype,
        ),
        random_left.to(
            device=module.lora_B.device,
            dtype=module.lora_B.dtype,
        ),
        audit,
    )


def _compact_product_singular_values(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("Low-rank product factors have incompatible shapes.")
    _, left_triangular = torch.linalg.qr(left, mode="reduced")
    _, right_triangular = torch.linalg.qr(right.T, mode="reduced")
    return torch.linalg.svdvals(left_triangular @ right_triangular.T)


def _stable_rank(singular_values: torch.Tensor) -> float:
    squared = singular_values.square()
    return float(squared.sum() / squared.max().clamp_min(torch.finfo(squared.dtype).tiny))


def _factor_digest(modules: Sequence[tuple[str, nn.Module]]) -> str:
    digest = hashlib.sha256()
    for layer_name, module in modules:
        digest.update(layer_name.encode())
        for factor_name in ("lora_A", "lora_B"):
            factor = getattr(module, factor_name).detach().cpu().contiguous()
            digest.update(factor_name.encode())
            digest.update(str(tuple(factor.shape)).encode())
            digest.update(factor.numpy().tobytes())
    return digest.hexdigest()


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _per_sample_mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).square().flatten(1).mean(1)


def _per_sample_rms(values: torch.Tensor) -> torch.Tensor:
    return values.float().square().flatten(1).mean(1).sqrt()


def _autocast_context(device: torch.device, mode: str):
    normalized = str(mode).lower()
    if normalized == "auto":
        normalized = "bf16" if device.type == "cuda" else "off"
    if normalized == "off":
        return nullcontext()
    if normalized not in {"bf16", "fp16"}:
        raise ValueError("mixed_precision must be auto, off, bf16, or fp16.")
    if device.type != "cuda":
        raise ValueError("CM intervention mixed precision is only supported on CUDA.")
    dtype = torch.bfloat16 if normalized == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)
