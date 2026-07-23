"""Paired-forward dropout diagnostics for the official ImbDiff-CM objective."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from fm_lab.diagnostics.imbdiff_cm_probe import (
    ImbDiffCMProbeManifest,
    RestoredImbDiffCMCheckpoint,
    materialize_probe_noise,
)
from fm_lab.evaluation.groups import frequency_ranked_groups
from fm_lab.integrations.official_imbdiff_cm import OfficialImbDiffCMTerms

_SCHEMA_VERSION = 1
_CONDITIONS = (
    ("expert_plus_dropout", "independent", True, False),
    ("expert_paired_dropout", "paired", True, False),
    ("expert_without_dropout", "disabled", True, False),
    ("general_dropout_only", "independent", False, False),
)
_FUNCTIONAL_METRICS = (
    "branch_distance",
    "base_loss",
    "auxiliary_loss",
    "total_loss",
)


@dataclass(frozen=True)
class _GradientSignature:
    named_parameters: tuple[tuple[str, nn.Parameter], ...]
    gradients: tuple[torch.Tensor | None, ...]


def probe_imbdiff_cm_dropout(
    restored: RestoredImbDiffCMCheckpoint,
    *,
    clean_images: torch.Tensor,
    manifest: ImbDiffCMProbeManifest,
    class_counts: Sequence[int],
    repeats: int,
    seed: int,
    compute_gradients: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate expert and dropout contributions using fixed checkpoint inputs."""

    if int(repeats) < 1:
        raise ValueError("Dropout probe repeats must be positive.")
    objective = restored.objective
    model = restored.model
    device = next(model.parameters()).device
    image_shape = tuple(int(value) for value in objective.image_shape)
    if clean_images.shape != (manifest.num_rows, *image_shape):
        raise ValueError("Dropout probe clean_images do not match the manifest.")
    if tuple(int(value) for value in class_counts) != tuple(objective.class_counts):
        raise ValueError("Dropout probe class counts do not match the objective.")

    clean = clean_images.to(device=device, dtype=torch.float32)
    labels = torch.from_numpy(manifest.labels.copy()).to(device=device, dtype=torch.long)
    groups = frequency_ranked_groups(class_counts)
    group_masks = _group_masks(labels, groups)
    dropout_seeds = _dropout_seed_matrix(
        timesteps=len(manifest.timesteps),
        repeats=int(repeats),
        seed=int(seed),
    )

    was_training = model.training
    model.train()
    rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    try:
        for timestep_index, timestep in enumerate(manifest.timesteps):
            if timestep >= objective.timesteps:
                raise ValueError(
                    f"Dropout probe timestep {timestep} exceeds the diffusion schedule."
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
            for repeat_index, dropout_seed in enumerate(dropout_seeds[timestep_index]):
                for condition, mode, capacity_on, capacity_off in _CONDITIONS:
                    with _forked_seed(clean, int(dropout_seed)), torch.no_grad():
                        terms = objective.probe_terms(
                            model=model,
                            clean=clean,
                            labels=labels,
                            timesteps=discrete_t,
                            noise=noise,
                            transfer_seed=int(
                                manifest.transfer_seeds[timestep_index]
                            ),
                            dropout_mode=mode,
                            capacity_on_enabled=capacity_on,
                            capacity_off_enabled=capacity_off,
                        )
                    rows.extend(
                        _functional_rows(
                            terms,
                            timestep=int(timestep),
                            repeat_index=repeat_index,
                            dropout_seed=int(dropout_seed),
                            condition=condition,
                            groups=group_masks,
                        )
                    )
                    del terms

            if compute_gradients:
                gradient_rows.extend(
                    _gradient_rows(
                        model=model,
                        objective=objective,
                        clean=clean,
                        labels=labels,
                        timesteps=discrete_t,
                        noise=noise,
                        transfer_seed=int(manifest.transfer_seeds[timestep_index]),
                        dropout_seed=int(dropout_seeds[timestep_index, 0]),
                        timestep=int(timestep),
                    )
                )
    finally:
        model.train(was_training)

    functional_summary = _summarize_functional_rows(rows)
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint": {
            "path": str(restored.checkpoint_path),
            "step": int(restored.checkpoint_step),
            "method": restored.method,
            "weights": restored.weights,
        },
        "manifest_digest": manifest.digest,
        "class_counts": [int(value) for value in class_counts],
        "frequency_groups": {
            name: [int(value) for value in values] for name, values in groups.items()
        },
        "timesteps": [int(value) for value in manifest.timesteps],
        "repeats": int(repeats),
        "seed": int(seed),
        "dropout_seeds": dropout_seeds.tolist(),
        "conditions": [
            {
                "name": name,
                "dropout_mode": mode,
                "capacity_on_enabled": capacity_on,
                "capacity_off_enabled": capacity_off,
            }
            for name, mode, capacity_on, capacity_off in _CONDITIONS
        ],
        "functional": functional_summary,
        "descriptive_ratios": _descriptive_ratios(functional_summary),
        "gradients": gradient_rows,
        "interpretation_boundary": (
            "Squared branch distances are not additively decomposable: the reported "
            "dropout-only and paired/independent ratios are descriptive, not fractions "
            "of causal variance. Gradients are for mean branch distance on the first "
            "dropout repeat, not for the complete training objective."
        ),
    }
    return summary, rows


def _functional_rows(
    terms: OfficialImbDiffCMTerms,
    *,
    timestep: int,
    repeat_index: int,
    dropout_seed: int,
    condition: str,
    groups: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    auxiliary = terms.coefficient_per_sample * terms.distance_per_sample
    values = {
        "branch_distance": terms.distance_per_sample,
        "base_loss": terms.base_per_sample,
        "auxiliary_loss": auxiliary,
        "total_loss": terms.total_per_sample,
    }
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups.items():
        row = {
            "timestep": int(timestep),
            "repeat": int(repeat_index),
            "dropout_seed": int(dropout_seed),
            "condition": str(condition),
            "dropout_mode": terms.dropout_mode,
            "frequency_group": group_name,
            "num_rows": int(mask.sum().item()),
        }
        row.update(
            {
                metric: float(metric_values[mask].detach().mean().cpu())
                for metric, metric_values in values.items()
            }
        )
        rows.append(row)
    return rows


def _gradient_rows(
    *,
    model: nn.Module,
    objective: Any,
    clean: torch.Tensor,
    labels: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    transfer_seed: int,
    dropout_seed: int,
    timestep: int,
) -> list[dict[str, Any]]:
    signatures: dict[str, _GradientSignature] = {}
    for condition, mode, capacity_on, capacity_off in _CONDITIONS:
        with _forked_seed(clean, dropout_seed):
            terms = objective.probe_terms(
                model=model,
                clean=clean,
                labels=labels,
                timesteps=timesteps,
                noise=noise,
                transfer_seed=transfer_seed,
                dropout_mode=mode,
                capacity_on_enabled=capacity_on,
                capacity_off_enabled=capacity_off,
            )
        signatures[condition] = _distance_gradient_signature(model, terms)
        del terms

    reference = signatures["expert_plus_dropout"]
    rows: list[dict[str, Any]] = []
    for condition, signature in signatures.items():
        comparison = _compare_gradient_signatures(reference, signature)
        for parameter_group, values in comparison.items():
            rows.append(
                {
                    "timestep": int(timestep),
                    "dropout_seed": int(dropout_seed),
                    "condition": condition,
                    "parameter_group": parameter_group,
                    **values,
                }
            )
    return rows


def _distance_gradient_signature(
    model: nn.Module,
    terms: OfficialImbDiffCMTerms,
) -> _GradientSignature:
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    gradients = torch.autograd.grad(
        terms.distance_per_sample.mean(),
        tuple(parameter for _, parameter in named_parameters),
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return _GradientSignature(
        named_parameters=named_parameters,
        gradients=tuple(
            gradient.detach() if gradient is not None else None
            for gradient in gradients
        ),
    )


def _compare_gradient_signatures(
    reference: _GradientSignature,
    candidate: _GradientSignature,
) -> dict[str, dict[str, float | None]]:
    reference_names = tuple(name for name, _ in reference.named_parameters)
    candidate_names = tuple(name for name, _ in candidate.named_parameters)
    if reference_names != candidate_names:
        raise ValueError("Dropout gradient signatures use different parameters.")

    results: dict[str, dict[str, float | None]] = {}
    for group_name, expert_flag in (("general", False), ("expert", True), ("all", None)):
        reference_norm_sq = 0.0
        candidate_norm_sq = 0.0
        dot = 0.0
        parameter_count = 0
        for (name, parameter), reference_gradient, candidate_gradient in zip(
            reference.named_parameters,
            reference.gradients,
            candidate.gradients,
            strict=True,
        ):
            is_expert = name.endswith(".lora_A") or name.endswith(".lora_B")
            if expert_flag is not None and is_expert != expert_flag:
                continue
            parameter_count += parameter.numel()
            if reference_gradient is not None:
                reference_norm_sq += float(
                    reference_gradient.float().square().sum().cpu()
                )
            if candidate_gradient is not None:
                candidate_norm_sq += float(
                    candidate_gradient.float().square().sum().cpu()
                )
            if reference_gradient is not None and candidate_gradient is not None:
                dot += float(
                    (
                        reference_gradient.float()
                        * candidate_gradient.float()
                    )
                    .sum()
                    .cpu()
                )
        reference_norm = reference_norm_sq**0.5
        candidate_norm = candidate_norm_sq**0.5
        cosine = (
            dot / (reference_norm * candidate_norm)
            if reference_norm > 0 and candidate_norm > 0
            else None
        )
        results[group_name] = {
            "num_parameters": int(parameter_count),
            "reference_norm": reference_norm,
            "gradient_norm": candidate_norm,
            "rms": (
                candidate_norm / max(parameter_count, 1) ** 0.5
                if parameter_count
                else 0.0
            ),
            "cosine_to_independent_expert_plus_dropout": cosine,
        }
    general_sq = results["general"]["gradient_norm"] ** 2
    expert_sq = results["expert"]["gradient_norm"] ** 2
    denominator = general_sq + expert_sq
    for values in results.values():
        values["expert_gradient_energy_fraction"] = (
            expert_sq / denominator if denominator > 0 else 0.0
        )
    return results


def _summarize_functional_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (
                int(row["timestep"]),
                str(row["condition"]),
                str(row["frequency_group"]),
            )
            for row in rows
        }
    )
    summaries: list[dict[str, Any]] = []
    for timestep, condition, group in keys:
        selected = [
            row
            for row in rows
            if int(row["timestep"]) == timestep
            and str(row["condition"]) == condition
            and str(row["frequency_group"]) == group
        ]
        summary: dict[str, Any] = {
            "timestep": timestep,
            "condition": condition,
            "frequency_group": group,
            "repeats": len(selected),
            "num_rows": int(selected[0]["num_rows"]),
        }
        for metric in _FUNCTIONAL_METRICS:
            values = np.asarray([float(row[metric]) for row in selected])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
        summaries.append(summary)
    return summaries


def _descriptive_ratios(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            int(row["timestep"]),
            str(row["frequency_group"]),
            str(row["condition"]),
        ): float(row["branch_distance_mean"])
        for row in summaries
    }
    keys = sorted({(key[0], key[1]) for key in lookup})
    rows: list[dict[str, Any]] = []
    for timestep, group in keys:
        independent = lookup[(timestep, group, "expert_plus_dropout")]
        paired = lookup[(timestep, group, "expert_paired_dropout")]
        disabled = lookup[(timestep, group, "expert_without_dropout")]
        dropout_only = lookup[(timestep, group, "general_dropout_only")]
        denominator = max(independent, np.finfo(np.float64).tiny)
        rows.append(
            {
                "timestep": timestep,
                "frequency_group": group,
                "paired_to_independent_distance": paired / denominator,
                "disabled_to_independent_distance": disabled / denominator,
                "dropout_only_to_independent_distance": dropout_only / denominator,
            }
        )
    return rows


def _group_masks(
    labels: torch.Tensor,
    groups: Mapping[str, Sequence[int]],
) -> dict[str, torch.Tensor]:
    masks = {"all": torch.ones_like(labels, dtype=torch.bool)}
    for group_name, class_ids in groups.items():
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_id in class_ids:
            mask |= labels == int(class_id)
        if bool(mask.any()):
            masks[group_name] = mask
    return masks


def _dropout_seed_matrix(*, timesteps: int, repeats: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(int(seed))
    return rng.randint(
        0,
        np.iinfo(np.int32).max,
        size=(int(timesteps), int(repeats)),
        dtype=np.int64,
    )


@contextmanager
def _forked_seed(reference: torch.Tensor, seed: int) -> Iterator[None]:
    devices: list[int] = []
    if reference.device.type == "cuda":
        devices = [
            reference.device.index
            if reference.device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield
