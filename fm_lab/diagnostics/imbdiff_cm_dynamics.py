"""Sparse in-loop training-dynamics diagnostics for faithful ImbDiff-CM."""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from fm_lab.diagnostics.imbdiff_cm_intervention import active_lora_modules
from fm_lab.diagnostics.imbdiff_cm_probe import radial_spectral_fractions
from fm_lab.evaluation.groups import frequency_ranked_groups
from fm_lab.integrations.official_imbdiff_cm import OfficialImbDiffCMTerms
from fm_lab.utils.logging import write_json

_COMPONENTS = ("base", "consistency", "diversity", "cm", "total")
_CAPACITY_GROUPS = ("general", "expert_a", "expert_b", "expert")
_SPECTRAL_BANDS = ("low", "mid_low", "mid_high", "high")


@dataclass
class _LayerSnapshot:
    weight: torch.Tensor
    factor_a: torch.Tensor
    factor_b: torch.Tensor

    @property
    def effective_expert(self) -> torch.Tensor:
        return (self.factor_b @ self.factor_a).reshape_as(self.weight)


@dataclass
class _FunctionalSnapshot:
    full: torch.Tensor
    general: torch.Tensor


@dataclass
class ImbDiffCMDynamicsStep:
    """State retained only across one observed optimizer step."""

    step: int
    labels: torch.Tensor
    timesteps: torch.Tensor
    unconditional_batch: bool
    terms: OfficialImbDiffCMTerms
    layers_before: dict[str, _LayerSnapshot]
    ema_layers_before: dict[str, _LayerSnapshot] | None
    functional_before: _FunctionalSnapshot | None
    selected_gradients: dict[str, dict[str, torch.Tensor]]
    component_rows: list[dict[str, Any]]
    layer_component_rows: list[dict[str, Any]]
    alignment_rows: list[dict[str, Any]]
    conditioned_rows: list[dict[str, Any]]
    conditioned_layer_rows: list[dict[str, Any]]
    conditioned_alignment_rows: list[dict[str, Any]]


class ImbDiffCMDynamicsObserver:
    """Observe exact CM graphs and the Adam actions they induce at sparse steps."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        run_dir: Path,
        model: nn.Module,
        objective: Any,
        ema_model: nn.Module | None,
    ) -> None:
        self.config = dict(config)
        self.output_dir = run_dir / str(config.get("output_subdir", "cm_dynamics"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps = _validated_steps(config.get("steps", ()))
        self.component_gradients = bool(config.get("component_gradients", True))
        self.conditioned_gradients = bool(config.get("conditioned_gradients", True))
        self.functional_updates = bool(config.get("functional_updates", True))
        self.class_counts = tuple(int(value) for value in objective.class_counts)
        self.timesteps = int(objective.timesteps)
        self.consistency_weight = float(objective.consistency_weight)
        self.diversity_weight = float(objective.diversity_weight)
        self.frequency_groups = frequency_ranked_groups(self.class_counts)
        self.all_lora_modules = active_lora_modules(model)
        self.selected_layer_names = _select_layer_names(
            self.all_lora_modules,
            requested=config.get("layers"),
            max_layers=int(config.get("max_layers", 5)),
        )
        self.named_parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self.parameter_names = tuple(name for name, _ in self.named_parameters)
        self.parameters = tuple(parameter for _, parameter in self.named_parameters)
        self.selected_parameter_names = frozenset(
            f"{layer_name}.{suffix}"
            for layer_name in self.selected_layer_names
            for suffix in ("weight", "lora_A", "lora_B")
        )
        missing = self.selected_parameter_names - set(self.parameter_names)
        if missing:
            raise ValueError(
                "Selected CM dynamics parameters are missing: " + ", ".join(sorted(missing))
            )
        self.initial_layers = _snapshot_layers(model, self.selected_layer_names)
        self.previous_observed_layers = {
            name: _cpu_snapshot(snapshot) for name, snapshot in self.initial_layers.items()
        }
        self.observed_path_lengths = {
            name: {"general": 0.0, "expert": 0.0} for name in self.selected_layer_names
        }
        self.conditioned_cumulative: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.observed_steps: list[int] = []
        self._write_manifest(model=model, ema_model=ema_model)

    def should_observe(self, step: int) -> bool:
        return int(step) in self.steps

    def before_backward(
        self,
        *,
        step: int,
        model: nn.Module,
        ema_model: nn.Module | None,
        terms: OfficialImbDiffCMTerms,
        labels: torch.Tensor,
    ) -> ImbDiffCMDynamicsStep:
        """Measure gradients on the live graph and snapshot pre-Adam state."""

        if int(step) not in self.steps:
            raise ValueError(f"Step {step} is not in the dynamics schedule.")
        if labels.shape != terms.timesteps.shape:
            raise ValueError("CM dynamics labels and timesteps must align.")
        labels = labels.detach().to(device=terms.timesteps.device, dtype=torch.long)
        components = _component_losses(
            terms,
            consistency_weight=self.consistency_weight,
            diversity_weight=self.diversity_weight,
        )
        component_rows: list[dict[str, Any]] = []
        selected_gradients: dict[str, dict[str, torch.Tensor]] = {}
        if self.component_gradients:
            for component_name in _COMPONENTS:
                gradients = torch.autograd.grad(
                    components[component_name],
                    self.parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                component_rows.extend(
                    _gradient_group_rows(
                        step=step,
                        axis="loss_component",
                        stratum=component_name,
                        named_parameters=self.named_parameters,
                        gradients=gradients,
                        loss_value=float(components[component_name].detach().cpu()),
                    )
                )
                selected_gradients[component_name] = _selected_gradient_map(
                    self.named_parameters,
                    gradients,
                    selected_names=self.selected_parameter_names,
                )
        else:
            selected_gradients["total"] = {}

        alignment_rows = _selected_gradient_alignment_rows(
            step=step,
            gradients_by_name=selected_gradients,
        )
        layer_component_rows = _layer_component_gradient_rows(
            step=step,
            gradients_by_name=selected_gradients,
            model=model,
            selected_layer_names=self.selected_layer_names,
        )
        conditioned_rows: list[dict[str, Any]] = []
        conditioned_layer_rows: list[dict[str, Any]] = []
        conditioned_selected: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        if self.conditioned_gradients:
            for axis, strata in self._condition_masks(labels, terms.timesteps).items():
                for stratum, mask in strata.items():
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    contribution = terms.total_per_sample[mask].sum() / len(labels)
                    gradients = torch.autograd.grad(
                        contribution,
                        self.parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    conditioned_rows.extend(
                        _gradient_group_rows(
                            step=step,
                            axis=axis,
                            stratum=stratum,
                            named_parameters=self.named_parameters,
                            gradients=gradients,
                            loss_value=float(contribution.detach().cpu()),
                            batch_count=count,
                            batch_fraction=count / len(labels),
                        )
                    )
                    selected = _selected_gradient_map(
                        self.named_parameters,
                        gradients,
                        selected_names=self.selected_parameter_names,
                    )
                    conditioned_selected[(axis, stratum)] = selected
                    conditioned_layer_rows.extend(
                        self._conditioned_layer_rows(
                            step=step,
                            axis=axis,
                            stratum=stratum,
                            selected_gradients=selected,
                            model=model,
                        )
                    )
        conditioned_alignment_rows = _conditioned_alignment_rows(
            step=step,
            gradients_by_stratum=conditioned_selected,
        )
        layers_before = _snapshot_layers(model, self.selected_layer_names)
        ema_layers_before = (
            _snapshot_layers(ema_model, self.selected_layer_names)
            if ema_model is not None
            else None
        )
        functional_before = (
            _functional_predictions(model, terms) if self.functional_updates else None
        )
        return ImbDiffCMDynamicsStep(
            step=int(step),
            labels=labels.detach(),
            timesteps=terms.timesteps.detach(),
            unconditional_batch=bool(terms.unconditional_batch),
            terms=terms,
            layers_before=layers_before,
            ema_layers_before=ema_layers_before,
            functional_before=functional_before,
            selected_gradients=selected_gradients,
            component_rows=component_rows,
            layer_component_rows=layer_component_rows,
            alignment_rows=alignment_rows,
            conditioned_rows=conditioned_rows,
            conditioned_layer_rows=conditioned_layer_rows,
            conditioned_alignment_rows=conditioned_alignment_rows,
        )

    def after_optimizer_step(
        self,
        state: ImbDiffCMDynamicsStep,
        *,
        model: nn.Module,
        ema_model: nn.Module | None,
    ) -> None:
        """Measure realized parameter and functional updates and flush artifacts."""

        layers_after = _snapshot_layers(model, self.selected_layer_names)
        ema_layers_after = (
            _snapshot_layers(ema_model, self.selected_layer_names)
            if ema_model is not None
            else None
        )
        layer_rows = self._layer_update_rows(
            state,
            layers_after=layers_after,
            ema_layers_after=ema_layers_after,
        )
        functional_rows: list[dict[str, Any]] = []
        if state.functional_before is not None:
            functional_after = _functional_predictions(model, state.terms)
            functional_rows = self._functional_update_rows(
                state,
                before=state.functional_before,
                after=functional_after,
            )
        _append_rows(self.output_dir / "gradient_components.csv", state.component_rows)
        _append_rows(
            self.output_dir / "layer_gradient_components.csv",
            state.layer_component_rows,
        )
        _append_rows(self.output_dir / "gradient_alignments.csv", state.alignment_rows)
        _append_rows(self.output_dir / "conditioned_gradients.csv", state.conditioned_rows)
        _append_rows(
            self.output_dir / "conditioned_layer_gradients.csv",
            state.conditioned_layer_rows,
        )
        _append_rows(
            self.output_dir / "conditioned_alignments.csv",
            state.conditioned_alignment_rows,
        )
        _append_rows(self.output_dir / "layer_updates.csv", layer_rows)
        _append_rows(self.output_dir / "functional_updates.csv", functional_rows)
        self.observed_steps.append(state.step)
        self._write_summary()

    def finalize(self, *, final_step: int) -> dict[str, Any]:
        """Finalize and return compact metadata for the parent run."""

        self._write_summary(final_step=final_step)
        return {
            "enabled": True,
            "output_dir": str(self.output_dir),
            "requested_steps": sorted(self.steps),
            "observed_steps": list(self.observed_steps),
            "selected_layers": list(self.selected_layer_names),
            "component_gradients": self.component_gradients,
            "conditioned_gradients": self.conditioned_gradients,
            "functional_updates": self.functional_updates,
        }

    def _condition_masks(
        self,
        labels: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        frequency = {
            group: _membership_mask(labels, class_ids)
            for group, class_ids in self.frequency_groups.items()
        }
        one_third = self.timesteps / 3.0
        two_thirds = 2.0 * self.timesteps / 3.0
        time = {
            "late_low_noise": timesteps < one_third,
            "middle": (timesteps >= one_third) & (timesteps < two_thirds),
            "early_high_noise": timesteps >= two_thirds,
        }
        return {"frequency": frequency, "diffusion_time": time}

    def _conditioned_layer_rows(
        self,
        *,
        step: int,
        axis: str,
        stratum: str,
        selected_gradients: Mapping[str, torch.Tensor],
        model: nn.Module,
    ) -> list[dict[str, Any]]:
        modules = dict(active_lora_modules(model))
        rows: list[dict[str, Any]] = []
        for layer_name in self.selected_layer_names:
            module = modules[layer_name]
            grad_a = selected_gradients.get(f"{layer_name}.lora_A")
            grad_b = selected_gradients.get(f"{layer_name}.lora_B")
            if grad_a is None or grad_b is None:
                continue
            factor_a = module.lora_A.detach().float().cpu()
            factor_b = module.lora_B.detach().float().cpu()
            effective = _raw_effective_descent(factor_a, factor_b, grad_a, grad_b)
            key = (axis, stratum, layer_name)
            accumulator = self.conditioned_cumulative.setdefault(
                key,
                {
                    "sum": torch.zeros_like(effective),
                    "path_length": 0.0,
                    "observations": 0,
                },
            )
            accumulator["sum"].add_(effective)
            accumulator["path_length"] += _norm(effective)
            accumulator["observations"] += 1
            rows.append(
                {
                    "step": int(step),
                    "axis": axis,
                    "stratum": stratum,
                    "layer_name": layer_name,
                    "raw_effective_expert_gradient_norm": _norm(effective),
                    "cumulative_effective_gradient_norm": _norm(accumulator["sum"]),
                    "cumulative_gradient_path_length": accumulator["path_length"],
                    "cumulative_observations": accumulator["observations"],
                }
            )
        return rows

    def _layer_update_rows(
        self,
        state: ImbDiffCMDynamicsStep,
        *,
        layers_after: Mapping[str, _LayerSnapshot],
        ema_layers_after: Mapping[str, _LayerSnapshot] | None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        total_gradients = state.selected_gradients.get("total", {})
        for layer_name in self.selected_layer_names:
            before = state.layers_before[layer_name]
            after = layers_after[layer_name]
            delta_a = after.factor_a - before.factor_a
            delta_b = after.factor_b - before.factor_b
            delta_general = after.weight - before.weight
            delta_expert = after.effective_expert - before.effective_expert
            decomposition = (
                before.factor_b @ delta_a + delta_b @ before.factor_a + delta_b @ delta_a
            ).reshape_as(delta_expert)
            reconstruction_error = _norm(delta_expert - decomposition)
            grad_a = total_gradients.get(f"{layer_name}.lora_A")
            grad_b = total_gradients.get(f"{layer_name}.lora_B")
            raw_effective = (
                _raw_effective_descent(
                    before.factor_a.cpu(),
                    before.factor_b.cpu(),
                    grad_a,
                    grad_b,
                ).reshape(delta_expert.shape)
                if grad_a is not None and grad_b is not None
                else None
            )
            previous = self.previous_observed_layers[layer_name]
            observed_general_chord = _norm(after.weight.cpu() - previous.weight)
            observed_expert_chord = _norm(after.effective_expert.cpu() - previous.effective_expert)
            self.observed_path_lengths[layer_name]["general"] += observed_general_chord
            self.observed_path_lengths[layer_name]["expert"] += observed_expert_chord
            initial = self.initial_layers[layer_name]
            row: dict[str, Any] = {
                "step": state.step,
                "layer_name": layer_name,
                "factor_a_norm": _norm(before.factor_a),
                "factor_b_norm": _norm(before.factor_b),
                "factor_a_gradient_norm": _norm(grad_a),
                "factor_b_gradient_norm": _norm(grad_b),
                "factor_a_scale_stabilized_sensitivity": (_norm(before.factor_a) * _norm(grad_a)),
                "factor_b_scale_stabilized_sensitivity": (_norm(before.factor_b) * _norm(grad_b)),
                "factor_a_update_norm": _norm(delta_a),
                "factor_b_update_norm": _norm(delta_b),
                "general_weight_update_norm": _norm(delta_general),
                "effective_expert_update_norm": _norm(delta_expert),
                "expert_to_general_update_norm_ratio": _safe_ratio(
                    _norm(delta_expert),
                    _norm(delta_general),
                ),
                "general_expert_update_cosine": _cosine(delta_general, delta_expert),
                "raw_gradient_to_adam_expert_update_cosine": _cosine(
                    raw_effective,
                    delta_expert,
                ),
                "effective_update_reconstruction_error": reconstruction_error,
                "effective_update_reconstruction_relative_error": _safe_ratio(
                    reconstruction_error,
                    _norm(delta_expert),
                ),
                "observed_general_chord_path_length": self.observed_path_lengths[layer_name][
                    "general"
                ],
                "observed_expert_chord_path_length": self.observed_path_lengths[layer_name][
                    "expert"
                ],
                "general_net_displacement_from_initial": _norm(after.weight - initial.weight),
                "expert_net_displacement_from_initial": _norm(
                    after.effective_expert - initial.effective_expert
                ),
                **_singular_summary(after.effective_expert),
            }
            if state.ema_layers_before is not None and ema_layers_after is not None:
                ema_before = state.ema_layers_before[layer_name]
                ema_after = ema_layers_after[layer_name]
                row.update(
                    {
                        "ema_general_weight_update_norm": _norm(
                            ema_after.weight - ema_before.weight
                        ),
                        "ema_effective_expert_update_norm": _norm(
                            ema_after.effective_expert - ema_before.effective_expert
                        ),
                    }
                )
            rows.append(row)
            self.previous_observed_layers[layer_name] = _cpu_snapshot(after)
        return rows

    def _functional_update_rows(
        self,
        state: ImbDiffCMDynamicsStep,
        *,
        before: _FunctionalSnapshot,
        after: _FunctionalSnapshot,
    ) -> list[dict[str, Any]]:
        full_update = after.full - before.full
        general_update = after.general - before.general
        expert_effect_update = full_update - general_update
        pre_expert = before.full - before.general
        post_expert = after.full - after.general
        updates = {
            "full": full_update,
            "general": general_update,
            "expert_effect": expert_effect_update,
        }
        spectra = {name: radial_spectral_fractions(delta) for name, delta in updates.items()}
        scopes = {
            "all": torch.ones(
                len(state.labels),
                dtype=torch.bool,
                device=state.labels.device,
            ),
            **self._condition_masks(state.labels, state.timesteps)["frequency"],
            **self._condition_masks(state.labels, state.timesteps)["diffusion_time"],
        }
        rows: list[dict[str, Any]] = []
        for scope, mask in scopes.items():
            count = int(mask.sum())
            if count == 0:
                continue
            row: dict[str, Any] = {
                "step": state.step,
                "scope": scope,
                "num_samples": count,
                "unconditional_batch": state.unconditional_batch,
                "full_update_rms": _rms(full_update[mask]),
                "general_update_rms": _rms(general_update[mask]),
                "expert_effect_update_rms": _rms(expert_effect_update[mask]),
                "expert_to_full_update_rms_ratio": _safe_ratio(
                    _rms(expert_effect_update[mask]),
                    _rms(full_update[mask]),
                ),
                "general_to_full_update_cosine": _cosine(
                    general_update[mask],
                    full_update[mask],
                ),
                "expert_effect_to_full_update_cosine": _cosine(
                    expert_effect_update[mask],
                    full_update[mask],
                ),
                "pre_expert_effect_rms": _rms(pre_expert[mask]),
                "post_expert_effect_rms": _rms(post_expert[mask]),
            }
            for update_name, fractions in spectra.items():
                for band in _SPECTRAL_BANDS:
                    row[f"{update_name}_spectral_{band}"] = float(
                        fractions[band][mask].mean().cpu()
                    )
            rows.append(row)
        return rows

    def _write_manifest(
        self,
        *,
        model: nn.Module,
        ema_model: nn.Module | None,
    ) -> None:
        write_json(
            {
                "schema_version": 1,
                "observer": "imbdiff_cm_training_dynamics",
                "steps": sorted(self.steps),
                "selected_layers": list(self.selected_layer_names),
                "num_active_lora_layers": len(self.all_lora_modules),
                "num_trainable_parameters": sum(
                    parameter.numel() for _, parameter in self.named_parameters
                ),
                "component_gradients": self.component_gradients,
                "conditioned_gradients": self.conditioned_gradients,
                "functional_updates": self.functional_updates,
                "ema_enabled": ema_model is not None,
                "model_training_at_initialization": bool(model.training),
                "gradient_contract": (
                    "Component and stratum gradients use the live optimizer-step "
                    "graph, batch, noise, endpoint transfer, and dropout draws."
                ),
                "functional_contract": (
                    "Pre/post-Adam predictions use the same live noisy batch and "
                    "conditioning with dropout disabled to isolate parameter action."
                ),
                "path_length_boundary": (
                    "Observed chord path length joins sparse observation points and "
                    "is a lower bound on the complete stepwise optimizer path."
                ),
            },
            self.output_dir / "manifest.json",
        )

    def _write_summary(self, *, final_step: int | None = None) -> None:
        write_json(
            {
                "schema_version": 1,
                "requested_steps": sorted(self.steps),
                "observed_steps": list(self.observed_steps),
                "num_observed_steps": len(self.observed_steps),
                "final_training_step": final_step,
                "selected_layers": list(self.selected_layer_names),
                "complete": (
                    final_step is not None
                    and all(
                        step in self.observed_steps for step in self.steps if step <= final_step
                    )
                ),
            },
            self.output_dir / "summary.json",
        )


def build_imbdiff_cm_dynamics_observer(
    *,
    training_config: Mapping[str, Any],
    run_dir: Path,
    model: nn.Module,
    objective: Any,
    ema_model: nn.Module | None,
    compile_active: bool,
) -> ImbDiffCMDynamicsObserver | None:
    """Build the optional observer and reject scientifically ambiguous setups."""

    config = training_config.get("cm_dynamics", {}) or {}
    if not bool(config.get("enabled", False)):
        return None
    if not bool(getattr(objective, "uses_capacity_model", False)):
        raise ValueError("training.cm_dynamics requires an official CM capacity objective.")
    if not hasattr(objective, "capture_next_training_terms"):
        raise ValueError("The active CM objective does not expose graph-connected terms.")
    if compile_active:
        raise ValueError("training.cm_dynamics requires training.compile.enabled=false.")
    observer = ImbDiffCMDynamicsObserver(
        config=config,
        run_dir=run_dir,
        model=model,
        objective=objective,
        ema_model=ema_model,
    )
    total_steps = int(training_config.get("steps", 0))
    if max(observer.steps) > total_steps:
        raise ValueError("training.cm_dynamics.steps cannot exceed training.steps.")
    return observer


def _validated_steps(configured: Any) -> frozenset[int]:
    if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
        raise ValueError("training.cm_dynamics.steps must be a list of positive integers.")
    values = tuple(configured)
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values
    ):
        raise ValueError("training.cm_dynamics.steps must contain positive integers.")
    if len(set(values)) != len(values):
        raise ValueError("training.cm_dynamics.steps must not contain duplicates.")
    return frozenset(int(value) for value in values)


def _select_layer_names(
    modules: Sequence[tuple[str, nn.Module]],
    *,
    requested: Any,
    max_layers: int,
) -> tuple[str, ...]:
    names = tuple(name for name, _ in modules)
    if requested is not None:
        if not isinstance(requested, Sequence) or isinstance(requested, (str, bytes)):
            raise ValueError("training.cm_dynamics.layers must be a list.")
        selected = tuple(str(value) for value in requested)
        unknown = set(selected) - set(names)
        if unknown:
            raise ValueError("Unknown CM dynamics layers: " + ", ".join(sorted(unknown)))
        if not selected:
            raise ValueError("training.cm_dynamics.layers must not be empty.")
        return selected
    if max_layers < 1:
        raise ValueError("training.cm_dynamics.max_layers must be positive.")
    indices = np.linspace(0, len(names) - 1, min(max_layers, len(names)), dtype=int)
    return tuple(names[index] for index in dict.fromkeys(indices.tolist()))


def _component_losses(
    terms: OfficialImbDiffCMTerms,
    *,
    consistency_weight: float,
    diversity_weight: float,
) -> dict[str, torch.Tensor]:
    base = terms.base_per_sample.mean()
    consistency = float(consistency_weight) * terms.consistency_per_sample.mean()
    diversity = float(diversity_weight) * terms.diversity_per_sample.mean()
    return {
        "base": base,
        "consistency": consistency,
        "diversity": diversity,
        "cm": consistency + diversity,
        "total": terms.loss,
    }


def _gradient_group_rows(
    *,
    step: int,
    axis: str,
    stratum: str,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    gradients: Sequence[torch.Tensor | None],
    loss_value: float,
    batch_count: int | None = None,
    batch_fraction: float | None = None,
) -> list[dict[str, Any]]:
    fine_groups = ("general", "expert_a", "expert_b")
    norm_squares: dict[str, torch.Tensor | None] = dict.fromkeys(fine_groups)
    parameter_counts = dict.fromkeys(fine_groups, 0)
    active_counts = dict.fromkeys(fine_groups, 0)
    for (name, parameter), gradient in zip(
        named_parameters,
        gradients,
        strict=True,
    ):
        group = (
            "expert_a"
            if name.endswith(".lora_A")
            else "expert_b"
            if name.endswith(".lora_B")
            else "general"
        )
        parameter_counts[group] += parameter.numel()
        if gradient is None:
            continue
        detached = gradient.detach().float()
        squared = detached.square().sum()
        norm_squares[group] = (
            squared if norm_squares[group] is None else norm_squares[group] + squared
        )
        active_counts[group] += detached.numel()
    square_values = {
        group: float(value.cpu()) if value is not None else 0.0
        for group, value in norm_squares.items()
    }
    square_values["expert"] = square_values["expert_a"] + square_values["expert_b"]
    parameter_counts["expert"] = parameter_counts["expert_a"] + parameter_counts["expert_b"]
    active_counts["expert"] = active_counts["expert_a"] + active_counts["expert_b"]
    rows: list[dict[str, Any]] = []
    for group_name in _CAPACITY_GROUPS:
        norm = math.sqrt(square_values[group_name])
        rows.append(
            {
                "step": int(step),
                "axis": axis,
                "stratum": stratum,
                "capacity_group": group_name,
                "loss_contribution": float(loss_value),
                "conditional_mean_loss": (
                    float(loss_value) / batch_fraction
                    if batch_fraction is not None and batch_fraction > 0.0
                    else None
                ),
                "batch_count": batch_count,
                "batch_fraction": batch_fraction,
                "gradient_norm": norm,
                "gradient_rms": (
                    norm / math.sqrt(active_counts[group_name])
                    if active_counts[group_name]
                    else 0.0
                ),
                "conditional_mean_gradient_norm": (
                    norm / batch_fraction
                    if batch_fraction is not None and batch_fraction > 0.0
                    else None
                ),
                "conditional_mean_gradient_rms": (
                    norm / math.sqrt(active_counts[group_name]) / batch_fraction
                    if (
                        batch_fraction is not None
                        and batch_fraction > 0.0
                        and active_counts[group_name]
                    )
                    else None
                ),
                "group_num_parameters": parameter_counts[group_name],
                "active_gradient_numel": active_counts[group_name],
            }
        )
    norms = {str(row["capacity_group"]): float(row["gradient_norm"]) for row in rows}
    denominator = norms["general"] ** 2 + norms["expert"] ** 2
    expert_fraction = norms["expert"] ** 2 / denominator if denominator > 0.0 else None
    for row in rows:
        row["expert_gradient_energy_fraction"] = expert_fraction
    return rows


def _selected_gradient_map(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    gradients: Sequence[torch.Tensor | None],
    *,
    selected_names: frozenset[str],
) -> dict[str, torch.Tensor]:
    return {
        name: gradient.detach().float().cpu()
        for (name, _), gradient in zip(named_parameters, gradients, strict=True)
        if name in selected_names and gradient is not None
    }


def _selected_gradient_alignment_rows(
    *,
    step: int,
    gradients_by_name: Mapping[str, Mapping[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    rows = []
    for left, right in itertools.combinations(gradients_by_name, 2):
        for group in _CAPACITY_GROUPS:
            cosine = _gradient_map_cosine(
                gradients_by_name[left],
                gradients_by_name[right],
                group=group,
            )
            rows.append(
                {
                    "step": int(step),
                    "left_component": left,
                    "right_component": right,
                    "capacity_group": group,
                    "cosine": cosine,
                    "conflict": None if cosine is None else bool(cosine < 0.0),
                    "parameter_scope": "selected_adapted_layers",
                }
            )
    return rows


def _layer_component_gradient_rows(
    *,
    step: int,
    gradients_by_name: Mapping[str, Mapping[str, torch.Tensor]],
    model: nn.Module,
    selected_layer_names: Sequence[str],
) -> list[dict[str, Any]]:
    modules = dict(active_lora_modules(model))
    rows = []
    for component, gradients in gradients_by_name.items():
        for layer_name in selected_layer_names:
            module = modules[layer_name]
            factor_a = module.lora_A.detach().float().cpu()
            factor_b = module.lora_B.detach().float().cpu()
            grad_a = gradients.get(f"{layer_name}.lora_A")
            grad_b = gradients.get(f"{layer_name}.lora_B")
            if grad_a is None or grad_b is None:
                continue
            effective = _raw_effective_descent(
                factor_a,
                factor_b,
                grad_a,
                grad_b,
            )
            rows.append(
                {
                    "step": int(step),
                    "component": component,
                    "layer_name": layer_name,
                    "factor_a_norm": _norm(factor_a),
                    "factor_b_norm": _norm(factor_b),
                    "factor_a_gradient_norm": _norm(grad_a),
                    "factor_b_gradient_norm": _norm(grad_b),
                    "factor_a_scale_stabilized_sensitivity": (_norm(factor_a) * _norm(grad_a)),
                    "factor_b_scale_stabilized_sensitivity": (_norm(factor_b) * _norm(grad_b)),
                    "raw_effective_expert_gradient_norm": _norm(effective),
                }
            )
    return rows


def _conditioned_alignment_rows(
    *,
    step: int,
    gradients_by_stratum: Mapping[
        tuple[str, str],
        Mapping[str, torch.Tensor],
    ],
) -> list[dict[str, Any]]:
    rows = []
    axes = sorted({axis for axis, _ in gradients_by_stratum})
    for axis in axes:
        names = sorted(stratum for row_axis, stratum in gradients_by_stratum if row_axis == axis)
        for left, right in itertools.combinations(names, 2):
            for group in _CAPACITY_GROUPS:
                cosine = _gradient_map_cosine(
                    gradients_by_stratum[(axis, left)],
                    gradients_by_stratum[(axis, right)],
                    group=group,
                )
                rows.append(
                    {
                        "step": int(step),
                        "axis": axis,
                        "left_stratum": left,
                        "right_stratum": right,
                        "capacity_group": group,
                        "cosine": cosine,
                        "conflict": None if cosine is None else bool(cosine < 0.0),
                        "parameter_scope": "selected_adapted_layers",
                    }
                )
    return rows


def _gradient_map_cosine(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    group: str,
) -> float | None:
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for name in sorted(set(left) | set(right)):
        if not _parameter_in_group(name, group):
            continue
        left_value = left.get(name)
        right_value = right.get(name)
        if left_value is None or right_value is None:
            continue
        dot += float((left_value * right_value).sum())
        left_square += float(left_value.square().sum())
        right_square += float(right_value.square().sum())
    denominator = math.sqrt(left_square * right_square)
    return dot / denominator if denominator > 0.0 else None


def _parameter_in_group(name: str, group: str) -> bool:
    is_a = name.endswith(".lora_A")
    is_b = name.endswith(".lora_B")
    if group == "expert_a":
        return is_a
    if group == "expert_b":
        return is_b
    if group == "expert":
        return is_a or is_b
    if group == "general":
        return not (is_a or is_b)
    raise ValueError(f"Unknown capacity group: {group}")


def _snapshot_layers(
    model: nn.Module,
    selected_names: Sequence[str],
) -> dict[str, _LayerSnapshot]:
    modules = dict(active_lora_modules(model))
    return {
        name: _LayerSnapshot(
            weight=modules[name].weight.detach().float().clone(),
            factor_a=modules[name].lora_A.detach().float().clone(),
            factor_b=modules[name].lora_B.detach().float().clone(),
        )
        for name in selected_names
    }


def _cpu_snapshot(snapshot: _LayerSnapshot) -> _LayerSnapshot:
    return _LayerSnapshot(
        weight=snapshot.weight.cpu(),
        factor_a=snapshot.factor_a.cpu(),
        factor_b=snapshot.factor_b.cpu(),
    )


@torch.no_grad()
def _functional_predictions(
    model: nn.Module,
    terms: OfficialImbDiffCMTerms,
) -> _FunctionalSnapshot:
    was_training = model.training
    model.eval()
    try:
        full = model(
            terms.noisy,
            terms.timesteps,
            y=terms.conditioned_labels,
            augm=None,
            use_cm=True,
        )
        general = model(
            terms.noisy,
            terms.timesteps,
            y=terms.conditioned_labels,
            augm=None,
            use_cm=False,
        )
    finally:
        model.train(was_training)
    return _FunctionalSnapshot(full=full.detach().float(), general=general.detach().float())


def _membership_mask(labels: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    values = torch.as_tensor(class_ids, device=labels.device, dtype=labels.dtype)
    return (labels[:, None] == values[None, :]).any(dim=1)


def _raw_effective_descent(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    grad_a: torch.Tensor | None,
    grad_b: torch.Tensor | None,
) -> torch.Tensor:
    if grad_a is None or grad_b is None:
        return torch.zeros(
            (factor_b.shape[0], factor_a.shape[1]),
            dtype=factor_a.dtype,
        )
    return -(factor_b @ grad_a + grad_b @ factor_a)


def _singular_summary(weight: torch.Tensor) -> dict[str, float | int]:
    matrix = weight.reshape(weight.shape[0], -1).detach().float().cpu()
    singular = torch.linalg.svdvals(matrix)
    frobenius = float(torch.linalg.vector_norm(singular))
    spectral = float(singular.max()) if len(singular) else 0.0
    return {
        "effective_expert_rank_tolerance": int(torch.count_nonzero(singular > spectral * 1e-6)),
        "effective_expert_spectral_norm": spectral,
        "effective_expert_frobenius_norm": frobenius,
        "effective_expert_nuclear_norm": float(singular.sum()),
        "effective_expert_stable_rank": ((frobenius / spectral) ** 2 if spectral > 0.0 else 0.0),
    }


def _norm(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(torch.linalg.vector_norm(value.detach().float()).cpu())


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _cosine(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> float | None:
    if left is None or right is None:
        return None
    left_flat = left.detach().float().reshape(-1).cpu()
    right_flat = right.detach().float().reshape(-1).cpu()
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left_flat, right_flat) / denominator)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(rows[0])
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
