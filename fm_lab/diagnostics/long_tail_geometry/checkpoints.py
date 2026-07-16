"""Checkpoint restoration and exact ordinary-FM probe-loss replay."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.manifest import (
    ProbeManifest,
    materialize_probe_batch,
)
from fm_lab.experiments.factory import build_model, build_source
from fm_lab.paths.prediction import normalize_prediction_kind
from fm_lab.training.prediction import model_prediction
from fm_lab.training.trainer import validate_checkpoint_compatibility
from fm_lab.utils.checkpoints import load_checkpoint


@dataclass(frozen=True)
class ProbeLossResult:
    """Exact row-level loss replay summary for one manifest."""

    mean_loss: float
    row_losses: torch.Tensor
    row_losses_sha256: str


def restore_probe_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore raw ordinary-FM weights after validating the checkpoint contract."""

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Probe checkpoint is missing its experiment config.")
    config = copy.deepcopy(config)
    _validate_ordinary_probe_config(config)
    validate_checkpoint_compatibility(checkpoint, active_config=config)
    source = build_source(config)
    model = build_model(config, dim=source.dim)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Probe checkpoint is missing model_state_dict.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _validate_capacity_disabled(model)
    return model, config


@torch.no_grad()
def evaluate_probe_loss(
    *,
    model: nn.Module,
    objective: Any,
    path: Any,
    manifest: ProbeManifest,
    target: Any,
    source: Any,
    device: torch.device,
) -> ProbeLossResult:
    """Replay deterministic per-row ordinary flow-matching losses."""

    _validate_ordinary_objective(objective)
    _validate_capacity_disabled(model)
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    try:
        for rows in manifest.microbatch_row_indices():
            batch = materialize_probe_batch(
                target,
                source,
                manifest,
                rows,
                device=device,
            )
            xt = path.sample_xt(batch.x0, batch.x1, batch.t)
            target_velocity = path.target_velocity(batch.x0, batch.x1, batch.t)
            output = model_prediction(
                model,
                xt,
                batch.t,
                class_labels=batch.labels,
            )
            state = path.prediction_state(xt, batch.t, min_denom=objective.min_denom)
            model_output = normalize_prediction_kind(objective.model_output).value
            loss_space = normalize_prediction_kind(objective.loss_space).value
            prediction = state.prediction(output, model_output).convert(loss_space)
            supervision = {
                "source": batch.x0,
                "target": batch.x1,
                "velocity": target_velocity,
            }
            expected = state.prediction(
                supervision[model_output],
                model_output,
            ).convert(loss_space)
            if objective.loss != "mse":
                raise ValueError("Stage-0 probe replay supports only ordinary MSE.")
            losses.append(
                (prediction - expected).square().flatten(1).mean(1).double().cpu()
            )
    finally:
        model.train(was_training)

    row_losses = torch.cat(losses).contiguous()
    digest = hashlib.sha256(row_losses.numpy().tobytes()).hexdigest()
    return ProbeLossResult(
        mean_loss=float(row_losses.mean()),
        row_losses=row_losses,
        row_losses_sha256=digest,
    )


def _validate_ordinary_probe_config(config: dict[str, Any]) -> None:
    objective = config.get("objective", {}) or {}
    if str(objective.get("name", "flow_matching")).lower() != "flow_matching":
        raise ValueError("Stage-0 probes require the ordinary flow_matching objective.")
    if objective.get("modifiers", []) or float(objective.get("straightness_weight", 0)):
        raise ValueError("Stage-0 probes must not use objective modifiers.")
    capacity = (config.get("model", {}) or {}).get("capacity", {}) or {}
    if bool(capacity.get("enabled", False)):
        raise ValueError("Stage-0 probes must not enable model capacity adapters.")


def _validate_ordinary_objective(objective: Any) -> None:
    if str(getattr(objective, "name", "")).lower() != "flow_matching":
        raise ValueError("Stage-0 probes require a flow-matching objective.")
    if getattr(objective, "modifiers", ()):
        raise ValueError("Stage-0 probes must not use objective modifiers.")
    if float(getattr(objective, "straightness_weight", 0.0)) != 0.0:
        raise ValueError("Stage-0 probes must not use straightness regularization.")
    if float(getattr(objective, "interpolant_acceleration_weight", 0.0)) != 0.0:
        raise ValueError("Stage-0 probes must not regularize interpolant acceleration.")


def _validate_capacity_disabled(model: nn.Module) -> None:
    metadata = getattr(model, "capacity_metadata", None)
    if callable(metadata) and bool(metadata().get("enabled", False)):
        raise ValueError("Stage-0 probes must use a capacity-disabled model.")
