"""Minimal toy flow matching trainer."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import trange

from fm_lab.couplings.base import Coupling, pair_with_condition
from fm_lab.data.base import TargetDistribution
from fm_lab.paths.base import FlowPath
from fm_lab.paths.prediction import PredictionKind, normalize_prediction_kind
from fm_lab.plotting.diagnostics import plot_training_history
from fm_lab.plotting.trajectories import (
    plot_generated_samples,
    plot_trajectories,
    plot_umap_projected_trajectories,
)
from fm_lab.solvers.base import Solver
from fm_lab.solvers.schedules import make_time_grid
from fm_lab.sources.base import SourceDistribution
from fm_lab.training.losses import build_objective
from fm_lab.training.prediction import (
    classifier_free_guided_prediction,
    model_prediction,
    output_kind_for_objective,
    velocity_model_for_objective,
)
from fm_lab.training.runtime import (
    build_optimizer,
    build_warmup_scheduler,
    create_ema_model,
    update_ema_model,
)
from fm_lab.training.sampling_guidance import (
    apply_density_guidance,
    apply_density_prior_rescaling,
    apply_prior_guidance,
    build_sampling_guidance_config,
)
from fm_lab.training.time_sampling import (
    TrainingTimeSampler,
    build_training_time_sampler,
)
from fm_lab.utils.checkpoints import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from fm_lab.utils.logging import write_json


def _explicit_checkpoint_steps(
    configured: object,
    *,
    total_steps: int,
    checkpoint_every: int,
) -> frozenset[int]:
    """Validate an optional explicit intermediate-checkpoint schedule."""

    if configured is None:
        configured = ()
    if not isinstance(configured, (list, tuple)):
        raise ValueError("training.checkpoint_steps must be a list of integers.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in configured):
        raise ValueError("training.checkpoint_steps must contain only integers.")
    steps = tuple(int(value) for value in configured)
    if len(set(steps)) != len(steps):
        raise ValueError("training.checkpoint_steps must not contain duplicates.")
    if any(value < 0 or value > total_steps for value in steps):
        raise ValueError(
            "training.checkpoint_steps values must be between 0 and training.steps."
        )
    if steps and checkpoint_every > 0:
        raise ValueError(
            "training.checkpoint_steps and positive checkpoint_every are mutually exclusive."
        )
    return frozenset(steps)


def train_flow_matching(
    *,
    config: dict[str, Any],
    run_dir: Path,
    target: TargetDistribution,
    source: SourceDistribution,
    coupling: Coupling,
    path: FlowPath,
    model: nn.Module,
    solvers: list[Solver],
    device: torch.device,
) -> dict[str, Any]:
    """Train a toy flow model and save first-stage artifacts."""

    if source.dim != target.dim:
        raise ValueError(f"Source dim {source.dim} does not match target dim {target.dim}.")

    training_config = config.get("training", {})
    time_sampler = build_training_time_sampler(training_config.get("time_sampling"))
    batch_size = int(training_config.get("batch_size", 1024))
    steps = int(training_config.get("steps", 10_000))
    lr = float(training_config.get("lr", 1e-4))
    log_every = int(training_config.get("log_every", max(1, min(500, steps))))
    checkpoint_every = int(training_config.get("checkpoint_every", 0))
    checkpoint_steps = _explicit_checkpoint_steps(
        training_config.get("checkpoint_steps", ()),
        total_steps=steps,
        checkpoint_every=checkpoint_every,
    )
    gradient_clip = float(training_config.get("gradient_clip", 0.0))
    warmup_steps = int(training_config.get("warmup_steps", 0))
    ema_decay_value = training_config.get("ema_decay")
    ema_decay = float(ema_decay_value) if ema_decay_value is not None else None
    if checkpoint_every < 0:
        raise ValueError("training.checkpoint_every must be non-negative.")
    if gradient_clip < 0:
        raise ValueError("training.gradient_clip must be non-negative.")
    early_stopping = _build_early_stopping(training_config.get("early_stopping", {}))
    objective = build_objective(
        config.get("objective", {}),
        diffusion_config=config.get("diffusion", {}),
        class_counts=getattr(target, "class_counts", None),
    )
    _validate_training_compatibility(objective, coupling, path, model)
    checkpoint_config = _checkpoint_config(config, path=path, objective=objective)
    prediction_contract = _prediction_contract(path=path, objective=objective)
    class_counts = getattr(target, "class_counts", None)
    training_contract = build_training_contract(
        checkpoint_config,
        path=path,
        objective=objective,
        class_counts=class_counts,
    )
    trainable_path = _is_trainable_path(path)
    resume_from = training_config.get("resume_from")
    resume_checkpoint: dict[str, Any] | None = None
    if resume_from:
        if trainable_path:
            raise ValueError("Exact resume is not yet supported for trainable paths.")
        resume_checkpoint = load_checkpoint(resume_from, map_location=device)
        validate_checkpoint_compatibility(
            resume_checkpoint,
            active_config=checkpoint_config,
            active_training_contract=training_contract,
        )

    model.to(device)
    if isinstance(path, nn.Module):
        path.to(device)
    condition_dropout = _condition_dropout_probability(config, model)

    theta_optimizer = build_optimizer(model, training_config)
    theta_scheduler = build_warmup_scheduler(theta_optimizer, warmup_steps=warmup_steps)
    ema_model = create_ema_model(model) if ema_decay is not None else None
    psi_optimizer: torch.optim.Optimizer | None = None
    learned_acceleration_schedule: _LearnedAccelerationSchedule | None = None
    if trainable_path:
        _validate_trainable_path_objective(objective)
        learned_acceleration_schedule = _build_learned_acceleration_schedule(
            training_config.get("learned_acceleration", {}),
            default_psi_lr=lr,
        )
        psi_optimizer = torch.optim.AdamW(
            _path_parameters(path),
            lr=learned_acceleration_schedule.psi_lr,
        )
    history: list[dict[str, float | int]] = []
    best_state: _TrainingState | None = None
    start_step = 1
    if resume_from:
        assert resume_checkpoint is not None
        checkpoint = resume_checkpoint
        snapshot = _restore_checkpoint_resume_state(
            checkpoint,
            early_stopping=early_stopping,
            expects_scheduler=theta_scheduler is not None,
            expects_ema=ema_model is not None,
            expects_path=isinstance(path, nn.Module),
            expects_path_optimizer=psi_optimizer is not None,
        )
        best_state = snapshot.best_state
        if snapshot.continuation_state is not None:
            _restore_training_state(
                snapshot.continuation_state,
                model=model,
                ema_model=ema_model,
                path=path,
                theta_optimizer=theta_optimizer,
                theta_scheduler=theta_scheduler,
                psi_optimizer=psi_optimizer,
            )
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            theta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if ema_model is not None:
                ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            if theta_scheduler is not None:
                theta_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(snapshot.rng_state)
        history = snapshot.history
        start_step = snapshot.step + 1
        if start_step > steps:
            raise ValueError(
                f"Resume checkpoint step {start_step - 1} already meets training.steps={steps}."
            )

    if 0 in checkpoint_steps and not resume_from:
        save_checkpoint(
            run_dir / "checkpoints" / "step_000000.pt",
            model=model,
            ema_model=ema_model,
            optimizer=theta_optimizer,
            scheduler=theta_scheduler,
            step=0,
            config=checkpoint_config,
            prediction_contract=prediction_contract,
            training_contract=training_contract,
            resume_state=_checkpoint_resume_state(early_stopping, best_state),
            metrics={"latest_loss": float("nan"), "initial_control": True},
            history=[],
            rng_state=capture_rng_state(),
        )

    final_step = 0
    progress = trange(start_step, steps + 1, desc="training", dynamic_ncols=True)
    for step in progress:
        final_step = step
        should_log = step == 1 or step % log_every == 0 or step == steps
        should_record = should_log or early_stopping.enabled
        record: dict[str, float | int] | None = None
        should_stop = False

        if trainable_path:
            assert psi_optimizer is not None
            assert learned_acceleration_schedule is not None
            _train_learned_acceleration_step(
                objective=objective,
                model=model,
                path=path,
                source=source,
                target=target,
                coupling=coupling,
                batch_size=batch_size,
                device=device,
                step=step,
                theta_optimizer=theta_optimizer,
                psi_optimizer=psi_optimizer,
                schedule=learned_acceleration_schedule,
                time_sampler=time_sampler,
            )
            loss_metrics = {}
            if should_record:
                x0, x1, t, class_labels, original_class_labels = _sample_training_batch(
                    source=source,
                    target=target,
                    coupling=coupling,
                    batch_size=batch_size,
                    device=device,
                    class_conditional=bool(getattr(model, "is_class_conditional", False)),
                    condition_dropout=condition_dropout,
                    time_sampler=time_sampler,
                )
                _, loss_metrics = objective(
                    model=model,
                    path=path,
                    x0=x0,
                    x1=x1,
                    t=t,
                    compute_diagnostics=should_log,
                    class_labels=class_labels,
                    original_class_labels=original_class_labels,
                )
                record = {"step": step, **loss_metrics}
                previous_best_step = early_stopping.best_step
                should_stop = early_stopping.update(record)
                improved = (
                    early_stopping.enabled
                    and early_stopping.best_step == step
                    and previous_best_step != step
                )
                if improved:
                    best_state = _capture_training_state(
                        model=model,
                        ema_model=ema_model,
                        path=path,
                        theta_optimizer=theta_optimizer,
                        theta_scheduler=theta_scheduler,
                        psi_optimizer=psi_optimizer,
                        step=step,
                    )
                    best_state.record = dict(record)
        else:
            x0, x1, t, class_labels, original_class_labels = _sample_training_batch(
                source=source,
                target=target,
                coupling=coupling,
                batch_size=batch_size,
                device=device,
                class_conditional=bool(getattr(model, "is_class_conditional", False)),
                condition_dropout=condition_dropout,
                time_sampler=time_sampler,
            )
            loss, loss_metrics = objective(
                model=model,
                path=path,
                x0=x0,
                x1=x1,
                t=t,
                compute_diagnostics=should_log,
                class_labels=class_labels,
                original_class_labels=original_class_labels,
            )
            if should_record:
                record = {"step": step, **loss_metrics}
                previous_best_step = early_stopping.best_step
                should_stop = early_stopping.update(record)
                improved = (
                    early_stopping.enabled
                    and early_stopping.best_step == step
                    and previous_best_step != step
                )
                if improved:
                    best_state = _capture_training_state(
                        model=model,
                        ema_model=ema_model,
                        path=path,
                        theta_optimizer=theta_optimizer,
                        theta_scheduler=theta_scheduler,
                        psi_optimizer=psi_optimizer,
                        step=step,
                    )
                    best_state.record = dict(record)

            theta_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )
                if record is not None:
                    record["gradient_norm"] = float(gradient_norm.detach().cpu())
            theta_optimizer.step()
            if theta_scheduler is not None:
                theta_scheduler.step()
            if ema_model is not None and ema_decay is not None:
                update_ema_model(ema_model, model, decay=ema_decay)

        if record is not None:
            history.append(record)
            if should_log:
                progress.set_postfix(loss=f"{record['loss']:.4f}")
            if should_stop:
                progress.set_postfix(loss=f"{record['loss']:.4f}", stopped="early")
                break

        interval_checkpoint = bool(checkpoint_every and step % checkpoint_every == 0)
        if interval_checkpoint or step in checkpoint_steps:
            save_checkpoint(
                run_dir / "checkpoints" / f"step_{step:06d}.pt",
                model=model,
                ema_model=ema_model,
                optimizer=theta_optimizer,
                scheduler=theta_scheduler,
                step=step,
                config=checkpoint_config,
                prediction_contract=prediction_contract,
                training_contract=training_contract,
                resume_state=_checkpoint_resume_state(early_stopping, best_state),
                metrics={"latest_loss": float(loss_metrics.get("loss", float("nan")))},
                history=history,
                rng_state=capture_rng_state(),
            )

    continuation_state: dict[str, Any] | None = None
    if best_state is not None:
        terminal_state = _capture_training_state(
            model=model,
            ema_model=ema_model,
            path=path,
            theta_optimizer=theta_optimizer,
            theta_scheduler=theta_scheduler,
            psi_optimizer=psi_optimizer,
            step=final_step,
        )
        terminal_state.record = dict(history[-1])
        continuation_state = _checkpoint_continuation_state(
            terminal_state=terminal_state,
            history=history,
            rng_state=capture_rng_state(),
        )

    selected_step = final_step
    selected_record = history[-1]
    restored_best_checkpoint = False
    if best_state is not None:
        _restore_training_state(
            best_state,
            model=model,
            ema_model=ema_model,
            path=path,
            theta_optimizer=theta_optimizer,
            theta_scheduler=theta_scheduler,
            psi_optimizer=psi_optimizer,
        )
        selected_step = best_state.step
        selected_record = best_state.record or selected_record
        restored_best_checkpoint = selected_step != final_step

    metrics = {
        "final_loss": selected_record["loss"],
        "last_loss": history[-1]["loss"],
        "requested_steps": steps,
        "trained_steps": final_step,
        "checkpoint_step": selected_step,
        "checkpoint_loss": selected_record["loss"],
        "restored_best_checkpoint": restored_best_checkpoint,
        "early_stopping": early_stopping.summary(),
        "target": target.metadata(),
        "source": source.metadata(),
        "coupling": getattr(coupling, "name", coupling.__class__.__name__),
        "path": getattr(path, "name", path.__class__.__name__),
        "path_metadata": path.metadata() if hasattr(path, "metadata") else {},
        "objective": objective.metadata(),
        "device": str(device),
    }
    write_json(metrics, run_dir / "metrics.json")
    _write_history(history, run_dir / "diagnostics" / "training_history.csv")
    plot_training_history(history, run_dir / "plots" / "training_loss.png")
    save_checkpoint(
        run_dir / "checkpoint.pt",
        model=model,
        ema_model=ema_model,
        optimizer=(
            {"theta": theta_optimizer, "psi": psi_optimizer}
            if psi_optimizer is not None
            else theta_optimizer
        ),
        path_module=path if isinstance(path, nn.Module) else None,
        step=selected_step,
        config=checkpoint_config,
        prediction_contract=prediction_contract,
        training_contract=training_contract,
        resume_state=_checkpoint_resume_state(early_stopping, best_state),
        continuation_state=continuation_state,
        metrics=metrics,
        history=history,
        scheduler=theta_scheduler,
        rng_state=capture_rng_state(),
    )

    sampling_skip_reason = _velocity_sampling_skip_reason(objective)
    if sampling_skip_reason is not None:
        metrics["sampling"] = {
            "skipped": True,
            "reason": sampling_skip_reason,
        }
        write_json(metrics, run_dir / "metrics.json")
        return metrics

    sample_artifacts = sample_and_plot(
        config=config,
        run_dir=run_dir,
        target=target,
        source=source,
        path=path,
        model=ema_model if ema_model is not None else model,
        solvers=solvers,
        device=device,
    )
    sample_artifacts["checkpoint_weights"] = (
        "ema" if ema_model is not None else "raw"
    )
    metrics["sampling"] = sample_artifacts
    write_json(metrics, run_dir / "metrics.json")
    return metrics


@dataclass(frozen=True)
class _LearnedAccelerationSchedule:
    warmup_steps: int = 5000
    theta_steps: int = 1
    psi_steps: int = 1
    psi_lr: float = 1e-4


@dataclass
class _TrainingState:
    step: int
    model_state: dict[str, Any]
    theta_optimizer_state: dict[str, Any]
    theta_scheduler_state: dict[str, Any] | None = None
    ema_model_state: dict[str, Any] | None = None
    path_state: dict[str, Any] | None = None
    psi_optimizer_state: dict[str, Any] | None = None
    record: dict[str, float | int] | None = None


@dataclass(frozen=True)
class _ExactResumeSnapshot:
    step: int
    history: list[dict[str, float | int]]
    rng_state: dict[str, Any]
    best_state: _TrainingState | None
    continuation_state: _TrainingState | None = None


def _train_learned_acceleration_step(
    *,
    objective: Any,
    model: nn.Module,
    path: FlowPath,
    source: SourceDistribution,
    target: TargetDistribution,
    coupling: Coupling,
    batch_size: int,
    device: torch.device,
    step: int,
    theta_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer,
    schedule: _LearnedAccelerationSchedule,
    time_sampler: TrainingTimeSampler,
) -> None:
    model.train()
    if isinstance(path, nn.Module):
        path.train()

    for _ in range(schedule.theta_steps):
        x0, x1, t, _, _ = _sample_training_batch(
            source=source,
            target=target,
            coupling=coupling,
            batch_size=batch_size,
            device=device,
            time_sampler=time_sampler,
        )
        loss, _ = objective.theta_update_loss(model=model, path=path, x0=x0, x1=x1, t=t)
        theta_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        theta_optimizer.step()

    if step <= schedule.warmup_steps:
        return

    for _ in range(schedule.psi_steps):
        x0, x1, t, _, _ = _sample_training_batch(
            source=source,
            target=target,
            coupling=coupling,
            batch_size=batch_size,
            device=device,
            time_sampler=time_sampler,
        )
        psi_optimizer.zero_grad(set_to_none=True)
        with _frozen_parameters(model):
            loss, _ = objective.psi_update_loss(model=model, path=path, x0=x0, x1=x1, t=t)
        loss.backward()
        psi_optimizer.step()


def _capture_training_state(
    *,
    model: nn.Module,
    ema_model: nn.Module | None,
    path: FlowPath,
    theta_optimizer: torch.optim.Optimizer,
    theta_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    psi_optimizer: torch.optim.Optimizer | None,
    step: int,
) -> _TrainingState:
    path_state = path.state_dict() if isinstance(path, nn.Module) else None
    return _TrainingState(
        step=step,
        model_state=_clone_state(model.state_dict()),
        ema_model_state=(
            _clone_state(ema_model.state_dict()) if ema_model is not None else None
        ),
        path_state=_clone_state(path_state) if path_state is not None else None,
        theta_optimizer_state=_clone_state(theta_optimizer.state_dict()),
        theta_scheduler_state=(
            _clone_state(theta_scheduler.state_dict())
            if theta_scheduler is not None
            else None
        ),
        psi_optimizer_state=(
            _clone_state(psi_optimizer.state_dict()) if psi_optimizer is not None else None
        ),
    )


def _restore_training_state(
    state: _TrainingState,
    *,
    model: nn.Module,
    ema_model: nn.Module | None,
    path: FlowPath,
    theta_optimizer: torch.optim.Optimizer,
    theta_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    psi_optimizer: torch.optim.Optimizer | None,
) -> None:
    model.load_state_dict(state.model_state)
    if state.ema_model_state is not None:
        if ema_model is None:
            raise ValueError(
                "Best checkpoint state includes an EMA model, but no EMA model exists."
            )
        ema_model.load_state_dict(state.ema_model_state)
    theta_optimizer.load_state_dict(state.theta_optimizer_state)
    if state.theta_scheduler_state is not None:
        if theta_scheduler is None:
            raise ValueError(
                "Best checkpoint state includes a scheduler, but no scheduler exists."
            )
        theta_scheduler.load_state_dict(state.theta_scheduler_state)
    if state.path_state is not None:
        if not isinstance(path, nn.Module):
            raise ValueError(
                "Best checkpoint state includes a path module, but path is not a module."
            )
        path.load_state_dict(state.path_state)
    if state.psi_optimizer_state is not None:
        if psi_optimizer is None:
            raise ValueError(
                "Best checkpoint state includes a path optimizer, but no path optimizer exists."
            )
        psi_optimizer.load_state_dict(state.psi_optimizer_state)


def _clone_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    return copy.deepcopy(value)


def _checkpoint_resume_state(
    early_stopping: _EarlyStopping,
    best_state: _TrainingState | None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "early_stopping": early_stopping.state_dict(),
        "best_training_state": (
            None if best_state is None else _serialize_training_state(best_state)
        ),
    }


def _checkpoint_continuation_state(
    *,
    terminal_state: _TrainingState,
    history: list[dict[str, float | int]],
    rng_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "version": 2,
        "step": terminal_state.step,
        "history": _clone_state(history),
        "rng_state_dict": _clone_state(rng_state),
        "training_state": _serialize_training_state(terminal_state),
    }


def _restore_checkpoint_resume_state(
    checkpoint: Mapping[str, Any],
    *,
    early_stopping: _EarlyStopping,
    expects_scheduler: bool,
    expects_ema: bool,
    expects_path: bool,
    expects_path_optimizer: bool,
) -> _ExactResumeSnapshot:
    raw = checkpoint.get("resume_state")
    if not isinstance(raw, Mapping) or raw.get("version") != 1:
        raise ValueError(
            "Checkpoint is missing exact resume metadata: resume_state version 1."
        )
    if set(raw) != {"version", "early_stopping", "best_training_state"}:
        raise ValueError("Checkpoint resume_state version 1 schema is malformed.")
    early_state = raw.get("early_stopping")
    if not isinstance(early_state, Mapping):
        raise ValueError("Checkpoint resume_state is missing early_stopping state.")
    best_state = _deserialize_optional_training_state(
        raw.get("best_training_state"),
        label="Checkpoint resume_state best_training_state",
        expects_scheduler=expects_scheduler,
        expects_ema=expects_ema,
        expects_path=expects_path,
        expects_path_optimizer=expects_path_optimizer,
    )
    _validate_best_training_state_consistency(early_state, best_state)

    continuation = checkpoint.get("continuation_state")
    if continuation is not None:
        if best_state is None:
            raise ValueError(
                "Checkpoint continuation_state requires a resume_state "
                "best_training_state for the selected-best top-level state."
            )
        selected_step = _validated_resume_step(checkpoint.get("step"), label="checkpoint")
        if selected_step != best_state.step:
            raise ValueError(
                "Checkpoint resume_state best_training_state step does not match "
                "the continuation checkpoint's top-level selected step."
            )
        if not isinstance(continuation, Mapping):
            raise ValueError("Checkpoint continuation_state must be a mapping.")
        expected_keys = {
            "version",
            "step",
            "history",
            "rng_state_dict",
            "training_state",
        }
        if set(continuation) != expected_keys or continuation.get("version") != 2:
            raise ValueError("Checkpoint continuation_state version 2 schema is malformed.")
        terminal_raw = continuation["training_state"]
        if not isinstance(terminal_raw, Mapping):
            raise ValueError("Checkpoint continuation_state training_state is malformed.")
        terminal_state = _deserialize_training_state(
            terminal_raw,
            label="Checkpoint continuation_state training_state",
            expects_scheduler=expects_scheduler,
            expects_ema=expects_ema,
            expects_path=expects_path,
            expects_path_optimizer=expects_path_optimizer,
        )
        history = _validated_resume_history(
            continuation["history"],
            label="Checkpoint continuation_state history",
        )
        rng_state = _validated_resume_rng_state(
            continuation["rng_state_dict"],
            label="Checkpoint continuation_state rng_state_dict",
        )
        step = _validated_resume_step(continuation["step"], label="continuation_state")
        if terminal_state.step != step or int(history[-1]["step"]) != step:
            raise ValueError(
                "Checkpoint continuation_state step, training_state, and history disagree."
            )
        early_stopping.load_state_dict(early_state)
        return _ExactResumeSnapshot(
            step=step,
            history=history,
            rng_state=rng_state,
            best_state=best_state,
            continuation_state=terminal_state,
        )
    _validate_periodic_top_level_state(
        checkpoint,
        expects_scheduler=expects_scheduler,
        expects_ema=expects_ema,
        expects_path=expects_path,
        expects_path_optimizer=expects_path_optimizer,
    )
    step = _validated_resume_step(checkpoint.get("step"), label="checkpoint")
    history = _validated_resume_history(
        checkpoint.get("history"), label="Checkpoint history"
    )
    rng_state = _validated_resume_rng_state(
        checkpoint.get("rng_state_dict"), label="Checkpoint rng_state_dict"
    )
    early_stopping.load_state_dict(early_state)
    return _ExactResumeSnapshot(
        step=step,
        history=history,
        rng_state=rng_state,
        best_state=best_state,
    )


def _validate_best_training_state_consistency(
    early_state: Mapping[str, Any],
    best_state: _TrainingState | None,
) -> None:
    observation = (
        early_state.get("best_step"),
        early_state.get("best_score"),
        early_state.get("best_loss"),
    )
    observed = tuple(value is not None for value in observation)
    if any(observed) and not all(observed):
        raise ValueError(
            "Checkpoint resume_state best_training_state is inconsistent with a "
            "partial early-stopping best observation."
        )
    best_step = observation[0]
    if best_step is None:
        if best_state is not None:
            raise ValueError(
                "Checkpoint resume_state best_training_state must be null when "
                "early stopping has no best observation."
            )
        return
    if isinstance(best_step, bool) or not isinstance(best_step, int) or best_step < 0:
        raise ValueError(
            "Checkpoint resume_state best_training_state has an invalid "
            "early-stopping best_step."
        )
    if best_state is None:
        raise ValueError(
            "Checkpoint resume_state best_training_state is required when "
            "early stopping has a best observation."
        )
    if best_state.step != best_step:
        raise ValueError(
            "Checkpoint resume_state best_training_state step does not match "
            "early-stopping best_step."
        )


def _serialize_training_state(state: _TrainingState) -> dict[str, Any]:
    return {
        "step": state.step,
        "model_state": _clone_state(state.model_state),
        "theta_optimizer_state": _clone_state(state.theta_optimizer_state),
        "theta_scheduler_state": _clone_state(state.theta_scheduler_state),
        "ema_model_state": _clone_state(state.ema_model_state),
        "path_state": _clone_state(state.path_state),
        "psi_optimizer_state": _clone_state(state.psi_optimizer_state),
        "record": _clone_state(state.record),
    }


def _deserialize_optional_training_state(
    raw: object,
    *,
    label: str,
    expects_scheduler: bool,
    expects_ema: bool,
    expects_path: bool,
    expects_path_optimizer: bool,
) -> _TrainingState | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping or null.")
    return _deserialize_training_state(
        raw,
        label=label,
        expects_scheduler=expects_scheduler,
        expects_ema=expects_ema,
        expects_path=expects_path,
        expects_path_optimizer=expects_path_optimizer,
    )


def _deserialize_training_state(
    raw: Mapping[str, Any],
    *,
    label: str,
    expects_scheduler: bool,
    expects_ema: bool,
    expects_path: bool,
    expects_path_optimizer: bool,
) -> _TrainingState:
    required = {
        "step",
        "model_state",
        "theta_optimizer_state",
        "theta_scheduler_state",
        "ema_model_state",
        "path_state",
        "psi_optimizer_state",
        "record",
    }
    if set(raw) != required:
        missing = required - set(raw)
        extra = set(raw) - required
        raise ValueError(
            f"{label} schema is malformed; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    if not isinstance(raw["model_state"], Mapping) or not isinstance(
        raw["theta_optimizer_state"], Mapping
    ):
        raise ValueError(f"{label} model and optimizer states must be mappings.")
    _validate_optional_nested_state(
        raw["theta_scheduler_state"], expected=expects_scheduler, label=f"{label} scheduler"
    )
    _validate_optional_nested_state(
        raw["ema_model_state"], expected=expects_ema, label=f"{label} EMA"
    )
    _validate_optional_nested_state(
        raw["path_state"], expected=expects_path, label=f"{label} path"
    )
    _validate_optional_nested_state(
        raw["psi_optimizer_state"],
        expected=expects_path_optimizer,
        label=f"{label} path optimizer",
    )
    record = raw.get("record")
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record must be a mapping.")
    step = _validated_resume_step(raw["step"], label=label)
    if int(record.get("step", -1)) != step:
        raise ValueError(f"{label} record step does not match its state step.")
    return _TrainingState(
        step=step,
        model_state=_clone_state(dict(raw["model_state"])),
        theta_optimizer_state=_clone_state(dict(raw["theta_optimizer_state"])),
        theta_scheduler_state=_clone_state(raw.get("theta_scheduler_state")),
        ema_model_state=_clone_state(raw.get("ema_model_state")),
        path_state=_clone_state(raw.get("path_state")),
        psi_optimizer_state=_clone_state(raw.get("psi_optimizer_state")),
        record=dict(record),
    )


def _validate_optional_nested_state(value: object, *, expected: bool, label: str) -> None:
    if expected and not isinstance(value, Mapping):
        raise ValueError(f"{label} state must be a mapping for the active component.")
    if not expected and value is not None:
        raise ValueError(f"{label} state must be null when the component is inactive.")


def _validated_resume_step(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} step must be a non-negative integer.")
    return value


def _validated_resume_history(value: object, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list.")
    if any(not isinstance(record, Mapping) for record in value):
        raise ValueError(f"{label} records must be mappings.")
    return [_clone_state(dict(record)) for record in value]


def _validated_resume_rng_state(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not {"python", "numpy", "torch"} <= set(value):
        raise ValueError(f"{label} is missing Python, NumPy, or Torch RNG state.")
    return _clone_state(dict(value))


def _validate_periodic_top_level_state(
    checkpoint: Mapping[str, Any],
    *,
    expects_scheduler: bool,
    expects_ema: bool,
    expects_path: bool,
    expects_path_optimizer: bool,
) -> None:
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError("Checkpoint model_state_dict must be a mapping.")
    optimizer = checkpoint.get("optimizer_state_dict")
    if expects_path_optimizer:
        if not isinstance(optimizer, Mapping) or not {"theta", "psi"} <= set(optimizer):
            raise ValueError("Checkpoint path optimizer state is malformed.")
    elif not isinstance(optimizer, Mapping):
        raise ValueError("Checkpoint optimizer_state_dict must be a mapping.")
    _validate_optional_nested_state(
        checkpoint.get("scheduler_state_dict"),
        expected=expects_scheduler,
        label="Checkpoint scheduler",
    )
    _validate_optional_nested_state(
        checkpoint.get("ema_model_state_dict"),
        expected=expects_ema,
        label="Checkpoint EMA",
    )
    _validate_optional_nested_state(
        checkpoint.get("path_state_dict"),
        expected=expects_path,
        label="Checkpoint path",
    )


def _sample_training_batch(
    *,
    source: SourceDistribution,
    target: TargetDistribution,
    coupling: Coupling,
    batch_size: int,
    device: torch.device,
    class_conditional: bool = False,
    condition_dropout: float = 0.0,
    time_sampler: TrainingTimeSampler | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    x0 = source.sample(batch_size, device=device)
    if class_conditional:
        x1, class_labels = _sample_target_with_optional_labels(target, batch_size, device=device)
        if class_labels is None:
            raise ValueError("Class conditioning requires a target with sample_with_labels().")
    else:
        x1 = target.sample(batch_size, device=device)
        class_labels = None
    x0, x1, class_labels = pair_with_condition(coupling, x0, x1, class_labels)
    original_class_labels = class_labels.clone() if class_labels is not None else None
    if class_labels is not None and condition_dropout > 0:
        drop = torch.rand(batch_size, device=device) < condition_dropout
        class_labels = class_labels.clone()
        class_labels[drop] = -1
    sampler = time_sampler if time_sampler is not None else TrainingTimeSampler()
    t = sampler.sample(batch_size, device)
    return x0, x1, t, class_labels, original_class_labels


def _sample_target_with_optional_labels(
    target: TargetDistribution,
    n: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    sample_with_labels = getattr(target, "sample_with_labels", None)
    if callable(sample_with_labels):
        samples, labels = sample_with_labels(n, device=device)
        return samples, labels
    return target.sample(n, device=device), None


@torch.no_grad()
def sample_and_plot(
    *,
    config: dict[str, Any],
    run_dir: Path,
    target: TargetDistribution,
    source: SourceDistribution,
    model: nn.Module,
    solvers: list[Solver],
    device: torch.device,
    path: FlowPath | None = None,
) -> dict[str, Any]:
    """Generate final samples and trajectory plots for each configured solver."""

    sampling_config = config.get("sampling", {})
    solver_config = config.get("solvers", {})
    n_samples = int(sampling_config.get("n_samples", 2048))
    n_trajectories = int(sampling_config.get("n_trajectories", 128))
    nfe = int(sampling_config.get("nfe", max(solver_config.get("nfes", [32]))))
    schedule = sampling_config.get("schedule", solver_config.get("schedule", "uniform"))
    plot_max_points = int(sampling_config.get("plot_max_points", n_samples))
    sample_batch_size = int(sampling_config.get("sample_batch_size", n_samples))
    if sample_batch_size < 1:
        raise ValueError("sampling.sample_batch_size must be positive.")
    trajectory_target_max_points = int(
        sampling_config.get("trajectory_target_max_points", min(n_samples, 3000))
    )
    sampling_seed = _sampling_seed(config)
    trajectory_umap_config = sampling_config.get("trajectory_umap", {}) or {}
    trajectory_umap_enabled = bool(trajectory_umap_config.get("enabled", False))
    trajectory_umap_max_target_points = int(
        trajectory_umap_config.get("max_target_points", trajectory_target_max_points)
    )
    trajectory_umap_max_trajectories = trajectory_umap_config.get("max_trajectories")
    if trajectory_umap_max_trajectories is not None:
        trajectory_umap_max_trajectories = int(trajectory_umap_max_trajectories)
    trajectory_umap_n_neighbors = int(trajectory_umap_config.get("n_neighbors", 30))
    trajectory_umap_min_dist = float(trajectory_umap_config.get("min_dist", 0.1))
    trajectory_umap_metric = str(trajectory_umap_config.get("metric", "euclidean"))
    trajectory_umap_random_state = int(
        trajectory_umap_config.get("random_state", sampling_seed)
    )
    trajectory_umap_save_coordinates = bool(
        trajectory_umap_config.get("save_coordinates", True)
    )
    guidance = build_sampling_guidance_config(sampling_config)
    target_metadata = target.metadata()
    image_shape = target_metadata.get("image_shape")
    image_value_range = target_metadata.get("image_value_range", (0.0, 1.0))

    model.eval()
    base_model = model
    objective = build_objective(
        config.get("objective", {}),
        diffusion_config=config.get("diffusion", {}),
        class_counts=getattr(target, "class_counts", None),
    )
    if getattr(objective, "prediction_type", None) == "score":
        raise ValueError("ODE sampling does not support score output.")
    if path is None:
        if guidance.density is not None and guidance.density.enabled:
            raise ValueError("Density guidance requires a sampling path.")
        model_output = getattr(objective, "model_output", None)
        if model_output is not None and model_output != "velocity":
            raise ValueError("Non-velocity model output requires a sampling path.")
        if getattr(objective, "prediction_type", None) == "x":
            raise ValueError("sample_and_plot requires path for x-prediction checkpoints.")
    else:
        model = velocity_model_for_objective(base_model, path, objective)
        model = apply_density_guidance(
            base_model=base_model,
            velocity_model=model,
            path=path,
            objective=objective,
            config=guidance.density,
        )
        model.eval()
    t_grid = make_time_grid(nfe, schedule=schedule, device=device)

    with _temporary_torch_seed(sampling_seed, device):
        target_samples, target_labels = _sample_target_with_optional_labels(
            target,
            n_samples,
            device=device,
        )
        x0_samples = source.sample(n_samples, device=device)
        trajectory_x0 = source.sample(n_trajectories, device=device)
        x0_samples = apply_prior_guidance(x0_samples, source=source, config=guidance.prior)
        trajectory_x0 = apply_prior_guidance(
            trajectory_x0,
            source=source,
            config=guidance.prior,
        )
        x0_samples = apply_density_prior_rescaling(
            x0_samples,
            source=source,
            config=guidance.density,
        )
        trajectory_x0 = apply_density_prior_rescaling(
            trajectory_x0,
            source=source,
            config=guidance.density,
        )
    generated_labels = _sampling_class_labels(
        config,
        model=model,
        n_samples=n_samples,
        device=device,
    )
    trajectory_labels = _sampling_class_labels(
        config,
        model=model,
        n_samples=n_trajectories,
        device=device,
    )
    cfg_config = sampling_config.get("classifier_free_guidance", {}) or {}
    cfg_scale = (
        float(cfg_config.get("scale", 1.0))
        if bool(cfg_config.get("enabled", True))
        else 1.0
    )
    if cfg_scale < 0:
        raise ValueError("sampling.classifier_free_guidance.scale must be non-negative.")
    target_samples_cpu = target_samples.detach().cpu()
    target_labels_cpu = target_labels.detach().cpu() if target_labels is not None else None
    del target_samples
    if target_labels is not None:
        del target_labels
    requires_source_label = _requires_source_label(model)
    generated: dict[str, torch.Tensor] = {}
    artifact_summary: dict[str, Any] = {
        "output_kind": output_kind_for_objective(objective).value,
        "path": getattr(path, "name", None),
        "min_denom": float(getattr(objective, "min_denom", 1.0e-3)),
        "solvers": [solver.name for solver in solvers],
        "n_samples": n_samples,
        "n_trajectories": n_trajectories,
        "nfe": nfe,
        "schedule": schedule,
        "plot_max_points": plot_max_points,
        "sample_batch_size": sample_batch_size,
        "trajectory_target_max_points": trajectory_target_max_points,
        "seed": sampling_seed,
    }
    guidance_summary = guidance.summary()
    guidance_summary["classifier_free_guidance"] = {
        "enabled": bool(cfg_config.get("enabled", True)),
        "scale": cfg_scale,
    }
    artifact_summary["guidance"] = guidance_summary
    if image_shape is not None:
        artifact_summary["image_shape"] = image_shape
        artifact_summary["image_value_range"] = image_value_range
    samples_dir = run_dir / "samples"
    trajectories_dir = run_dir / "trajectories"
    samples_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    def trajectory_v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _model_velocity(
            model,
            x,
            t,
            source_label=trajectory_x0,
            class_labels=trajectory_labels,
            guidance_scale=cfg_scale,
        )

    for solver in solvers:
        generated[solver.name] = _solve_final_samples_in_chunks(
            model=model,
            solver=solver,
            x0_samples=x0_samples,
            t_grid=t_grid,
            batch_size=sample_batch_size,
            class_labels=generated_labels,
            guidance_scale=cfg_scale,
        )
        np.save(samples_dir / f"{solver.name}_nfe{nfe}.npy", generated[solver.name].numpy())

        trajectory = solver.solve(
            trajectory_v_fn,
            trajectory_x0.clone(),
            t_grid,
            return_trajectory=True,
        )
        trajectory_cpu = trajectory.detach().cpu()
        np.save(trajectories_dir / f"{solver.name}_nfe{nfe}.npy", trajectory_cpu.numpy())
        if requires_source_label:
            direction_cpu = model.direction(trajectory_x0).detach().cpu()
            artifact_summary.setdefault("line_containment", {})[solver.name] = (
                _line_containment_stats(
                    trajectory=trajectory_cpu,
                    source_label=trajectory_x0.detach().cpu(),
                    direction=direction_cpu,
                )
            )
        plot_trajectories(
            trajectory_cpu,
            run_dir / "plots" / f"trajectories_{solver.name}_nfe{nfe}.png",
            target_samples=target_samples_cpu,
            max_target_points=trajectory_target_max_points,
            image_shape=image_shape,
            image_value_range=image_value_range,
        )
        if trajectory_umap_enabled:
            coordinates_path = None
            if trajectory_umap_save_coordinates:
                coordinates_path = trajectories_dir / f"{solver.name}_nfe{nfe}_umap3d.npz"
            artifact_summary.setdefault("trajectory_umap", {})[solver.name] = (
                plot_umap_projected_trajectories(
                    trajectory_cpu,
                    run_dir / "plots" / f"trajectory_umap3d_{solver.name}_nfe{nfe}.png",
                    target_samples=target_samples_cpu,
                    generated_samples=generated[solver.name],
                    target_labels=target_labels_cpu,
                    max_target_points=trajectory_umap_max_target_points,
                    max_trajectories=trajectory_umap_max_trajectories,
                    n_neighbors=trajectory_umap_n_neighbors,
                    min_dist=trajectory_umap_min_dist,
                    metric=trajectory_umap_metric,
                    random_state=trajectory_umap_random_state,
                    coordinates_path=coordinates_path,
                    interactive_path=(
                        run_dir / "plots" / f"trajectory_umap3d_{solver.name}_nfe{nfe}.html"
                    ),
                    image_shape=image_shape,
                    image_value_range=image_value_range,
                    dataset_name=str(target_metadata.get("name", "")),
                )
            )

    np.save(samples_dir / "source_reference.npy", x0_samples.detach().cpu().numpy())
    np.save(samples_dir / "target_reference.npy", target_samples_cpu.numpy())
    if target_labels_cpu is not None:
        np.save(samples_dir / "target_reference_labels.npy", target_labels_cpu.numpy())
    if generated_labels is not None:
        np.save(samples_dir / "generated_labels.npy", generated_labels.detach().cpu().numpy())
        artifact_summary["classifier_free_guidance_scale"] = cfg_scale
    np.save(
        trajectories_dir / f"source_reference_nfe{nfe}.npy",
        trajectory_x0.detach().cpu().numpy(),
    )
    plot_generated_samples(
        target_samples_cpu,
        generated,
        run_dir / "plots" / f"generated_samples_nfe{nfe}.png",
        max_points=plot_max_points,
        image_shape=image_shape,
        image_value_range=image_value_range,
    )
    return artifact_summary


def _solve_final_samples_in_chunks(
    *,
    model: nn.Module,
    solver: Solver,
    x0_samples: torch.Tensor,
    t_grid: torch.Tensor,
    batch_size: int,
    class_labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    final_chunks: list[torch.Tensor] = []
    for start in range(0, x0_samples.shape[0], batch_size):
        source_label = x0_samples[start : start + batch_size]
        chunk_labels = None if class_labels is None else class_labels[start : start + batch_size]

        def v_fn(
            x: torch.Tensor,
            t: torch.Tensor,
            source_label: torch.Tensor = source_label,
            class_labels: torch.Tensor | None = chunk_labels,
        ) -> torch.Tensor:
            return _model_velocity(
                model,
                x,
                t,
                source_label=source_label,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
            )

        final = solver.solve(
            v_fn,
            source_label.clone(),
            t_grid,
            return_trajectory=False,
        )
        final_chunks.append(final.detach().cpu())
        del final
        _empty_device_cache(t_grid.device)
    return torch.cat(final_chunks, dim=0)


def _empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _validate_training_compatibility(
    objective: Any,
    coupling: Coupling,
    path: FlowPath,
    model: nn.Module,
) -> None:
    if bool(getattr(model, "is_class_conditional", False)):
        if _is_trainable_path(path):
            raise ValueError("Class conditioning with trainable paths is not supported yet.")
        if float(getattr(objective, "straightness_weight", 0.0)) > 0:
            raise ValueError(
                "Class conditioning with learned-flow straightness regularization "
                "is not supported yet."
            )
    objective_name = getattr(objective, "name", "")
    direction_objective_names = {
        "direction_only_straight",
        "direction_speed",
        "lagrangian_direction",
    }
    if objective_name not in direction_objective_names:
        if _requires_source_label(model):
            raise ValueError(
                "Source-label-conditioned models require the direction_only_straight objective."
            )
        return
    if getattr(path, "name", None) != "linear":
        raise ValueError("direction_only_straight requires a linear path in v1.")
    if not _requires_source_label(model):
        raise ValueError("direction_only_straight requires a source-label-conditioned model.")


_DISCRETE_OBJECTIVE_NAMES = frozenset({"discrete_diffusion", "ddpm", "cbdm", "oc", "cm"})
_DIFFUSION_OBJECTIVE_NAMES = frozenset(
    {
        "diffusion",
        "gaussian_diffusion",
        "diffusion_objective",
        "diffusion_epsilon",
        "epsilon_prediction",
        "noise_prediction",
        "diffusion_score",
        "score_matching",
        "diffusion_velocity",
        "diffusion_x",
        "x_prediction",
        "clean_prediction",
    }
)


def validate_resume_checkpoint_before_model(
    *,
    config: dict[str, Any],
    target: TargetDistribution,
    path: FlowPath,
) -> None:
    """Validate exact-resume metadata before the CLI constructs a model."""

    resume_from = (config.get("training", {}) or {}).get("resume_from")
    if not resume_from:
        return
    objective = build_objective(
        config.get("objective", {}),
        diffusion_config=config.get("diffusion", {}),
        class_counts=getattr(target, "class_counts", None),
    )
    checkpoint_config = _checkpoint_config(config, path=path, objective=objective)
    active_training_contract = build_training_contract(
        checkpoint_config,
        path=path,
        objective=objective,
        class_counts=getattr(target, "class_counts", None),
    )
    checkpoint = load_checkpoint(resume_from, map_location="cpu")
    validate_checkpoint_compatibility(
        checkpoint,
        active_config=checkpoint_config,
        active_training_contract=active_training_contract,
    )


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any],
    *,
    active_config: dict[str, Any],
    active_training_contract: dict[str, Any] | None = None,
) -> None:
    """Reject incompatible or ambiguous checkpoints before loading model state."""

    prediction_contract = checkpoint.get("prediction_contract")
    if not isinstance(prediction_contract, dict):
        raise ValueError(
            "Checkpoint is missing continuous prediction metadata: prediction_contract."
        )
    serialized = _validate_contract_values(
        prediction_contract,
        label="Checkpoint prediction_contract",
    )
    serialized_config = checkpoint.get("config")
    if not isinstance(serialized_config, dict):
        raise ValueError("Checkpoint is missing continuous prediction metadata: config.")
    embedded = _continuous_checkpoint_contract(serialized_config, label="Checkpoint config")
    active = _continuous_checkpoint_contract(active_config, label="Active config")
    _raise_contract_mismatch(
        serialized,
        embedded,
        left_label="prediction contract",
        right_label="embedded config",
    )
    _raise_contract_mismatch(
        serialized,
        active,
        left_label="prediction contract",
        right_label="active config",
    )
    if active_training_contract is not None:
        validate_training_contract(
            checkpoint.get("training_contract"),
            active_training_contract,
        )


_TRAINING_CONTRACT_VERSION = 2
_RUNTIME_DATA_FIELDS = frozenset({"root", "download", "workspace"})
_RUNTIME_TRAINING_FIELDS = frozenset(
    {"steps", "resume_from", "log_every", "checkpoint_every", "checkpoint_steps"}
)
_RUNTIME_EXPERIMENT_FIELDS = frozenset({"output_dir"})
_RUNTIME_TOP_LEVEL_SECTIONS = frozenset(
    {"sampling", "solvers", "evaluation", "diagnostics", "plotting"}
)
_SPECIAL_TRAINING_CONTRACT_SECTIONS = frozenset(
    {
        "objective",
        "path",
        "data",
        "source",
        "coupling",
        "model",
        "conditioning",
        "training",
        "experiment",
    }
)


def build_training_contract(
    config: dict[str, Any],
    *,
    path: FlowPath,
    objective: Any,
    class_counts: Sequence[int] | None,
) -> dict[str, Any]:
    """Serialize the semantics that must remain fixed across exact resume."""

    path_config = _training_config_mapping(config, "path")
    objective_config = _training_config_mapping(config, "objective")
    data_config = _training_config_mapping(config, "data")
    source_config = _training_config_mapping(config, "source")
    coupling_config = _training_config_mapping(config, "coupling")
    model_config = _training_config_mapping(config, "model")
    conditioning_config = _training_config_mapping(config, "conditioning")
    training_config = _training_config_mapping(config, "training")
    experiment_config = _training_config_mapping(config, "experiment")
    if not hasattr(objective, "metadata"):
        raise ValueError("Training objective must expose metadata for exact resume.")
    path_metadata = (
        path.metadata()
        if hasattr(path, "metadata")
        else {"name": getattr(path, "name", path.__class__.__name__)}
    )
    semantic_data = {
        key: value for key, value in data_config.items() if key not in _RUNTIME_DATA_FIELDS
    }
    semantic_data["class_counts"] = (
        None if class_counts is None else [int(value) for value in class_counts]
    )
    semantic_training = {
        key: value
        for key, value in training_config.items()
        if key not in _RUNTIME_TRAINING_FIELDS
    }
    semantic_experiment = {
        key: value
        for key, value in experiment_config.items()
        if key not in _RUNTIME_EXPERIMENT_FIELDS
    }
    semantic_config = {
        key: value
        for key, value in config.items()
        if key not in _RUNTIME_TOP_LEVEL_SECTIONS
        and key not in _SPECIAL_TRAINING_CONTRACT_SECTIONS
    }
    semantic_config.update(
        {
            "objective": {
                "config": objective_config,
                "metadata": objective.metadata(),
            },
            "path": {"config": path_config, "metadata": path_metadata},
            "data": semantic_data,
            "source": source_config,
            "coupling": coupling_config,
            "model": model_config,
            "conditioning": conditioning_config,
            "training": semantic_training,
            "experiment": semantic_experiment,
        }
    )
    payload = _canonical_plain_value(
        semantic_config,
        label="training contract payload",
    )
    assert isinstance(payload, dict)
    return {
        "version": _TRAINING_CONTRACT_VERSION,
        "payload": payload,
        "sha256": _training_contract_digest(payload),
    }


def validate_training_contract(saved: object, active: object) -> None:
    """Reject missing, corrupted, or semantically incompatible resume metadata."""

    saved_contract = _validated_training_contract(saved, label="Checkpoint training contract")
    active_contract = _validated_training_contract(active, label="Active training contract")
    if saved_contract["payload"] != active_contract["payload"]:
        changed = [
            field
            for field in sorted(
                set(saved_contract["payload"]) | set(active_contract["payload"])
            )
            if saved_contract["payload"].get(field)
            != active_contract["payload"].get(field)
        ]
        details = ", ".join(changed) or "payload"
        raise ValueError(
            "Checkpoint training contract is incompatible with the active run: "
            f"changed {details}."
        )


def _validated_training_contract(contract: object, *, label: str) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError(f"{label} is missing or malformed.")
    if contract.get("version") != _TRAINING_CONTRACT_VERSION:
        raise ValueError(
            f"{label} version must be {_TRAINING_CONTRACT_VERSION}."
        )
    payload = contract.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload must be a mapping.")
    canonical_payload = _canonical_plain_value(payload, label=f"{label} payload")
    assert isinstance(canonical_payload, dict)
    required = _SPECIAL_TRAINING_CONTRACT_SECTIONS
    missing = required - set(canonical_payload)
    if missing:
        raise ValueError(
            f"{label} payload is missing required sections: {', '.join(sorted(missing))}."
        )
    digest = contract.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{label} sha256 is missing or malformed.")
    expected = _training_contract_digest(canonical_payload)
    if digest != expected:
        raise ValueError(f"{label} sha256 does not match its payload; metadata was tampered.")
    return {
        "version": _TRAINING_CONTRACT_VERSION,
        "payload": canonical_payload,
        "sha256": digest,
    }


def _canonical_plain_value(value: object, *, label: str) -> object:
    if isinstance(value, Mapping):
        canonical: dict[str, object] = {}
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise ValueError(f"{label} mapping keys must be strings.")
        for key in sorted(keys):
            assert isinstance(key, str)
            canonical[key] = _canonical_plain_value(value[key], label=f"{label}.{key}")
        return canonical
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _canonical_plain_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} floats must be finite.")
        return value
    raise ValueError(f"{label} contains unsupported value {type(value).__name__}.")


def _training_config_mapping(
    config: Mapping[str, Any],
    section: str,
) -> dict[str, Any]:
    value = config.get(section, {}) or {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{section} config must be a mapping for the training contract."
        )
    return dict(value)


def _training_contract_digest(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _raise_contract_mismatch(
    left: dict[str, str],
    right: dict[str, str],
    *,
    left_label: str,
    right_label: str,
) -> None:
    mismatches = [
        name
        for name in ("path", "objective", "model_output", "loss_space")
        if left[name] != right[name]
    ]
    if mismatches:
        details = ", ".join(
            f"{name}={left[name]!r} ({left_label}) != {right[name]!r} ({right_label})"
            for name in mismatches
        )
        raise ValueError(f"Checkpoint prediction metadata is incompatible: {details}.")


def _continuous_checkpoint_contract(
    config: dict[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    objective_config = config.get("objective")
    if not isinstance(objective_config, dict):
        raise ValueError(f"{label} is missing continuous prediction metadata: objective.")
    objective_name = objective_config.get("name")
    if objective_name in _DISCRETE_OBJECTIVE_NAMES:
        raise ValueError("discrete checkpoints are incompatible with continuous ODE sampling.")
    path_config = config.get("path")
    if not isinstance(path_config, dict):
        raise ValueError(f"{label} is missing continuous prediction metadata: path.name.")
    missing = [
        key for key in ("name", "model_output", "loss_space") if key not in objective_config
    ]
    if missing:
        fields = ", ".join(f"objective.{key}" for key in missing)
        raise ValueError(f"{label} is missing continuous prediction metadata: {fields}.")
    return _validate_contract_values(
        {
            "path": path_config.get("name"),
            "objective": objective_name,
            "model_output": objective_config.get("model_output"),
            "loss_space": objective_config.get("loss_space"),
        },
        label=label,
    )


def _validate_contract_values(
    contract: dict[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    fields = ("path", "objective", "model_output", "loss_space")
    missing = [field for field in fields if field not in contract]
    if missing:
        names = ", ".join(f"prediction_contract.{field}" for field in missing)
        raise ValueError(f"{label} is missing continuous prediction metadata: {names}.")
    canonical: dict[str, str] = {}
    for field in fields:
        raw = contract[field]
        value = raw if isinstance(raw, str) else ""
        if not value or value != value.strip().lower():
            raise ValueError(
                f"{label} {field} must be a non-empty canonical lowercase value."
            )
        canonical[field] = value
    for field in ("model_output", "loss_space"):
        if canonical[field] not in {*(kind.value for kind in PredictionKind), "score"}:
            raise ValueError(
                f"{label} {field} must be source, target, velocity, or score."
            )
    if (
        "score" in {canonical["model_output"], canonical["loss_space"]}
        and canonical["objective"] not in _DIFFUSION_OBJECTIVE_NAMES
    ):
        raise ValueError(
            f"{label} score output is valid only for a Gaussian DiffusionObjective."
        )
    if canonical["objective"] in _DISCRETE_OBJECTIVE_NAMES:
        raise ValueError("discrete checkpoints are incompatible with continuous ODE sampling.")
    return canonical


def _checkpoint_config(
    config: dict[str, Any],
    *,
    path: FlowPath,
    objective: Any,
) -> dict[str, Any]:
    checkpoint_config = copy.deepcopy(config)
    path_config = dict(checkpoint_config.get("path", {}) or {})
    path_config["name"] = str(getattr(path, "name", path.__class__.__name__)).lower()
    checkpoint_config["path"] = path_config
    objective_config = dict(checkpoint_config.get("objective", {}) or {})
    objective_config["name"] = str(getattr(objective, "name", "")).lower()
    model_output = getattr(objective, "model_output", None)
    if model_output is None:
        model_output = getattr(objective, "prediction_type", None)
    if model_output is None:
        raise ValueError("Continuous objective is missing model output metadata.")
    objective_config["model_output"] = _checkpoint_prediction_value(model_output)
    objective_config["loss_space"] = _checkpoint_prediction_value(
        getattr(objective, "loss_space", model_output)
    )
    if hasattr(objective, "min_denom"):
        objective_config["min_denom"] = float(objective.min_denom)
    checkpoint_config["objective"] = objective_config
    return checkpoint_config


def _prediction_contract(*, path: FlowPath, objective: Any) -> dict[str, str]:
    raw_model_output = getattr(objective, "model_output", None)
    if raw_model_output is None:
        raw_model_output = getattr(objective, "prediction_type", None)
    if raw_model_output is None:
        raise ValueError("Continuous objective is missing model output metadata.")
    model_output = _checkpoint_prediction_value(raw_model_output)
    return {
        "path": str(getattr(path, "name", path.__class__.__name__)).lower(),
        "objective": str(getattr(objective, "name", "")).lower(),
        "model_output": model_output,
        "loss_space": _checkpoint_prediction_value(
            getattr(objective, "loss_space", raw_model_output)
        ),
    }


def _checkpoint_prediction_value(value: str | PredictionKind) -> str:
    if str(value).lower() == "score":
        return "score"
    return normalize_prediction_kind(value).value


def _velocity_sampling_skip_reason(objective: Any) -> str | None:
    prediction_type = getattr(objective, "prediction_type", None)
    if prediction_type is None or prediction_type in {"velocity", "x"}:
        return None
    return (
        "FM-style ODE sampling expects model(x, t) to return a velocity field. "
        f"The diffusion objective was trained for {prediction_type} prediction."
    )


def _validate_trainable_path_objective(objective: Any) -> None:
    required = ("theta_update_loss", "psi_update_loss")
    if not all(hasattr(objective, name) for name in required):
        raise ValueError("Trainable paths require the flow_matching objective in v1.")
    if float(getattr(objective, "straightness_weight", 0.0)) <= 0:
        raise ValueError(
            "Trainable learned-acceleration paths require objective.straightness.weight > 0."
        )


def _is_trainable_path(path: FlowPath) -> bool:
    return isinstance(path, nn.Module) and any(
        parameter.requires_grad for parameter in path.parameters()
    )


def _path_parameters(path: FlowPath) -> list[nn.Parameter]:
    if not isinstance(path, nn.Module):
        return []
    return [parameter for parameter in path.parameters() if parameter.requires_grad]


def _requires_source_label(model: nn.Module) -> bool:
    return bool(getattr(model, "requires_source_label", False))


def _model_velocity(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    source_label: torch.Tensor | None = None,
    class_labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    if _requires_source_label(model):
        if source_label is None:
            raise ValueError("Source-label-conditioned model requires source labels.")
        return model_prediction(model, x, t, source_label=source_label)
    if bool(getattr(model, "is_class_conditional", False)):
        if class_labels is None:
            raise ValueError("Class-conditional model requires sampling class labels.")
        return classifier_free_guided_prediction(
            model,
            x,
            t,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
        )
    return model_prediction(model, x, t)


def _condition_dropout_probability(config: dict[str, Any], model: nn.Module) -> float:
    if not bool(getattr(model, "is_class_conditional", False)):
        return 0.0
    conditioning = config.get("conditioning", {}) or {}
    probability = float(conditioning.get("dropout_probability", 0.1))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("conditioning.dropout_probability must be between 0 and 1.")
    return probability


def _sampling_class_labels(
    config: dict[str, Any],
    *,
    model: nn.Module,
    n_samples: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not bool(getattr(model, "is_class_conditional", False)):
        return None
    conditioning = config.get("conditioning", {}) or {}
    num_classes = int(conditioning["num_classes"])
    sampling = config.get("sampling", {}) or {}
    requested = sampling.get("classes", list(range(num_classes)))
    classes = torch.as_tensor(requested, device=device, dtype=torch.long).flatten()
    if classes.numel() == 0:
        raise ValueError("sampling.classes must contain at least one class.")
    if torch.any(classes < 0) or torch.any(classes >= num_classes):
        raise ValueError(f"sampling.classes must be in [0, {num_classes - 1}].")
    repeats = (n_samples + classes.numel() - 1) // classes.numel()
    return classes.repeat(repeats)[:n_samples]


def _line_containment_stats(
    *,
    trajectory: torch.Tensor,
    source_label: torch.Tensor,
    direction: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    displacement = trajectory - source_label[None, :, :]
    projection_length = (displacement * direction[None, :, :]).sum(dim=2, keepdim=True)
    projected = projection_length * direction[None, :, :]
    residual_norm = (displacement - projected).norm(dim=2)
    displacement_norm = displacement.norm(dim=2)
    relative = residual_norm / (displacement_norm + eps)
    return {
        "off_line_mean": float(residual_norm.mean()),
        "off_line_max": float(residual_norm.max()),
        "off_line_relative_mean": float(relative.mean()),
        "off_line_relative_max": float(relative.max()),
    }


def _sampling_seed(config: dict[str, Any]) -> int | None:
    sampling_config = config.get("sampling", {})
    if "seed" in sampling_config:
        value = sampling_config["seed"]
    else:
        value = config.get("experiment", {}).get("seed")
    if value is None:
        return None
    return int(value)


@contextmanager
def _temporary_torch_seed(seed: int | None, device: torch.device) -> Iterator[None]:
    if seed is None:
        yield
        return

    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


@contextmanager
def _frozen_parameters(module: nn.Module) -> Iterator[None]:
    parameters = list(module.parameters())
    previous = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous, strict=True):
            parameter.requires_grad_(requires_grad)


def _write_history(history: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step"] + sorted({key for row in history for key in row.keys()} - {"step"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(record)


@dataclass
class _EarlyStopping:
    enabled: bool
    monitor: str = "loss"
    patience_steps: int = 0
    min_delta: float = 0.0
    warmup_steps: int = 0
    ema_alpha: float = 1.0
    best_score: float | None = None
    best_loss: float | None = None
    best_step: int | None = None
    current_score: float | None = None
    stopped: bool = False
    stop_step: int | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "patience_steps": self.patience_steps,
            "min_delta": self.min_delta,
            "warmup_steps": self.warmup_steps,
            "ema_alpha": self.ema_alpha,
            "best_score": self.best_score,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "current_score": self.current_score,
            "stopped": self.stopped,
            "stop_step": self.stop_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_config = {
            "enabled": self.enabled,
            "monitor": self.monitor,
            "patience_steps": self.patience_steps,
            "min_delta": self.min_delta,
            "warmup_steps": self.warmup_steps,
            "ema_alpha": self.ema_alpha,
        }
        for field, expected in expected_config.items():
            actual = state.get(field, "loss") if field == "monitor" else state.get(field)
            if actual != expected:
                raise ValueError(
                    "Checkpoint early-stopping state is incompatible: "
                    f"{field}={actual!r} != {expected!r}."
                )
        for field in (
            "best_score",
            "best_loss",
            "best_step",
            "current_score",
            "stopped",
            "stop_step",
        ):
            if field not in state:
                raise ValueError(
                    f"Checkpoint early-stopping state is missing {field}."
                )
        self.best_score = (
            None if state["best_score"] is None else float(state["best_score"])
        )
        self.best_loss = None if state["best_loss"] is None else float(state["best_loss"])
        self.best_step = None if state["best_step"] is None else int(state["best_step"])
        self.current_score = (
            None if state["current_score"] is None else float(state["current_score"])
        )
        self.stopped = bool(state["stopped"])
        self.stop_step = None if state["stop_step"] is None else int(state["stop_step"])

    def update(self, record: dict[str, float | int]) -> bool:
        if not self.enabled:
            return False

        step = int(record["step"])
        loss = float(record["loss"])
        if self.monitor not in record:
            raise ValueError(
                f"Early-stopping monitor {self.monitor!r} is missing from training metrics."
            )
        monitored_value = float(record[self.monitor])
        if self.current_score is None:
            self.current_score = monitored_value
        else:
            self.current_score = (
                self.ema_alpha * monitored_value
                + (1.0 - self.ema_alpha) * self.current_score
            )
        record[f"{self.monitor}_ema"] = self.current_score

        if step < self.warmup_steps:
            return False
        if self.best_score is None or self.best_score - self.current_score > self.min_delta:
            self.best_score = self.current_score
            self.best_loss = loss
            self.best_step = step
            return False
        if self.best_step is not None and step - self.best_step >= self.patience_steps:
            self.stopped = True
            self.stop_step = step
            return True
        return False

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "stopped": self.stopped,
            "monitor": f"{self.monitor}_ema" if self.enabled else self.monitor,
            "patience_steps": self.patience_steps,
            "min_delta": self.min_delta,
            "warmup_steps": self.warmup_steps,
            "ema_alpha": self.ema_alpha,
            "best_step": self.best_step,
            "best_loss": self.best_loss,
            "best_monitor": self.best_score,
            "stop_step": self.stop_step,
        }


def _build_early_stopping(config: dict[str, Any]) -> _EarlyStopping:
    if not config or not bool(config.get("enabled", False)):
        return _EarlyStopping(enabled=False)

    patience_steps = int(config.get("patience_steps", 5000))
    monitor = config.get("monitor", "loss")
    warmup_steps = int(config.get("warmup_steps", 0))
    min_delta = float(config.get("min_delta", 0.0))
    ema_alpha = float(config.get("ema_alpha", 1.0))
    if patience_steps < 1:
        raise ValueError("training.early_stopping.patience_steps must be positive.")
    if warmup_steps < 0:
        raise ValueError("training.early_stopping.warmup_steps must be non-negative.")
    if min_delta < 0:
        raise ValueError("training.early_stopping.min_delta must be non-negative.")
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("training.early_stopping.ema_alpha must be in (0, 1].")
    if not isinstance(monitor, str) or not monitor:
        raise ValueError("training.early_stopping.monitor must be a non-empty string.")

    return _EarlyStopping(
        enabled=True,
        monitor=monitor,
        patience_steps=patience_steps,
        min_delta=min_delta,
        warmup_steps=warmup_steps,
        ema_alpha=ema_alpha,
    )


def _build_learned_acceleration_schedule(
    config: dict[str, Any],
    *,
    default_psi_lr: float,
) -> _LearnedAccelerationSchedule:
    warmup_steps = int(config.get("warmup_steps", 5000))
    theta_steps = int(config.get("theta_steps", 1))
    psi_steps = int(config.get("psi_steps", 1))
    psi_lr = float(config.get("psi_lr", default_psi_lr))
    if warmup_steps < 0:
        raise ValueError("training.learned_acceleration.warmup_steps must be non-negative.")
    if theta_steps < 1:
        raise ValueError("training.learned_acceleration.theta_steps must be positive.")
    if psi_steps < 1:
        raise ValueError("training.learned_acceleration.psi_steps must be positive.")
    if psi_lr <= 0:
        raise ValueError("training.learned_acceleration.psi_lr must be positive.")
    return _LearnedAccelerationSchedule(
        warmup_steps=warmup_steps,
        theta_steps=theta_steps,
        psi_steps=psi_steps,
        psi_lr=psi_lr,
    )
