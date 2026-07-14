"""Minimal toy flow matching trainer."""

from __future__ import annotations

import copy
import csv
from collections.abc import Iterator
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
from fm_lab.training.losses import build_objective, sample_uniform_time
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
from fm_lab.utils.checkpoints import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from fm_lab.utils.logging import write_json


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

    model.to(device)
    if isinstance(path, nn.Module):
        path.to(device)
    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 1024))
    steps = int(training_config.get("steps", 10_000))
    lr = float(training_config.get("lr", 1e-4))
    log_every = int(training_config.get("log_every", max(1, min(500, steps))))
    checkpoint_every = int(training_config.get("checkpoint_every", 0))
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
    condition_dropout = _condition_dropout_probability(config, model)

    trainable_path = _is_trainable_path(path)
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
    start_step = 1
    resume_from = training_config.get("resume_from")
    if resume_from:
        if trainable_path:
            raise ValueError("Exact resume is not yet supported for trainable paths.")
        checkpoint = load_checkpoint(resume_from, map_location=device)
        validate_checkpoint_compatibility(
            checkpoint,
            active_config=checkpoint_config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        theta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema_model is not None:
            ema_model.load_state_dict(
                checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"])
            )
        if theta_scheduler is not None and "scheduler_state_dict" in checkpoint:
            theta_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "rng_state_dict" in checkpoint:
            restore_rng_state(checkpoint["rng_state_dict"])
        history = list(checkpoint.get("history", []))
        start_step = int(checkpoint["step"]) + 1
        if start_step > steps:
            raise ValueError(
                f"Resume checkpoint step {start_step - 1} already meets training.steps={steps}."
            )

    final_step = 0
    best_state: _TrainingState | None = None
    progress = trange(start_step, steps + 1, desc="training", dynamic_ncols=True)
    for step in progress:
        final_step = step
        should_log = step == 1 or step % log_every == 0 or step == steps
        candidate_state: _TrainingState | None = None

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
            )
            loss_metrics = {}
            if should_log:
                x0, x1, t, class_labels, original_class_labels = _sample_training_batch(
                    source=source,
                    target=target,
                    coupling=coupling,
                    batch_size=batch_size,
                    device=device,
                    class_conditional=bool(getattr(model, "is_class_conditional", False)),
                    condition_dropout=condition_dropout,
                )
                _, loss_metrics = objective(
                    model=model,
                    path=path,
                    x0=x0,
                    x1=x1,
                    t=t,
                    compute_diagnostics=True,
                    class_labels=class_labels,
                    original_class_labels=original_class_labels,
                )
                if early_stopping.enabled:
                    candidate_state = _capture_training_state(
                        model=model,
                        ema_model=ema_model,
                        path=path,
                        theta_optimizer=theta_optimizer,
                        psi_optimizer=psi_optimizer,
                        step=step,
                    )
        else:
            x0, x1, t, class_labels, original_class_labels = _sample_training_batch(
                source=source,
                target=target,
                coupling=coupling,
                batch_size=batch_size,
                device=device,
                class_conditional=bool(getattr(model, "is_class_conditional", False)),
                condition_dropout=condition_dropout,
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
            if should_log and early_stopping.enabled:
                candidate_state = _capture_training_state(
                    model=model,
                    ema_model=ema_model,
                    path=path,
                    theta_optimizer=theta_optimizer,
                    psi_optimizer=psi_optimizer,
                    step=step,
                )

            theta_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            theta_optimizer.step()
            if theta_scheduler is not None:
                theta_scheduler.step()
            if ema_model is not None and ema_decay is not None:
                update_ema_model(ema_model, model, decay=ema_decay)

        if should_log:
            record = {"step": step, **loss_metrics}
            history.append(record)
            progress.set_postfix(loss=f"{record['loss']:.4f}")
            previous_best_step = early_stopping.best_step
            should_stop = early_stopping.update(record)
            improved = (
                early_stopping.enabled
                and early_stopping.best_step == step
                and previous_best_step != step
            )
            if improved:
                if candidate_state is None:
                    candidate_state = _capture_training_state(
                        model=model,
                        ema_model=ema_model,
                        path=path,
                        theta_optimizer=theta_optimizer,
                        psi_optimizer=psi_optimizer,
                        step=step,
                    )
                candidate_state.record = dict(record)
                best_state = candidate_state
            if should_stop:
                progress.set_postfix(loss=f"{record['loss']:.4f}", stopped="early")
                break

        if checkpoint_every and step % checkpoint_every == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"step_{step:06d}.pt",
                model=model,
                ema_model=ema_model,
                optimizer=theta_optimizer,
                scheduler=theta_scheduler,
                step=step,
                config=checkpoint_config,
                prediction_contract=prediction_contract,
                metrics={"latest_loss": float(loss_metrics.get("loss", float("nan")))},
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
    ema_model_state: dict[str, Any] | None = None
    path_state: dict[str, Any] | None = None
    psi_optimizer_state: dict[str, Any] | None = None
    record: dict[str, float | int] | None = None


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


def _sample_training_batch(
    *,
    source: SourceDistribution,
    target: TargetDistribution,
    coupling: Coupling,
    batch_size: int,
    device: torch.device,
    class_conditional: bool = False,
    condition_dropout: float = 0.0,
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
    t = sample_uniform_time(batch_size, device)
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


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any],
    *,
    active_config: dict[str, Any],
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

    def update(self, record: dict[str, float | int]) -> bool:
        if not self.enabled:
            return False

        step = int(record["step"])
        loss = float(record["loss"])
        if self.current_score is None:
            self.current_score = loss
        else:
            self.current_score = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.current_score
        record["loss_ema"] = self.current_score

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
            "monitor": "loss_ema" if self.enabled else "loss",
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

    return _EarlyStopping(
        enabled=True,
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
