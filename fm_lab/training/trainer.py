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

from fm_lab.couplings.base import Coupling
from fm_lab.data.base import TargetDistribution
from fm_lab.paths.base import FlowPath
from fm_lab.plotting.diagnostics import plot_training_history
from fm_lab.plotting.trajectories import plot_generated_samples, plot_trajectories
from fm_lab.solvers.base import Solver
from fm_lab.solvers.schedules import make_time_grid
from fm_lab.sources.base import SourceDistribution
from fm_lab.training.losses import build_objective, sample_uniform_time
from fm_lab.utils.checkpoints import save_checkpoint
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
    early_stopping = _build_early_stopping(training_config.get("early_stopping", {}))
    objective = build_objective(config.get("objective", {}))
    _validate_training_compatibility(objective, coupling, path, model)

    trainable_path = _is_trainable_path(path)
    theta_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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

    final_step = 0
    best_state: _TrainingState | None = None
    progress = trange(1, steps + 1, desc="training", dynamic_ncols=True)
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
                x0, x1, t = _sample_training_batch(
                    source=source,
                    target=target,
                    coupling=coupling,
                    batch_size=batch_size,
                    device=device,
                )
                _, loss_metrics = objective(
                    model=model,
                    path=path,
                    x0=x0,
                    x1=x1,
                    t=t,
                    compute_diagnostics=True,
                )
                if early_stopping.enabled:
                    candidate_state = _capture_training_state(
                        model=model,
                        path=path,
                        theta_optimizer=theta_optimizer,
                        psi_optimizer=psi_optimizer,
                        step=step,
                    )
        else:
            x0, x1, t = _sample_training_batch(
                source=source,
                target=target,
                coupling=coupling,
                batch_size=batch_size,
                device=device,
            )
            loss, loss_metrics = objective(
                model=model,
                path=path,
                x0=x0,
                x1=x1,
                t=t,
                compute_diagnostics=should_log,
            )
            if should_log and early_stopping.enabled:
                candidate_state = _capture_training_state(
                    model=model,
                    path=path,
                    theta_optimizer=theta_optimizer,
                    psi_optimizer=psi_optimizer,
                    step=step,
                )

            theta_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            theta_optimizer.step()

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

    selected_step = final_step
    selected_record = history[-1]
    restored_best_checkpoint = False
    if best_state is not None:
        _restore_training_state(
            best_state,
            model=model,
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
        optimizer=(
            {"theta": theta_optimizer, "psi": psi_optimizer}
            if psi_optimizer is not None
            else theta_optimizer
        ),
        path_module=path if isinstance(path, nn.Module) else None,
        step=selected_step,
        config=config,
        metrics=metrics,
    )

    sample_artifacts = sample_and_plot(
        config=config,
        run_dir=run_dir,
        target=target,
        source=source,
        model=model,
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
        x0, x1, t = _sample_training_batch(
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
        x0, x1, t = _sample_training_batch(
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
    path: FlowPath,
    theta_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer | None,
    step: int,
) -> _TrainingState:
    path_state = path.state_dict() if isinstance(path, nn.Module) else None
    return _TrainingState(
        step=step,
        model_state=_clone_state(model.state_dict()),
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
    path: FlowPath,
    theta_optimizer: torch.optim.Optimizer,
    psi_optimizer: torch.optim.Optimizer | None,
) -> None:
    model.load_state_dict(state.model_state)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0 = source.sample(batch_size, device=device)
    x1 = target.sample(batch_size, device=device)
    x0, x1 = coupling.pair(x0, x1)
    t = sample_uniform_time(batch_size, device)
    return x0, x1, t


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
) -> dict[str, Any]:
    """Generate final samples and trajectory plots for each configured solver."""

    sampling_config = config.get("sampling", {})
    solver_config = config.get("solvers", {})
    n_samples = int(sampling_config.get("n_samples", 2048))
    n_trajectories = int(sampling_config.get("n_trajectories", 128))
    nfe = int(sampling_config.get("nfe", max(solver_config.get("nfes", [32]))))
    schedule = sampling_config.get("schedule", solver_config.get("schedule", "uniform"))
    plot_max_points = int(sampling_config.get("plot_max_points", n_samples))
    trajectory_target_max_points = int(
        sampling_config.get("trajectory_target_max_points", min(n_samples, 3000))
    )
    sampling_seed = _sampling_seed(config)
    target_metadata = target.metadata()
    image_shape = target_metadata.get("image_shape")
    image_value_range = target_metadata.get("image_value_range", (0.0, 1.0))

    model.eval()
    t_grid = make_time_grid(nfe, schedule=schedule, device=device)

    with _temporary_torch_seed(sampling_seed, device):
        target_samples = target.sample(n_samples, device=device)
        x0_samples = source.sample(n_samples, device=device)
        trajectory_x0 = source.sample(n_trajectories, device=device)
    requires_source_label = _requires_source_label(model)
    generated: dict[str, torch.Tensor] = {}
    artifact_summary: dict[str, Any] = {
        "n_samples": n_samples,
        "n_trajectories": n_trajectories,
        "nfe": nfe,
        "schedule": schedule,
        "plot_max_points": plot_max_points,
        "trajectory_target_max_points": trajectory_target_max_points,
        "seed": sampling_seed,
    }
    if image_shape is not None:
        artifact_summary["image_shape"] = image_shape
        artifact_summary["image_value_range"] = image_value_range
    samples_dir = run_dir / "samples"
    trajectories_dir = run_dir / "trajectories"
    samples_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    def sample_v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _model_velocity(model, x, t, source_label=x0_samples)

    def trajectory_v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _model_velocity(model, x, t, source_label=trajectory_x0)

    for solver in solvers:
        final = solver.solve(sample_v_fn, x0_samples.clone(), t_grid, return_trajectory=False)
        generated[solver.name] = final.detach().cpu()
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
            target_samples=target_samples.detach().cpu(),
            max_target_points=trajectory_target_max_points,
            image_shape=image_shape,
            image_value_range=image_value_range,
        )

    np.save(samples_dir / "source_reference.npy", x0_samples.detach().cpu().numpy())
    np.save(samples_dir / "target_reference.npy", target_samples.detach().cpu().numpy())
    np.save(
        trajectories_dir / f"source_reference_nfe{nfe}.npy",
        trajectory_x0.detach().cpu().numpy(),
    )
    plot_generated_samples(
        target_samples.detach().cpu(),
        generated,
        run_dir / "plots" / f"generated_samples_nfe{nfe}.png",
        max_points=plot_max_points,
        image_shape=image_shape,
        image_value_range=image_value_range,
    )
    return artifact_summary


def _validate_training_compatibility(
    objective: Any,
    coupling: Coupling,
    path: FlowPath,
    model: nn.Module,
) -> None:
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
) -> torch.Tensor:
    if _requires_source_label(model):
        if source_label is None:
            raise ValueError("Source-label-conditioned model requires source labels.")
        return model(x, t, context={"source_label": source_label})
    return model(x, t)


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
