"""Minimal 2D flow matching trainer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import trange

from fm_lab.couplings.base import Coupling
from fm_lab.data.base import TargetDistribution
from fm_lab.paths.base import FlowPath
from fm_lab.plotting.trajectories import plot_generated_samples, plot_trajectories
from fm_lab.solvers.base import Solver
from fm_lab.sources.base import SourceDistribution
from fm_lab.training.losses import flow_matching_loss, sample_uniform_time
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
    training_config = config.get("training", {})
    batch_size = int(training_config.get("batch_size", 1024))
    steps = int(training_config.get("steps", 10_000))
    lr = float(training_config.get("lr", 1e-4))
    log_every = int(training_config.get("log_every", max(1, min(500, steps))))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[dict[str, float | int]] = []

    progress = trange(1, steps + 1, desc="training", dynamic_ncols=True)
    for step in progress:
        x0 = source.sample(batch_size, device=device)
        x1 = target.sample(batch_size, device=device)
        x0, x1 = coupling.pair(x0, x1)
        t = sample_uniform_time(batch_size, device)

        loss, loss_metrics = flow_matching_loss(model, path, x0, x1, t)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            record = {"step": step, **loss_metrics}
            history.append(record)
            progress.set_postfix(loss=f"{record['loss']:.4f}")

    metrics = {
        "final_loss": history[-1]["loss"],
        "target": target.metadata(),
        "source": source.metadata(),
        "coupling": getattr(coupling, "name", coupling.__class__.__name__),
        "path": getattr(path, "name", path.__class__.__name__),
        "device": str(device),
    }
    write_json(metrics, run_dir / "metrics.json")
    _write_history(history, run_dir / "diagnostics" / "training_history.csv")
    save_checkpoint(
        run_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        step=steps,
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

    model.eval()
    t_grid = torch.linspace(0.0, 1.0, nfe + 1, device=device)

    target_samples = target.sample(n_samples, device=device)
    x0_samples = source.sample(n_samples, device=device)
    generated: dict[str, torch.Tensor] = {}
    artifact_summary: dict[str, Any] = {
        "n_samples": n_samples,
        "n_trajectories": n_trajectories,
        "nfe": nfe,
    }

    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x, t)

    for solver in solvers:
        final = solver.solve(v_fn, x0_samples.clone(), t_grid, return_trajectory=False)
        generated[solver.name] = final.detach().cpu()
        np.save(run_dir / "samples" / f"{solver.name}_nfe{nfe}.npy", generated[solver.name].numpy())

        trajectory_x0 = source.sample(n_trajectories, device=device)
        trajectory = solver.solve(v_fn, trajectory_x0, t_grid, return_trajectory=True)
        trajectory_cpu = trajectory.detach().cpu()
        np.save(run_dir / "trajectories" / f"{solver.name}_nfe{nfe}.npy", trajectory_cpu.numpy())
        plot_trajectories(
            trajectory_cpu,
            run_dir / "plots" / f"trajectories_{solver.name}_nfe{nfe}.png",
            target_samples=target_samples.detach().cpu(),
        )

    np.save(run_dir / "samples" / "target_reference.npy", target_samples.detach().cpu().numpy())
    plot_generated_samples(
        target_samples.detach().cpu(),
        generated,
        run_dir / "plots" / f"generated_samples_nfe{nfe}.png",
    )
    return artifact_summary


def _write_history(history: list[dict[str, float | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        for record in history:
            writer.writerow(record)
