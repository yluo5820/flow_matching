"""Solver-sensitivity diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.metrics import sliced_wasserstein, squared_mmd
from fm_lab.solvers.base import Solver
from fm_lab.sources.base import SourceDistribution


@torch.no_grad()
def generate_solver_samples(
    *,
    model: torch.nn.Module,
    source: SourceDistribution,
    solvers: list[Solver],
    n_samples: int,
    nfe: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Generate samples from the same source noise with each solver."""

    model.eval()
    x0 = source.sample(n_samples, device=device)
    t_grid = torch.linspace(0.0, 1.0, nfe + 1, device=device)

    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x, t)

    return {
        solver.name: solver.solve(v_fn, x0.clone(), t_grid, return_trajectory=False).detach().cpu()
        for solver in solvers
    }


def pairwise_solver_distances(
    samples: dict[str, torch.Tensor],
    metrics: tuple[str, ...] = ("mmd", "sliced_wasserstein"),
    max_samples: int = 1024,
) -> list[dict[str, Any]]:
    """Compute pairwise distances between solver-generated sample sets."""

    names = list(samples.keys())
    rows: list[dict[str, Any]] = []
    for i, name_i in enumerate(names):
        for name_j in names[i + 1 :]:
            x = samples[name_i][:max_samples]
            y = samples[name_j][:max_samples]
            row: dict[str, Any] = {"solver_i": name_i, "solver_j": name_j}
            if "mmd" in metrics:
                row["mmd"] = squared_mmd(x, y)
            if "sliced_wasserstein" in metrics:
                row["sliced_wasserstein"] = sliced_wasserstein(x, y)
            rows.append(row)
    return rows


def solver_sensitivity_summary(
    rows: list[dict[str, Any]],
    metric: str = "sliced_wasserstein",
) -> dict[str, float]:
    """Summarize pairwise solver spread by max and mean distance."""

    values = [float(row[metric]) for row in rows if metric in row]
    if not values:
        return {f"{metric}_max": float("nan"), f"{metric}_mean": float("nan")}
    return {f"{metric}_max": max(values), f"{metric}_mean": sum(values) / len(values)}


def save_samples(samples: dict[str, torch.Tensor], output_dir: str | Path, suffix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in samples.items():
        np.save(output_dir / f"{name}_{suffix}.npy", value.numpy())


def write_distance_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["nfe", "solver_i", "solver_j", "mmd", "sliced_wasserstein"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
