"""UMAP projection diagnostics for saved sampling trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fm_lab.plotting.trajectories import plot_umap_projected_trajectories
from fm_lab.utils.config import ConfigError
from fm_lab.utils.logging import write_json


@dataclass
class TrajectoryUMAPConfig:
    run_dir: Path
    output_dir: Path | None = None
    solver: str = "auto"
    nfe: int = 64
    max_target_points: int = 3000
    max_trajectories: int | None = None
    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    save_coordinates: bool = True


def project_saved_trajectories(config: TrajectoryUMAPConfig) -> dict[str, Any]:
    """Project saved run trajectories into 3D UMAP space and write plots."""

    run_dir = config.run_dir
    if not run_dir.exists():
        raise ConfigError(f"Run directory does not exist: {run_dir}")
    output_dir = config.output_dir or run_dir
    plots_dir = output_dir / "plots"
    trajectories_dir = output_dir / "trajectories"
    diagnostics_dir = output_dir / "diagnostics"

    target_samples = _load_target_reference(run_dir)
    trajectory_paths = _resolve_trajectory_paths(
        run_dir,
        solver=config.solver,
        nfe=config.nfe,
    )

    results: dict[str, Any] = {}
    for trajectory_path in trajectory_paths:
        solver_name = _solver_from_trajectory_path(trajectory_path, nfe=config.nfe)
        coordinates_path = None
        if config.save_coordinates:
            coordinates_path = trajectories_dir / f"{solver_name}_nfe{config.nfe}_umap3d.npz"
        result = plot_umap_projected_trajectories(
            np.load(trajectory_path),
            plots_dir / f"trajectory_umap3d_{solver_name}_nfe{config.nfe}.png",
            target_samples=target_samples,
            max_target_points=config.max_target_points,
            max_trajectories=config.max_trajectories,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=config.random_state,
            coordinates_path=coordinates_path,
            interactive_path=plots_dir / f"trajectory_umap3d_{solver_name}_nfe{config.nfe}.html",
        )
        result["trajectory_path"] = str(trajectory_path)
        results[solver_name] = result

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "nfe": int(config.nfe),
        "solver": config.solver,
        "has_target_reference": target_samples is not None,
        "results": results,
    }
    diagnostics_path = diagnostics_dir / f"trajectory_umap_nfe{config.nfe}.json"
    summary["outputs"] = {"json": str(diagnostics_path)}
    write_json(summary, diagnostics_path)
    return summary


def _load_target_reference(run_dir: Path) -> np.ndarray | None:
    target_path = run_dir / "samples" / "target_reference.npy"
    if not target_path.exists():
        return None
    return np.load(target_path)


def _resolve_trajectory_paths(run_dir: Path, *, solver: str, nfe: int) -> list[Path]:
    trajectories_dir = run_dir / "trajectories"
    if solver != "auto":
        path = trajectories_dir / f"{solver}_nfe{nfe}.npy"
        if not path.exists():
            raise ConfigError(f"Required trajectory file is missing: {path}")
        return [path]

    matches = sorted(
        path
        for path in trajectories_dir.glob(f"*_nfe{nfe}.npy")
        if not path.name.startswith("source_reference_")
    )
    if not matches:
        raise ConfigError(f"No trajectory files found for nfe={nfe} in {trajectories_dir}.")
    return matches


def _solver_from_trajectory_path(path: Path, *, nfe: int) -> str:
    suffix = f"_nfe{nfe}"
    if not path.stem.endswith(suffix):
        raise ConfigError(f"Trajectory filename does not end with {suffix!r}: {path}")
    return path.stem[: -len(suffix)]
