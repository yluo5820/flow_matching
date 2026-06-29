"""Register model runs and trajectory UMAP views in the geometry workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fm_lab.diagnostics.trajectory_umap import TrajectoryUMAPConfig, project_saved_trajectories
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.utils.config import load_config


def build_and_register_trajectory_view(
    *,
    run_dir: str | Path,
    workspace: str | Path = DEFAULT_WORKSPACE,
    solver: str = "auto",
    nfe: int = 64,
    max_target_points: int = 3000,
    max_trajectories: int | None = None,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> dict[str, Any]:
    """Project saved trajectories and register the resulting view."""

    registry = GeometryRegistry(workspace)
    source_run_dir = Path(run_dir).expanduser().resolve()
    config = _load_run_config(source_run_dir)
    family, variant, variant_id = _variant_from_config(config, registry)
    run_id = source_run_dir.name
    output_dir = registry.workspace / "model_runs" / family / variant / run_id
    summary = project_saved_trajectories(
        TrajectoryUMAPConfig(
            run_dir=source_run_dir,
            output_dir=output_dir,
            solver=solver,
            nfe=nfe,
            max_target_points=max_target_points,
            max_trajectories=max_trajectories,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            save_coordinates=True,
        )
    )
    registry.register_model_run(
        run_id=run_id,
        run_dir=source_run_dir,
        variant_id=variant_id,
        family=family,
        variant=variant,
        config_path=source_run_dir / "config.yaml",
        metrics_path=source_run_dir / "metrics.json",
    )
    registered = []
    for solver_name, result in summary["results"].items():
        coordinates_path = result.get("coordinates_path")
        if not coordinates_path:
            continue
        trajectory_path = source_run_dir / "trajectories" / f"{solver_name}_nfe{nfe}.npy"
        generated_path = source_run_dir / "samples" / f"{solver_name}_nfe{nfe}.npy"
        target_path = source_run_dir / "samples" / "target_reference.npy"
        labels_path = source_run_dir / "samples" / "target_reference_labels.npy"
        view_id = f"{run_id}__{solver_name}__nfe{nfe}"
        registry.register_trajectory_view(
            view_id=view_id,
            run_id=run_id,
            variant_id=variant_id,
            solver=solver_name,
            nfe=nfe,
            coordinates_path=coordinates_path,
            trajectory_path=trajectory_path,
            generated_path=generated_path if generated_path.exists() else None,
            target_path=target_path if target_path.exists() else None,
            labels_path=labels_path if labels_path.exists() else None,
            output_dir=output_dir,
            interactive_path=result.get("interactive_path"),
            n_steps=int(result["n_steps"]),
            n_trajectories=int(result["n_trajectories"]),
        )
        registered.append(view_id)
    return {
        "run_id": run_id,
        "variant_id": variant_id,
        "output_dir": output_dir,
        "trajectory_views": registered,
        "summary": summary,
    }


def register_completed_run(
    run_dir: str | Path,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
) -> None:
    """Register a completed training run without building trajectory views."""

    registry = GeometryRegistry(workspace)
    path = Path(run_dir).expanduser().resolve()
    config = _load_run_config(path)
    family, variant, variant_id = _variant_from_config(config, registry)
    registry.register_model_run(
        run_id=path.name,
        run_dir=path,
        variant_id=variant_id,
        family=family,
        variant=variant,
        config_path=path / "config.yaml",
        metrics_path=path / "metrics.json",
    )
    _register_existing_trajectory_views(
        registry=registry,
        run_dir=path,
        run_id=path.name,
        variant_id=variant_id,
    )


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    return load_config(config_path) if config_path.exists() else {}


def _register_existing_trajectory_views(
    *,
    registry: GeometryRegistry,
    run_dir: Path,
    run_id: str,
    variant_id: str | None,
) -> None:
    for coordinates_path in sorted((run_dir / "trajectories").glob("*_nfe*_umap3d.npz")):
        parsed = _parse_coordinate_name(coordinates_path)
        if parsed is None:
            continue
        solver, nfe = parsed
        with np.load(coordinates_path) as coordinates:
            trajectory = np.asarray(coordinates["trajectory"])
        trajectory_path = run_dir / "trajectories" / f"{solver}_nfe{nfe}.npy"
        if not trajectory_path.exists():
            continue
        generated_path = run_dir / "samples" / f"{solver}_nfe{nfe}.npy"
        target_path = run_dir / "samples" / "target_reference.npy"
        labels_path = run_dir / "samples" / "target_reference_labels.npy"
        interactive_path = run_dir / "plots" / f"trajectory_umap3d_{solver}_nfe{nfe}.html"
        registry.register_trajectory_view(
            view_id=f"{run_id}__{solver}__nfe{nfe}",
            run_id=run_id,
            variant_id=variant_id,
            solver=solver,
            nfe=nfe,
            coordinates_path=coordinates_path,
            trajectory_path=trajectory_path,
            generated_path=generated_path if generated_path.exists() else None,
            target_path=target_path if target_path.exists() else None,
            labels_path=labels_path if labels_path.exists() else None,
            output_dir=run_dir,
            interactive_path=interactive_path if interactive_path.exists() else None,
            n_steps=int(trajectory.shape[0]),
            n_trajectories=int(trajectory.shape[1]),
        )


def _parse_coordinate_name(path: Path) -> tuple[str, int] | None:
    stem = path.stem
    suffix = "_umap3d"
    if not stem.endswith(suffix) or "_nfe" not in stem:
        return None
    base = stem[: -len(suffix)]
    solver, raw_nfe = base.rsplit("_nfe", 1)
    try:
        return solver, int(raw_nfe)
    except ValueError:
        return None


def _variant_from_config(
    config: dict[str, Any],
    registry: GeometryRegistry,
) -> tuple[str, str, str | None]:
    data = config.get("data", {})
    configured = data.get("variant_id")
    if configured:
        try:
            registry.get_dataset_variant(str(configured))
            family, variant = str(configured).split("/", 1)
            return family, variant, str(configured)
        except KeyError:
            family, variant = str(configured).split("/", 1)
            return family, variant, None
    family = str(data.get("name", "unknown")).lower()
    variant = "original"
    candidate = f"{family}/{variant}"
    try:
        registry.get_dataset_variant(candidate)
        return family, variant, candidate
    except KeyError:
        return family, variant, None
