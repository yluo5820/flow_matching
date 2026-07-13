"""Build Geometry Explorer datasets from intermediate sampling timesteps."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from fm_lab.experiments.factory import (
    build_model,
    build_path,
    build_solvers,
    build_source,
    resolve_device,
)
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.views import build_projection_view
from fm_lab.image_diagnostics.canvas_explorer import prepare_array_sprite_atlases
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.solvers import Solver, make_time_grid
from fm_lab.training.losses import build_objective
from fm_lab.training.prediction import velocity_model_for_objective
from fm_lab.training.sampling_guidance import (
    apply_density_guidance,
    apply_density_prior_rescaling,
    apply_prior_guidance,
    build_sampling_guidance_config,
)
from fm_lab.training.trainer import _model_velocity, _temporary_torch_seed
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import ConfigError, load_config, save_config
from fm_lab.utils.logging import write_json

DEFAULT_VIEW_CONFIG = Path("configs/geometry_explorer/views/raw_pixels.yaml")


@dataclass(frozen=True)
class TimestepClass:
    """One class in a timestep-labeled sampling dataset."""

    class_index: int
    step_index: int
    time: float
    label: str


@dataclass(frozen=True)
class SamplingTimestepConfig:
    """Configuration for generating a timestep-class dataset from a checkpoint."""

    run_dir: Path
    checkpoint_path: Path | None = None
    workspace: Path = DEFAULT_WORKSPACE
    variant_id: str | None = None
    device: str = "auto"
    solver: str = "auto"
    nfe: int | None = None
    schedule: str | None = None
    num_classes: int = 10
    total_rows: int = 50_000
    paths_per_class: int | None = None
    time_start: float = 0.1
    time_stop: float = 0.9
    batch_size: int = 128
    seed: int | None = None
    atlas_tile_size: int = 32
    atlas_size: int = 2048
    overwrite: bool = False


@dataclass(frozen=True)
class SamplingTimestepResult:
    """Paths and metadata for a generated timestep-class dataset."""

    variant_id: str
    output_dir: Path
    dataset_path: Path
    data_path: Path
    labels_path: Path
    manifest_path: Path
    rows: int
    paths_per_class: int
    timestep_classes: tuple[TimestepClass, ...]


def build_sampling_timestep_dataset(
    config: SamplingTimestepConfig,
) -> SamplingTimestepResult:
    """Sample a checkpoint and register intermediate states as timestep classes."""

    run_dir = config.run_dir.expanduser().resolve()
    checkpoint_path = (
        config.checkpoint_path.expanduser().resolve()
        if config.checkpoint_path is not None
        else run_dir / "checkpoint.pt"
    )
    if not checkpoint_path.exists():
        raise ConfigError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    run_config = checkpoint.get("config")
    if not isinstance(run_config, dict):
        run_config = load_config(run_dir / "config.yaml")

    device = resolve_device(config.device)
    source = build_source(run_config)
    path = build_path(run_config)
    model = build_model(run_config, dim=source.dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    _load_optional_path_state(path, checkpoint)
    model.to(device)
    model.eval()
    if isinstance(path, nn.Module):
        path.to(device)
        path.eval()

    solvers = build_solvers(run_config)
    solver = _select_solver(solvers, config.solver)
    nfe = _resolve_nfe(run_config, config.nfe)
    schedule = _resolve_schedule(run_config, config.schedule)
    t_grid = make_time_grid(nfe, schedule=schedule, device=device)
    timestep_classes = select_timestep_classes(
        t_grid.detach().cpu().numpy(),
        num_classes=config.num_classes,
        time_start=config.time_start,
        time_stop=config.time_stop,
    )
    paths_per_class = _resolve_paths_per_class(
        num_classes=len(timestep_classes),
        total_rows=config.total_rows,
        paths_per_class=config.paths_per_class,
    )
    variant_id = config.variant_id or _default_variant_id(
        run_config,
        solver=solver.name,
        nfe=nfe,
        num_classes=len(timestep_classes),
        paths_per_class=paths_per_class,
    )
    family, variant = _split_variant_id(variant_id)
    image_shape = _resolve_image_shape(run_config, dim=source.dim)
    value_range = _resolve_model_value_range(run_config)
    sampling_seed = _resolve_seed(run_config, config.seed)

    writer = TimestepDatasetWriter(
        workspace=config.workspace,
        variant_id=variant_id,
        family=family,
        variant=variant,
        base="generated_sampling_timestep",
        split="generated",
        image_shape=image_shape,
        value_range=value_range,
        timestep_classes=timestep_classes,
        paths_per_class=paths_per_class,
        dim=source.dim,
        config_payload={
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "solver": solver.name,
            "nfe": nfe,
            "schedule": schedule,
            "seed": sampling_seed,
            "num_classes": len(timestep_classes),
            "total_rows": paths_per_class * len(timestep_classes),
            "paths_per_class": paths_per_class,
            "time_start": config.time_start,
            "time_stop": config.time_stop,
            "batch_size": config.batch_size,
            "source_config": run_config.get("source", {}),
            "path_config": run_config.get("path", {}),
            "model_config": run_config.get("model", {}),
            "objective_config": run_config.get("objective", {}),
            "sampling_config": run_config.get("sampling", {}),
        },
        source_metadata={
            "source_run_dir": str(run_dir),
            "source_checkpoint": str(checkpoint_path),
            "source_solver": solver.name,
            "source_nfe": nfe,
            "source_schedule": schedule,
            "sampling_seed": sampling_seed,
        },
        atlas_tile_size=config.atlas_tile_size,
        atlas_size=config.atlas_size,
        overwrite=config.overwrite,
    )

    sampling_config = run_config.get("sampling", {})
    guidance = build_sampling_guidance_config(sampling_config)
    objective = build_objective(run_config.get("objective", {}))
    velocity_model = velocity_model_for_objective(model, path, objective)
    velocity_model = apply_density_guidance(
        base_model=model,
        velocity_model=velocity_model,
        path=path,
        objective=objective,
        config=guidance.density,
    )
    velocity_model.eval()
    step_indices = torch.as_tensor(
        [item.step_index for item in timestep_classes],
        dtype=torch.long,
        device=device,
    )
    generated_paths = 0
    with torch.no_grad(), _temporary_torch_seed(sampling_seed, device):
        while generated_paths < paths_per_class:
            batch_size = min(config.batch_size, paths_per_class - generated_paths)
            x0 = source.sample(batch_size, device=device)
            x0 = apply_prior_guidance(x0, source=source, config=guidance.prior)
            x0 = apply_density_prior_rescaling(x0, source=source, config=guidance.density)
            selected = _solve_selected_timesteps(
                solver=solver,
                model=velocity_model,
                x0=x0,
                t_grid=t_grid,
                step_indices=step_indices,
            )
            trajectory_indices = np.arange(
                generated_paths,
                generated_paths + batch_size,
                dtype=np.int64,
            )
            writer.append(selected, trajectory_indices=trajectory_indices)
            generated_paths += batch_size

    return writer.finish()


def build_sampling_timestep_dataset_from_trajectory(
    *,
    trajectory_path: str | Path,
    variant_id: str,
    workspace: str | Path = DEFAULT_WORKSPACE,
    nfe: int | None = None,
    schedule: str = "uniform",
    num_classes: int = 10,
    total_rows: int = 50_000,
    paths_per_class: int | None = None,
    time_start: float = 0.1,
    time_stop: float = 0.9,
    image_shape: tuple[int, ...] | None = None,
    value_range: tuple[float, float] = (-1.0, 1.0),
    seed: int = 0,
    atlas_tile_size: int = 32,
    atlas_size: int = 2048,
    overwrite: bool = False,
) -> SamplingTimestepResult:
    """Register selected states from an existing full trajectory array."""

    path = Path(trajectory_path).expanduser().resolve()
    trajectory = np.load(path, mmap_mode="r")
    if trajectory.ndim != 3:
        raise ConfigError(
            f"Trajectory must have shape (steps, paths, dim), got {trajectory.shape}."
        )
    inferred_nfe = int(trajectory.shape[0] - 1)
    nfe = inferred_nfe if nfe is None else int(nfe)
    if nfe != inferred_nfe:
        raise ConfigError(
            f"nfe={nfe} does not match trajectory steps {trajectory.shape[0]}."
        )
    t_grid = make_time_grid(nfe, schedule=schedule, device="cpu").numpy()
    timestep_classes = select_timestep_classes(
        t_grid,
        num_classes=num_classes,
        time_start=time_start,
        time_stop=time_stop,
    )
    requested_paths = _resolve_paths_per_class(
        num_classes=len(timestep_classes),
        total_rows=total_rows,
        paths_per_class=paths_per_class,
    )
    available_paths = int(trajectory.shape[1])
    selected_paths = min(requested_paths, available_paths)
    rng = np.random.default_rng(seed)
    positions = np.arange(available_paths, dtype=np.int64)
    if selected_paths < available_paths:
        positions = np.sort(rng.choice(positions, size=selected_paths, replace=False))
    family, variant = _split_variant_id(variant_id)
    if image_shape is None:
        image_shape = _square_or_flat_shape(int(trajectory.shape[2]))
    writer = TimestepDatasetWriter(
        workspace=workspace,
        variant_id=variant_id,
        family=family,
        variant=variant,
        base="generated_sampling_timestep",
        split="generated",
        image_shape=image_shape,
        value_range=value_range,
        timestep_classes=timestep_classes,
        paths_per_class=selected_paths,
        dim=int(trajectory.shape[2]),
        config_payload={
            "trajectory_path": str(path),
            "solver": "precomputed",
            "nfe": nfe,
            "schedule": schedule,
            "seed": seed,
            "num_classes": len(timestep_classes),
            "requested_paths_per_class": requested_paths,
            "paths_per_class": selected_paths,
            "time_start": time_start,
            "time_stop": time_stop,
        },
        source_metadata={
            "source_trajectory_path": str(path),
            "source_solver": "precomputed",
            "source_nfe": nfe,
            "source_schedule": schedule,
            "sampling_seed": seed,
        },
        atlas_tile_size=atlas_tile_size,
        atlas_size=atlas_size,
        overwrite=overwrite,
    )
    step_indices = [item.step_index for item in timestep_classes]
    states = np.asarray(trajectory[np.ix_(step_indices, positions)])
    states = states.transpose(1, 0, 2)
    writer.append(states, trajectory_indices=positions)
    return writer.finish()


class TimestepDatasetWriter:
    """Incrementally write a timestep-class dataset and register it."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        variant_id: str,
        family: str,
        variant: str,
        base: str,
        split: str,
        image_shape: tuple[int, ...],
        value_range: tuple[float, float],
        timestep_classes: tuple[TimestepClass, ...],
        paths_per_class: int,
        dim: int,
        config_payload: dict[str, Any],
        source_metadata: dict[str, Any],
        atlas_tile_size: int,
        atlas_size: int,
        overwrite: bool,
    ) -> None:
        self.registry = GeometryRegistry(workspace)
        self.variant_id = variant_id
        self.family = family
        self.variant = variant
        self.base = base
        self.split = split
        self.image_shape = image_shape
        self.value_range = value_range
        self.timestep_classes = timestep_classes
        self.paths_per_class = int(paths_per_class)
        self.dim = int(dim)
        self.config_payload = dict(config_payload)
        self.source_metadata = dict(source_metadata)
        self.atlas_tile_size = int(atlas_tile_size)
        self.atlas_size = int(atlas_size)
        self.row_count = self.paths_per_class * len(self.timestep_classes)
        self.output_dir = self.registry.workspace / "datasets" / family / variant
        if self.output_dir.exists() and not overwrite:
            raise ConfigError(
                f"Dataset output already exists: {self.output_dir}. "
                "Pass --overwrite to replace generated files."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.output_dir / "data.npy"
        self.labels_path = self.output_dir / "labels.npy"
        self.dataset_path = self.output_dir / "dataset_index.parquet"
        self.config_path = self.output_dir / "config_used.yaml"
        self.manifest_path = self.output_dir / "manifest.json"
        self._data = np.lib.format.open_memmap(
            self.data_path,
            mode="w+",
            dtype=np.float32,
            shape=(self.row_count, self.dim),
        )
        max_label_len = max(len(item.label) for item in self.timestep_classes)
        self._labels = np.empty(self.row_count, dtype=f"<U{max_label_len}")
        self._cursor = 0
        self._metadata_chunks: list[pd.DataFrame] = []

    def append(self, states: np.ndarray, *, trajectory_indices: np.ndarray) -> None:
        """Append selected states with shape (paths, classes, dim)."""

        states_np = np.asarray(states, dtype=np.float32)
        if states_np.ndim != 3:
            raise ConfigError(
                "Selected states must have shape (paths, classes, dim), "
                f"got {states_np.shape}."
            )
        n_paths, n_classes, dim = states_np.shape
        if n_classes != len(self.timestep_classes):
            raise ConfigError(
                f"Selected states have {n_classes} classes, expected {len(self.timestep_classes)}."
            )
        if dim != self.dim:
            raise ConfigError(f"Selected states have dim {dim}, expected {self.dim}.")
        trajectory_indices = np.asarray(trajectory_indices, dtype=np.int64)
        if len(trajectory_indices) != n_paths:
            raise ConfigError(
                f"Trajectory index count {len(trajectory_indices)} does not match states {n_paths}."
            )
        block_rows = n_paths * n_classes
        end = self._cursor + block_rows
        if end > self.row_count:
            raise ConfigError("More rows were appended than the configured dataset size.")

        block = states_np.reshape(block_rows, dim)
        self._data[self._cursor:end] = block
        class_labels = np.asarray([item.label for item in self.timestep_classes])
        block_labels = np.tile(class_labels, n_paths)
        self._labels[self._cursor:end] = block_labels
        self._metadata_chunks.append(
            self._metadata_block(
                row_ids=np.arange(self._cursor, end, dtype=np.int64),
                labels=block_labels,
                trajectory_indices=np.repeat(trajectory_indices, n_classes),
            )
        )
        self._cursor = end

    def finish(self) -> SamplingTimestepResult:
        """Flush files, build sprite atlases, write metadata, and register the dataset."""

        if self._cursor != self.row_count:
            raise ConfigError(
                f"Dataset is incomplete: wrote {self._cursor} rows, expected {self.row_count}."
            )
        self._data.flush()
        np.save(self.labels_path, self._labels)
        metadata = pd.concat(self._metadata_chunks, ignore_index=True)
        images = np.load(self.data_path, mmap_mode="r")
        atlas_bundle = prepare_array_sprite_atlases(
            metadata,
            images,
            output_dir=self.output_dir / "assets" / "atlases",
            image_shape=self.image_shape,
            image_value_range=self.value_range,
            tile_size=self.atlas_tile_size,
            max_atlas_size=self.atlas_size,
        )
        metadata = _with_sprite_columns(
            atlas_bundle.frame,
            atlas_paths=atlas_bundle.atlas_paths,
            tile_size=atlas_bundle.tile_size,
            atlas_columns=atlas_bundle.atlas_columns,
            atlas_size=self.atlas_size,
        )
        write_parquet(metadata, self.dataset_path)
        save_config(
            {
                "variant_id": self.variant_id,
                "family": self.family,
                "variant": self.variant,
                "base": self.base,
                "split": self.split,
                "image_shape": list(self.image_shape),
                "value_range": list(self.value_range),
                "timestep_classes": [
                    {
                        "class_index": item.class_index,
                        "step_index": item.step_index,
                        "time": item.time,
                        "label": item.label,
                    }
                    for item in self.timestep_classes
                ],
                "source": self.config_payload,
            },
            self.config_path,
        )
        label_counts = self._label_counts()
        manifest = {
            "variant_id": self.variant_id,
            "family": self.family,
            "variant": self.variant,
            "base": self.base,
            "split": self.split,
            "rows": self.row_count,
            "paths_per_class": self.paths_per_class,
            "label_counts": label_counts,
            "image_shape": list(self.image_shape),
            "value_range": list(self.value_range),
            "actual_data_min": float(np.nanmin(images)),
            "actual_data_max": float(np.nanmax(images)),
            "dataset_path": str(self.dataset_path),
            "data_path": str(self.data_path),
            "labels_path": str(self.labels_path),
            "timestep_classes": [
                {
                    "class_index": item.class_index,
                    "step_index": item.step_index,
                    "time": item.time,
                    "label": item.label,
                }
                for item in self.timestep_classes
            ],
            "source": self.config_payload,
        }
        write_json(manifest, self.manifest_path)
        self.registry.register_dataset_variant(
            variant_id=self.variant_id,
            family=self.family,
            variant=self.variant,
            base=self.base,
            split=self.split,
            dataset_path=self.dataset_path,
            data_path=self.data_path,
            labels_path=self.labels_path,
            config_path=self.config_path,
            row_count=self.row_count,
            label_counts=label_counts,
            image_shape=self.image_shape,
            value_range=self.value_range,
        )
        return SamplingTimestepResult(
            variant_id=self.variant_id,
            output_dir=self.output_dir,
            dataset_path=self.dataset_path,
            data_path=self.data_path,
            labels_path=self.labels_path,
            manifest_path=self.manifest_path,
            rows=self.row_count,
            paths_per_class=self.paths_per_class,
            timestep_classes=self.timestep_classes,
        )

    def _metadata_block(
        self,
        *,
        row_ids: np.ndarray,
        labels: np.ndarray,
        trajectory_indices: np.ndarray,
    ) -> pd.DataFrame:
        n_classes = len(self.timestep_classes)
        class_indices = np.tile(
            np.asarray([item.class_index for item in self.timestep_classes], dtype=np.int64),
            len(trajectory_indices) // n_classes,
        )
        step_indices = np.tile(
            np.asarray([item.step_index for item in self.timestep_classes], dtype=np.int64),
            len(trajectory_indices) // n_classes,
        )
        times = np.tile(
            np.asarray([item.time for item in self.timestep_classes], dtype=np.float32),
            len(trajectory_indices) // n_classes,
        )
        solver_final_step = self.source_metadata.get("source_nfe")
        is_solver_final = (
            step_indices == int(solver_final_step)
            if solver_final_step is not None
            else np.zeros(len(step_indices), dtype=bool)
        )
        frame = pd.DataFrame(
            {
                "row_id": row_ids,
                "image_path": "",
                "dataset": self.family,
                "split": self.split,
                "label": labels,
                "family": self.family,
                "prompt_id": labels,
                "prompt": labels,
                "tags": [
                    [self.family, "generated", "sampling_timestep", str(label)]
                    for label in labels
                ],
                "source_index": trajectory_indices,
                "original_index": trajectory_indices,
                "sample_type": "generated_sampling_timestep",
                "status": "success",
                "variant_id": self.variant_id,
                "variant": self.variant,
                "base_variant": self.base,
                "trajectory_index": trajectory_indices,
                "timestep_class_index": class_indices,
                "timestep_step_index": step_indices,
                "timestep_time": times,
                "timestep_label": labels,
                "is_last_selected_timestep": step_indices
                == max(item.step_index for item in self.timestep_classes),
                "is_solver_final_timestep": is_solver_final,
            }
        )
        for key, value in self.source_metadata.items():
            frame[key] = value
        return frame

    def _label_counts(self) -> dict[str, int]:
        return {
            item.label: self.paths_per_class
            for item in self.timestep_classes
        }


def select_timestep_classes(
    t_grid: np.ndarray,
    *,
    num_classes: int,
    time_start: float = 0.1,
    time_stop: float = 0.9,
) -> tuple[TimestepClass, ...]:
    """Select nearest solver grid points for evenly spaced requested times."""

    values = np.asarray(t_grid, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ConfigError("t_grid must be a one-dimensional array with at least two entries.")
    if num_classes < 1:
        raise ConfigError("num_classes must be positive.")
    if not (0.0 <= time_start <= time_stop <= 1.0):
        raise ConfigError("time_start and time_stop must satisfy 0 <= start <= stop <= 1.")
    requested = np.linspace(float(time_start), float(time_stop), int(num_classes))
    indices = [int(np.argmin(np.abs(values - value))) for value in requested]
    if len(set(indices)) != len(indices):
        raise ConfigError(
            "Selected timestep classes are not unique. Reduce --num-classes or widen "
            "the --time-start/--time-stop interval."
        )
    return tuple(
        TimestepClass(
            class_index=class_index,
            step_index=step_index,
            time=float(values[step_index]),
            label=_time_label(class_index, float(values[step_index])),
        )
        for class_index, step_index in enumerate(indices)
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for timestep-class dataset generation."""

    args = parse_args(argv)
    result = build_sampling_timestep_dataset(
        SamplingTimestepConfig(
            run_dir=Path(args.run_dir),
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            workspace=Path(args.workspace),
            variant_id=args.variant_id,
            device=args.device,
            solver=args.solver,
            nfe=args.nfe,
            schedule=args.schedule,
            num_classes=args.num_classes,
            total_rows=args.total_rows,
            paths_per_class=args.paths_per_class,
            time_start=args.time_start,
            time_stop=args.time_stop,
            batch_size=args.batch_size,
            seed=args.seed,
            atlas_tile_size=args.atlas_tile_size,
            atlas_size=args.atlas_size,
            overwrite=args.overwrite,
        )
    )
    payload: dict[str, Any] = {
        "variant_id": result.variant_id,
        "rows": result.rows,
        "paths_per_class": result.paths_per_class,
        "dataset_path": str(result.dataset_path),
        "data_path": str(result.data_path),
        "labels_path": str(result.labels_path),
        "manifest_path": str(result.manifest_path),
    }
    if args.build_view:
        view_result = build_projection_view(
            variant_id=result.variant_id,
            config_path=args.view_config,
            workspace=args.workspace,
            project_root=Path.cwd(),
        )
        payload["view"] = {
            "view_id": view_result["view_id"],
            "explorer_data": str(view_result["explorer_data"]),
            "rows": view_result["rows"],
        }
    print(write_json_to_stdout(payload))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Geometry Explorer dataset whose classes are sampling timesteps."
    )
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to <run-dir>/checkpoint.pt.",
    )
    parser.add_argument(
        "--workspace",
        default=str(DEFAULT_WORKSPACE),
        help="Geometry Explorer workspace root.",
    )
    parser.add_argument(
        "--variant-id",
        default=None,
        help=(
            "Dataset variant id, e.g. cifar10/generated_sampling_timesteps. "
            "Defaults to an inferred cifar10/generated_timestep_classes_* id."
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--solver",
        default="auto",
        help="Solver name. Defaults to first configured solver.",
    )
    parser.add_argument("--nfe", type=int, default=None, help="Override sampling nfe.")
    parser.add_argument("--schedule", default=None, help="Override sampling schedule.")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--total-rows", type=int, default=50_000)
    parser.add_argument(
        "--paths-per-class",
        type=int,
        default=None,
        help="Override number of generated paths per timestep class.",
    )
    parser.add_argument("--time-start", type=float, default=0.1)
    parser.add_argument("--time-stop", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--atlas-tile-size", type=int, default=32)
    parser.add_argument("--atlas-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--build-view",
        action="store_true",
        help="Build the raw-pixel Geometry Explorer view after registering the dataset.",
    )
    parser.add_argument(
        "--view-config",
        default=str(DEFAULT_VIEW_CONFIG),
        help="Projection view YAML used when --build-view is set.",
    )
    return parser.parse_args(argv)


def write_json_to_stdout(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def _solve_selected_timesteps(
    *,
    solver: Solver,
    model: nn.Module,
    x0: torch.Tensor,
    t_grid: torch.Tensor,
    step_indices: torch.Tensor,
) -> np.ndarray:
    def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _model_velocity(model, x, t, source_label=x0)

    trajectory = solver.solve(v_fn, x0.clone(), t_grid, return_trajectory=True)
    selected = trajectory.index_select(0, step_indices)
    return selected.detach().cpu().numpy().transpose(1, 0, 2)


def _select_solver(solvers: list[Solver], name: str) -> Solver:
    if not solvers:
        raise ConfigError("No solvers are configured.")
    if name == "auto":
        return solvers[0]
    for solver in solvers:
        if solver.name == name:
            return solver
    available = ", ".join(solver.name for solver in solvers)
    raise ConfigError(f"Solver {name!r} is not configured. Available: {available}.")


def _resolve_nfe(config: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return int(override)
    sampling = config.get("sampling", {})
    if "nfe" in sampling:
        return int(sampling["nfe"])
    solver_config = config.get("solvers", {})
    nfes = solver_config.get("nfes", [32])
    return int(max(nfes))


def _resolve_schedule(config: dict[str, Any], override: str | None) -> str:
    if override is not None:
        return str(override)
    sampling = config.get("sampling", {})
    solver_config = config.get("solvers", {})
    return str(sampling.get("schedule", solver_config.get("schedule", "uniform")))


def _resolve_paths_per_class(
    *,
    num_classes: int,
    total_rows: int,
    paths_per_class: int | None,
) -> int:
    if paths_per_class is not None:
        value = int(paths_per_class)
    else:
        value = int(total_rows) // int(num_classes)
    if value < 1:
        raise ConfigError("paths_per_class must be positive.")
    return value


def _resolve_seed(config: dict[str, Any], override: int | None) -> int | None:
    if override is not None:
        return int(override)
    sampling = config.get("sampling", {})
    if "seed" in sampling:
        return int(sampling["seed"])
    experiment = config.get("experiment", {})
    if "seed" in experiment:
        return int(experiment["seed"])
    return None


def _resolve_image_shape(config: dict[str, Any], *, dim: int) -> tuple[int, ...]:
    model_shape = config.get("model", {}).get("image_shape")
    if model_shape:
        return tuple(int(value) for value in model_shape)
    data_shape = config.get("data", {}).get("image_shape")
    if data_shape:
        return tuple(int(value) for value in data_shape)
    return _square_or_flat_shape(dim)


def _resolve_model_value_range(config: dict[str, Any]) -> tuple[float, float]:
    data_config = config.get("data", {})
    normalize = str(data_config.get("normalize", "zero_one")).lower()
    if normalize in {"minus_one_one", "-1_1", "centered"}:
        return (-1.0, 1.0)
    if normalize in {"zero_one", "01", "unit"}:
        return (0.0, 1.0)
    return tuple(float(value) for value in data_config.get("value_range", [-1.0, 1.0]))


def _load_optional_path_state(path: Any, checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("path_state_dict")
    if state is None:
        return
    if not isinstance(path, nn.Module):
        raise ConfigError(
            "Checkpoint contains path_state_dict but configured path is not trainable."
        )
    path.load_state_dict(state)


def _default_variant_id(
    config: dict[str, Any],
    *,
    solver: str,
    nfe: int,
    num_classes: int,
    paths_per_class: int,
) -> str:
    family = str(config.get("data", {}).get("name", "generated")).lower()
    variant = f"generated_timestep_classes_{solver}_nfe{nfe}_{num_classes}x{paths_per_class}"
    return f"{family}/{variant}"


def _square_or_flat_shape(dim: int) -> tuple[int, ...]:
    side = int(round(math.sqrt(dim)))
    if side * side == dim:
        return (side, side)
    return (dim,)


def _time_label(class_index: int, time_value: float) -> str:
    text = f"{time_value:.3f}".replace(".", "p")
    return f"timestep_{class_index:02d}_t{text}"


def _split_variant_id(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise ConfigError("Dataset variant id must be formatted as family/variant.")
    family, variant = value.split("/", 1)
    if not family or not variant:
        raise ConfigError("Dataset variant id must be formatted as family/variant.")
    return family, variant


def _with_sprite_columns(
    frame: pd.DataFrame,
    *,
    atlas_paths: list[Path],
    tile_size: int,
    atlas_columns: int,
    atlas_size: int,
) -> pd.DataFrame:
    output = frame.copy()
    atlas_path_values = [str(path.resolve()) for path in atlas_paths]
    output["sprite_atlas_path"] = [
        atlas_path_values[int(index)] for index in output["atlas_index"]
    ]
    output["sprite_atlas_index"] = output["atlas_index"].astype(int)
    output["sprite_atlas_column"] = output["atlas_column"].astype(int)
    output["sprite_atlas_row"] = output["atlas_row"].astype(int)
    output["sprite_tile_size"] = int(tile_size)
    output["sprite_atlas_columns"] = int(atlas_columns)
    output["sprite_atlas_size"] = int(atlas_size)
    return output.drop(columns=["atlas_index", "atlas_column", "atlas_row"], errors="ignore")


if __name__ == "__main__":
    main()
