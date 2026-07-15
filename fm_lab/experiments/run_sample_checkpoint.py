"""Generate samples and trajectory plots from a saved training checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fm_lab.experiments.factory import (
    build_model,
    build_path,
    build_solvers,
    build_source,
    build_target,
    resolve_device,
)
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.image_diagnostics.canvas_explorer import prepare_array_sprite_atlases
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.training.trainer import sample_and_plot, validate_checkpoint_compatibility
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import ConfigError, deep_update, load_config, save_config
from fm_lab.utils.logging import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sample artifacts from an existing fm_lab checkpoint."
    )
    parser.add_argument("--run-dir", required=True, help="Completed training run directory.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to <run-dir>/checkpoint.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to writing into --run-dir.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--n-samples", type=int, default=None, help="Override sampling.n_samples.")
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=None,
        help="Override sampling.n_trajectories.",
    )
    parser.add_argument("--nfe", type=int, default=None, help="Override sampling.nfe.")
    parser.add_argument(
        "--weights",
        choices=("raw", "ema"),
        default="raw",
        help="Checkpoint weights used for sampling. Defaults to raw model weights.",
    )
    parser.add_argument(
        "--classifier-free-guidance-scale",
        type=float,
        default=None,
        help="Override sampling.classifier_free_guidance.scale.",
    )
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=None,
        help="Maximum points shown in generated sample plots.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=None,
        help="Batch size used when integrating final generated samples.",
    )
    parser.add_argument(
        "--trajectory-target-max-points",
        type=int,
        default=None,
        help="Maximum target reference points shown in trajectory plots.",
    )
    umap_group = parser.add_mutually_exclusive_group()
    umap_group.add_argument(
        "--trajectory-umap",
        action="store_true",
        help="Enable 3D UMAP trajectory plots for this sampling pass.",
    )
    umap_group.add_argument(
        "--no-trajectory-umap",
        action="store_true",
        help="Disable 3D UMAP trajectory plots for this sampling pass.",
    )
    parser.add_argument(
        "--trajectory-umap-target-points",
        type=int,
        default=None,
        help="Maximum target points included in trajectory UMAP fits.",
    )
    parser.add_argument(
        "--trajectory-umap-neighbors",
        type=int,
        default=None,
        help="UMAP n_neighbors for trajectory plots.",
    )
    parser.add_argument(
        "--trajectory-umap-min-dist",
        type=float,
        default=None,
        help="UMAP min_dist for trajectory plots.",
    )
    parser.add_argument(
        "--prior-guidance-scale",
        type=float,
        default=None,
        help="Scale Gaussian source latents before deterministic sampling.",
    )
    parser.add_argument(
        "--density-guidance-quantile",
        type=float,
        default=None,
        help="Density-guidance quantile q. Values below 0.5 target higher density.",
    )
    parser.add_argument(
        "--density-guidance-strength",
        type=float,
        default=None,
        help="Multiplier for density-guidance score scaling. Default: 1.",
    )
    parser.add_argument(
        "--density-guidance-t-min",
        type=float,
        default=None,
        help="Earliest sampler time where density guidance is active.",
    )
    parser.add_argument(
        "--density-guidance-t-max",
        type=float,
        default=None,
        help="Latest sampler time where density guidance is active.",
    )
    parser.add_argument(
        "--density-guidance-prior-quantile",
        type=float,
        default=None,
        help=(
            "Initial Gaussian prior shell quantile for density guidance. "
            "Default is 0.5, matching the reference implementation."
        ),
    )
    parser.add_argument(
        "--no-density-guidance-prior-rescale",
        action="store_true",
        help="Disable reference-style prior shell rescaling for density guidance.",
    )
    parser.add_argument(
        "--register-dataset",
        default=None,
        help="Register generated samples as a geometry dataset variant, e.g. mnist/generated.",
    )
    parser.add_argument(
        "--register-only",
        action="store_true",
        help=(
            "Register existing generated samples from --output-dir without running sampling. "
            "Requires --register-dataset and diagnostics/checkpoint_sampling.json."
        ),
    )
    parser.add_argument(
        "--dataset-workspace",
        default=None,
        help=(
            "Geometry explorer workspace. Defaults to config data.workspace "
            "or outputs/geometry_explorer."
        ),
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Label assigned to every generated sample. Defaults to the variant name.",
    )
    parser.add_argument(
        "--dataset-solver",
        default="auto",
        help="Generated sample solver to register. Defaults to the first configured solver.",
    )
    parser.add_argument(
        "--dataset-base",
        default="generated",
        help="Base variant metadata written when registering generated samples.",
    )
    parser.add_argument(
        "--dataset-split",
        default="generated",
        help="Split metadata written when registering generated samples.",
    )
    parser.add_argument(
        "--dataset-atlas-tile-size",
        type=int,
        default=28,
        help="Tile size for prepacked generated-sample sprite atlases.",
    )
    parser.add_argument(
        "--dataset-atlas-size",
        type=int,
        default=2048,
        help="Atlas image size for prepacked generated-sample sprite atlases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise ConfigError(f"Checkpoint does not exist: {checkpoint_path}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "register_only", False):
        _register_existing_samples(
            args=args,
            run_dir=run_dir,
            output_dir=output_dir,
        )
        return

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config = load_config(run_dir / "config.yaml")
    sampling_overrides = _sampling_overrides(args)
    if sampling_overrides:
        config = deep_update(config, {"sampling": sampling_overrides})

    validate_checkpoint_compatibility(checkpoint, active_config=config)

    if output_dir.resolve() != run_dir.resolve():
        config = deep_update(config, {"experiment": {"output_dir": str(output_dir)}})
        save_config(config, output_dir / "config.yaml")

    target = build_target(config)
    source = build_source(config)
    path = build_path(config)
    model = build_model(config, dim=source.dim)
    weights = str(getattr(args, "weights", "raw"))
    state_key = "model_state_dict" if weights == "raw" else "ema_model_state_dict"
    state_dict = checkpoint.get(state_key)
    if not isinstance(state_dict, dict):
        raise ConfigError(
            f"Checkpoint does not contain requested {weights} weights ({state_key})."
        )
    model.load_state_dict(state_dict)
    model.to(device)
    solvers = build_solvers(config)

    summary = sample_and_plot(
        config=config,
        run_dir=output_dir,
        target=target,
        source=source,
        path=path,
        model=model,
        solvers=solvers,
        device=device,
    )
    summary["checkpoint_weights"] = weights
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "sampling": summary,
    }
    if getattr(args, "register_dataset", None) is not None:
        dataset = _register_generated_dataset(
            variant_id=str(args.register_dataset),
            run_dir=run_dir,
            output_dir=output_dir,
            config=config,
            summary=summary,
            target_metadata=target.metadata(),
            solver=_dataset_solver(getattr(args, "dataset_solver", "auto"), solvers),
            workspace=args.dataset_workspace
            or config.get("data", {}).get("workspace")
            or DEFAULT_WORKSPACE,
            label=getattr(args, "dataset_label", None),
            base=getattr(args, "dataset_base", "generated"),
            split=getattr(args, "dataset_split", "generated"),
            atlas_tile_size=getattr(args, "dataset_atlas_tile_size", 28),
            atlas_size=getattr(args, "dataset_atlas_size", 2048),
        )
        payload["registered_dataset"] = dataset
    output_path = output_dir / "diagnostics" / "checkpoint_sampling.json"
    payload["outputs"] = {"json": str(output_path)}
    write_json(payload, output_path)
    print(f"Wrote checkpoint sampling artifacts: {output_path}")
    if getattr(args, "register_dataset", None) is not None:
        print(f"Registered generated dataset: {payload['registered_dataset']['variant_id']}")


def _register_existing_samples(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    output_dir: Path,
) -> None:
    if getattr(args, "register_dataset", None) is None:
        raise ConfigError("--register-only requires --register-dataset.")
    summary_path = output_dir / "diagnostics" / "checkpoint_sampling.json"
    if not summary_path.exists():
        raise ConfigError(f"Checkpoint sampling summary does not exist: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("sampling")
    if not isinstance(summary, dict):
        raise ConfigError(f"Checkpoint sampling summary is missing 'sampling': {summary_path}")

    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    config = load_config(config_path)
    target = build_target(config)
    solvers = build_solvers(config)
    dataset = _register_generated_dataset(
        variant_id=str(args.register_dataset),
        run_dir=run_dir,
        output_dir=output_dir,
        config=config,
        summary=summary,
        target_metadata=target.metadata(),
        solver=_dataset_solver(getattr(args, "dataset_solver", "auto"), solvers),
        workspace=args.dataset_workspace
        or config.get("data", {}).get("workspace")
        or DEFAULT_WORKSPACE,
        label=getattr(args, "dataset_label", None),
        base=getattr(args, "dataset_base", "generated"),
        split=getattr(args, "dataset_split", "generated"),
        atlas_tile_size=getattr(args, "dataset_atlas_tile_size", 28),
        atlas_size=getattr(args, "dataset_atlas_size", 2048),
    )
    payload["registered_dataset"] = dataset
    payload["outputs"] = {"json": str(summary_path)}
    write_json(payload, summary_path)
    print(f"Registered generated dataset: {dataset['variant_id']}")
    print(f"Dataset index: {dataset['dataset_path']}")


def _sampling_overrides(args: argparse.Namespace) -> dict:
    sampling: dict = {}
    if args.n_samples is not None:
        sampling["n_samples"] = args.n_samples
    if args.n_trajectories is not None:
        sampling["n_trajectories"] = args.n_trajectories
    if args.nfe is not None:
        sampling["nfe"] = args.nfe
    if args.plot_max_points is not None:
        sampling["plot_max_points"] = args.plot_max_points
    if args.sample_batch_size is not None:
        sampling["sample_batch_size"] = args.sample_batch_size
    if args.trajectory_target_max_points is not None:
        sampling["trajectory_target_max_points"] = args.trajectory_target_max_points
    cfg_scale = getattr(args, "classifier_free_guidance_scale", None)
    if cfg_scale is not None:
        if cfg_scale < 0:
            raise ValueError("--classifier-free-guidance-scale must be non-negative.")
        sampling["classifier_free_guidance"] = {"scale": cfg_scale}

    trajectory_umap: dict = {}
    if args.trajectory_umap:
        trajectory_umap["enabled"] = True
    if args.no_trajectory_umap:
        trajectory_umap["enabled"] = False
    if args.trajectory_umap_target_points is not None:
        trajectory_umap["max_target_points"] = args.trajectory_umap_target_points
    if args.trajectory_umap_neighbors is not None:
        trajectory_umap["n_neighbors"] = args.trajectory_umap_neighbors
    if args.trajectory_umap_min_dist is not None:
        trajectory_umap["min_dist"] = args.trajectory_umap_min_dist
    if trajectory_umap:
        sampling["trajectory_umap"] = trajectory_umap
    guidance: dict[str, Any] = {}
    prior_guidance_scale = getattr(args, "prior_guidance_scale", None)
    if prior_guidance_scale is not None:
        guidance["prior"] = {"scale": prior_guidance_scale}
    density_guidance: dict[str, Any] = {}
    density_guidance_quantile = getattr(args, "density_guidance_quantile", None)
    density_guidance_strength = getattr(args, "density_guidance_strength", None)
    density_guidance_t_min = getattr(args, "density_guidance_t_min", None)
    density_guidance_t_max = getattr(args, "density_guidance_t_max", None)
    density_guidance_prior_quantile = getattr(
        args,
        "density_guidance_prior_quantile",
        None,
    )
    if density_guidance_quantile is not None:
        density_guidance["quantile"] = density_guidance_quantile
    if density_guidance_strength is not None:
        density_guidance["strength"] = density_guidance_strength
    if density_guidance_t_min is not None:
        density_guidance["t_min"] = density_guidance_t_min
    if density_guidance_t_max is not None:
        density_guidance["t_max"] = density_guidance_t_max
    if density_guidance_prior_quantile is not None:
        density_guidance["prior_rescale_quantile"] = density_guidance_prior_quantile
    if getattr(args, "no_density_guidance_prior_rescale", False):
        density_guidance["prior_rescale_quantile"] = None
    if density_guidance:
        guidance["density"] = density_guidance
    if guidance:
        sampling["guidance"] = guidance
    return sampling


def _dataset_solver(value: str, solvers: list) -> str:
    if value != "auto":
        return value
    if not solvers:
        raise ConfigError("Cannot infer dataset solver because no solvers are configured.")
    return str(solvers[0].name)


def _register_generated_dataset(
    *,
    variant_id: str,
    run_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    target_metadata: dict[str, Any],
    solver: str,
    workspace: str | Path,
    label: str | None,
    base: str,
    split: str,
    atlas_tile_size: int,
    atlas_size: int,
) -> dict[str, Any]:
    family, variant = _split_variant_id(variant_id)
    nfe = int(summary["nfe"])
    sample_path = output_dir / "samples" / f"{solver}_nfe{nfe}.npy"
    if not sample_path.exists():
        raise ConfigError(f"Generated samples do not exist: {sample_path}")

    samples = np.asarray(np.load(sample_path), dtype=np.float32)
    if samples.ndim != 2:
        raise ConfigError(f"Generated samples must have shape (n, dim), got {samples.shape}.")
    image_shape = tuple(int(value) for value in summary.get("image_shape") or ())
    if not image_shape:
        image_shape = tuple(int(value) for value in target_metadata.get("image_shape") or ())
    value_range = tuple(float(value) for value in summary.get("image_value_range") or ())
    if not value_range:
        value_range = tuple(
            float(value) for value in target_metadata.get("image_value_range") or ()
        )
    if not image_shape:
        raise ConfigError("Cannot register generated dataset without image_shape metadata.")
    if len(value_range) != 2:
        raise ConfigError("Cannot register generated dataset without image_value_range metadata.")

    registry = GeometryRegistry(workspace)
    dataset_dir = registry.workspace / "datasets" / family / variant
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_path = dataset_dir / "data.npy"
    labels_path = dataset_dir / "labels.npy"
    dataset_path = dataset_dir / "dataset_index.parquet"
    config_path = dataset_dir / "config_used.yaml"
    manifest_path = dataset_dir / "manifest.json"
    assigned_label = label or variant
    labels = np.asarray([assigned_label] * len(samples))

    np.save(data_path, samples)
    np.save(labels_path, labels)
    metadata = _generated_metadata(
        samples=samples,
        family=family,
        variant=variant,
        variant_id=variant_id,
        label=assigned_label,
        run_dir=run_dir,
        output_dir=output_dir,
        sample_path=sample_path,
    )
    atlas_bundle = prepare_array_sprite_atlases(
        metadata,
        samples,
        output_dir=dataset_dir / "assets" / "atlases",
        image_shape=image_shape,
        image_value_range=value_range,
        tile_size=atlas_tile_size,
        max_atlas_size=atlas_size,
    )
    metadata = _with_sprite_columns(
        atlas_bundle.frame,
        atlas_paths=atlas_bundle.atlas_paths,
        tile_size=atlas_bundle.tile_size,
        atlas_columns=atlas_bundle.atlas_columns,
        atlas_size=atlas_size,
    )
    write_parquet(metadata, dataset_path)
    save_config(
        {
            "family": family,
            "variant": variant,
            "base": base,
            "split": split,
            "source": {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "sample_path": str(sample_path),
                "checkpoint": str(run_dir / "checkpoint.pt"),
                "solver": solver,
                "nfe": nfe,
            },
            "sampling": summary,
        },
        config_path,
    )
    label_counts = {assigned_label: int(len(samples))}
    manifest = {
        "variant_id": variant_id,
        "family": family,
        "variant": variant,
        "base": base,
        "split": split,
        "rows": int(len(samples)),
        "label_counts": label_counts,
        "image_shape": list(image_shape),
        "value_range": list(value_range),
        "dataset_path": str(dataset_path),
        "data_path": str(data_path),
        "labels_path": str(labels_path),
        "source_run_dir": str(run_dir),
        "source_output_dir": str(output_dir),
        "source_sample_path": str(sample_path),
        "solver": solver,
        "nfe": nfe,
        "guidance": summary.get("guidance", {}),
    }
    write_json(manifest, manifest_path)
    registry.register_dataset_variant(
        variant_id=variant_id,
        family=family,
        variant=variant,
        base=base,
        split=split,
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=config_path,
        row_count=len(samples),
        label_counts=label_counts,
        image_shape=image_shape,
        value_range=value_range,
    )
    return {
        "variant_id": variant_id,
        "dataset_path": str(dataset_path),
        "data_path": str(data_path),
        "labels_path": str(labels_path),
        "manifest_path": str(manifest_path),
        "rows": int(len(samples)),
        "solver": solver,
        "nfe": nfe,
        "guidance": summary.get("guidance", {}),
    }


def _split_variant_id(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise ConfigError("Registered dataset id must be formatted as family/variant.")
    family, variant = value.split("/", 1)
    if not family or not variant:
        raise ConfigError("Registered dataset id must be formatted as family/variant.")
    return family, variant


def _generated_metadata(
    *,
    samples: np.ndarray,
    family: str,
    variant: str,
    variant_id: str,
    label: str,
    run_dir: Path,
    output_dir: Path,
    sample_path: Path,
) -> pd.DataFrame:
    row_ids = np.arange(len(samples), dtype=int)
    return pd.DataFrame(
        {
            "row_id": row_ids,
            "image_path": "",
            "dataset": family,
            "split": "generated",
            "label": label,
            "family": label,
            "prompt_id": label,
            "prompt": label,
            "tags": [[family, "generated", label] for _ in row_ids],
            "source_index": row_ids,
            "original_index": row_ids,
            "sample_type": "generated",
            "status": "success",
            "variant_id": variant_id,
            "variant": variant,
            "base_variant": "generated",
            "source_run_dir": str(run_dir),
            "source_output_dir": str(output_dir),
            "source_sample_path": str(sample_path),
        }
    )


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
