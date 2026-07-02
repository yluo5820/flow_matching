"""Generate samples and trajectory plots from a saved training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from fm_lab.experiments.factory import (
    build_model,
    build_path,
    build_solvers,
    build_source,
    build_target,
    resolve_device,
)
from fm_lab.training.trainer import sample_and_plot
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise ConfigError(f"Checkpoint does not exist: {checkpoint_path}")

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config = load_config(run_dir / "config.yaml")
    sampling_overrides = _sampling_overrides(args)
    if sampling_overrides:
        config = deep_update(config, {"sampling": sampling_overrides})

    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.resolve() != run_dir.resolve():
        config = deep_update(config, {"experiment": {"output_dir": str(output_dir)}})
        save_config(config, output_dir / "config.yaml")

    target = build_target(config)
    source = build_source(config)
    path = build_path(config)
    model = build_model(config, dim=source.dim)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "sampling": summary,
    }
    output_path = output_dir / "diagnostics" / "checkpoint_sampling.json"
    payload["outputs"] = {"json": str(output_path)}
    write_json(payload, output_path)
    print(f"Wrote checkpoint sampling artifacts: {output_path}")


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
    return sampling


if __name__ == "__main__":
    main()
