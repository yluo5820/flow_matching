"""Training entry point.

Phase 1 wires config loading, seeding, and run-directory creation. The actual
flow matching training loop is added in the next implementation stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fm_lab.utils.config import deep_update, load_config
from fm_lab.utils.logging import create_run_dir
from fm_lab.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an fm_lab training experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional run directory override. Defaults to experiment.output_dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the run directory and metadata without launching training.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--dataset-variant",
        default=None,
        help="Registered dataset variant id, e.g. mnist/tail_digit1.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Geometry explorer workspace used to resolve --dataset-variant.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override training.steps for quick runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training.batch_size.",
    )
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
    parser.add_argument("--objective", default=None, help="Override objective.name.")
    parser.add_argument("--objective-loss", default=None, help="Override objective.loss.")
    parser.add_argument(
        "--diffusion-prediction-type",
        default=None,
        choices=("epsilon", "score", "velocity"),
        help="Override objective.prediction_type for diffusion objectives.",
    )
    parser.add_argument(
        "--straightness-weight",
        type=float,
        default=None,
        help="Override objective.straightness.weight.",
    )
    parser.add_argument(
        "--straightness-sample-size",
        type=int,
        default=None,
        help="Override objective.straightness.sample_size.",
    )
    parser.add_argument(
        "--direction-weight",
        type=float,
        default=None,
        help="Override objective.direction_weight for direction_only_straight.",
    )
    parser.add_argument(
        "--speed-weight",
        type=float,
        default=None,
        help="Override objective.speed_weight for direction_only_straight.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_overrides = _training_overrides(args)
    if training_overrides:
        config = deep_update(config, {"training": training_overrides})
    if args.output_dir is not None:
        config = deep_update(config, {"experiment": {"output_dir": args.output_dir}})
    data_overrides = _data_overrides(args)
    if data_overrides:
        config = deep_update(config, {"data": data_overrides})
    sampling_overrides = _sampling_overrides(args)
    if sampling_overrides:
        config = deep_update(config, {"sampling": sampling_overrides})
    objective_overrides = _objective_overrides(args)
    if objective_overrides:
        config = deep_update(config, {"objective": objective_overrides})

    experiment = config.setdefault("experiment", {})
    seed = int(experiment.get("seed", 0))
    seed_everything(seed)

    run_dir = create_run_dir(config, root=args.output_dir, unique=args.output_dir is None)
    if args.dry_run:
        print(f"Created dry-run directory: {Path(run_dir)}")
        return

    from fm_lab.experiments.factory import (
        build_coupling,
        build_model,
        build_path,
        build_solvers,
        build_source,
        build_target,
        resolve_device,
    )
    from fm_lab.training.trainer import train_flow_matching

    target = build_target(config)
    source = build_source(config)
    coupling = build_coupling(config)
    path = build_path(config)
    model = build_model(config, dim=source.dim)
    solvers = build_solvers(config)
    device = resolve_device(args.device)

    metrics = train_flow_matching(
        config=config,
        run_dir=run_dir,
        target=target,
        source=source,
        coupling=coupling,
        path=path,
        model=model,
        solvers=solvers,
        device=device,
    )
    if config.get("data", {}).get("variant_id"):
        from fm_lab.geometry_explorer.trajectories import register_completed_run

        register_completed_run(
            run_dir,
            workspace=config.get("data", {}).get("workspace", "outputs/geometry_explorer"),
        )
    print(f"Finished run: {run_dir}")
    early_stopping = metrics.get("early_stopping", {})
    if early_stopping.get("stopped"):
        print(
            "Stopped early at step "
            f"{metrics['trained_steps']} "
            f"(best step {early_stopping.get('best_step')})."
        )
    print(f"Final loss: {metrics['final_loss']:.6f}")


def _objective_overrides(args: argparse.Namespace) -> dict:
    objective: dict = {}
    if args.objective is not None:
        objective["name"] = args.objective
    if args.objective_loss is not None:
        objective["loss"] = args.objective_loss
    if args.diffusion_prediction_type is not None:
        objective["prediction_type"] = args.diffusion_prediction_type
    if args.direction_weight is not None:
        objective["direction_weight"] = args.direction_weight
    if args.speed_weight is not None:
        objective["speed_weight"] = args.speed_weight

    straightness = {}
    if args.straightness_weight is not None:
        straightness["weight"] = args.straightness_weight
    if args.straightness_sample_size is not None:
        straightness["sample_size"] = args.straightness_sample_size
    if straightness:
        objective["straightness"] = straightness
    return objective


def _training_overrides(args: argparse.Namespace) -> dict:
    training = {}
    if args.steps is not None:
        training["steps"] = args.steps
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    return training


def _data_overrides(args: argparse.Namespace) -> dict:
    data = {}
    if args.dataset_variant is not None:
        data["variant_id"] = args.dataset_variant
        data["name"] = args.dataset_variant.split("/", 1)[0]
    if args.workspace is not None:
        data["workspace"] = args.workspace
    return data


def _sampling_overrides(args: argparse.Namespace) -> dict:
    sampling = {}
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
    return sampling


if __name__ == "__main__":
    main()
