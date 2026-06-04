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
        "--steps",
        type=int,
        default=None,
        help="Override training.steps for quick runs.",
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Override sampling.n_samples.")
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=None,
        help="Override sampling.n_trajectories.",
    )
    parser.add_argument("--nfe", type=int, default=None, help="Override sampling.nfe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.steps is not None:
        config = deep_update(config, {"training": {"steps": args.steps}})
    if args.output_dir is not None:
        config = deep_update(config, {"experiment": {"output_dir": args.output_dir}})
    sampling_overrides = {}
    if args.n_samples is not None:
        sampling_overrides["n_samples"] = args.n_samples
    if args.n_trajectories is not None:
        sampling_overrides["n_trajectories"] = args.n_trajectories
    if args.nfe is not None:
        sampling_overrides["nfe"] = args.nfe
    if sampling_overrides:
        config = deep_update(config, {"sampling": sampling_overrides})

    experiment = config.setdefault("experiment", {})
    seed = int(experiment.get("seed", 0))
    seed_everything(seed)

    run_dir = create_run_dir(config, root=args.output_dir)
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
    print(f"Finished run: {run_dir}")
    print(f"Final loss: {metrics['final_loss']:.6f}")


if __name__ == "__main__":
    main()
