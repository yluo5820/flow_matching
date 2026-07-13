"""Run solver-sensitivity diagnostics for a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from fm_lab.diagnostics.solver_sensitivity import (
    generate_solver_samples,
    pairwise_solver_distances,
    save_samples,
    solver_sensitivity_summary,
    write_distance_rows,
)
from fm_lab.experiments.factory import (
    build_model,
    build_path,
    build_solvers,
    build_source,
    resolve_device,
)
from fm_lab.plotting import plot_distance_matrix
from fm_lab.training.losses import build_objective
from fm_lab.training.prediction import velocity_model_for_objective
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import deep_update, load_config
from fm_lab.utils.logging import create_run_dir, write_json
from fm_lab.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run solver-sensitivity diagnostics.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--config", default=None, help="Optional config override path.")
    parser.add_argument("--output-dir", default=None, help="Diagnostics run directory.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--n-samples", type=int, default=1024, help="Generated samples per solver.")
    parser.add_argument(
        "--max-metric-samples",
        type=int,
        default=1024,
        help="Samples used by metrics.",
    )
    parser.add_argument("--schedule", default=None, help="Optional solver time schedule override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    config = load_config(args.config) if args.config else payload["config"]
    if args.schedule is not None:
        config = deep_update(config, {"solvers": {"schedule": args.schedule}})
    output_dir = args.output_dir or _default_solver_dir(args.checkpoint)
    config = deep_update(config, {"experiment": {"output_dir": output_dir}})
    seed_everything(int(config.get("experiment", {}).get("seed", 0)))
    run_dir = create_run_dir(config, root=output_dir, unique=args.output_dir is None)
    device = resolve_device(args.device)

    summaries = run_solver_sensitivity(
        payload=payload,
        config=config,
        run_dir=run_dir,
        device=device,
        n_samples=args.n_samples,
        max_metric_samples=args.max_metric_samples,
    )
    write_json({"summaries": summaries}, run_dir / "diagnostics" / "solver_sensitivity.json")
    print(f"Finished solver sensitivity: {run_dir}")


def run_solver_sensitivity(
    *,
    payload: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    n_samples: int,
    max_metric_samples: int,
) -> list[dict[str, float]]:
    source = build_source(config)
    path = build_path(config)
    objective = build_objective(config.get("objective", {}))
    model = build_model(config, dim=source.dim)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model = velocity_model_for_objective(model, path, objective)
    solvers = build_solvers(config)
    nfes = [int(value) for value in config.get("solvers", {}).get("nfes", [16, 32, 64])]
    schedule = config.get("solvers", {}).get("schedule", "uniform")
    metric_names = tuple(
        config.get("diagnostics", {}).get("solver_sensitivity", {}).get("metrics", [])
    )
    if not metric_names:
        metric_names = ("mmd", "sliced_wasserstein")

    summaries = []
    for nfe in nfes:
        samples = generate_solver_samples(
            model=model,
            source=source,
            solvers=solvers,
            n_samples=n_samples,
            nfe=nfe,
            schedule=schedule,
            device=device,
        )
        save_samples(samples, run_dir / "samples", suffix=f"nfe{nfe}")
        rows = pairwise_solver_distances(
            samples,
            metrics=metric_names,
            max_samples=max_metric_samples,
        )
        for row in rows:
            row["nfe"] = nfe
            row["schedule"] = schedule
        write_distance_rows(rows, run_dir / "diagnostics" / f"solver_sensitivity_nfe{nfe}.csv")
        metric = "sliced_wasserstein" if "sliced_wasserstein" in metric_names else metric_names[0]
        plot_distance_matrix(
            rows,
            labels=[solver.name for solver in solvers],
            metric=metric,
            output_path=run_dir / "plots" / f"solver_sensitivity_{metric}_nfe{nfe}.png",
            title=f"{metric} nfe={nfe}",
        )
        summary = {
            "nfe": float(nfe),
            "schedule": schedule,
            **solver_sensitivity_summary(rows, metric=metric),
        }
        summaries.append(summary)
    return summaries


def _default_solver_dir(checkpoint: str) -> str:
    return str(Path(checkpoint).parent / "solver_sensitivity")


if __name__ == "__main__":
    main()
