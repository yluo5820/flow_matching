"""Run learned-field curvature and Jacobian diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from fm_lab.diagnostics import curvature_stats, jacobian_stats
from fm_lab.experiments.factory import (
    build_coupling,
    build_model,
    build_path,
    build_source,
    build_target,
    resolve_device,
)
from fm_lab.experiments.sampling import sample_path_batch
from fm_lab.training.losses import build_objective
from fm_lab.training.prediction import velocity_model_for_objective
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import ConfigError, deep_update, load_config
from fm_lab.utils.logging import create_run_dir
from fm_lab.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned-field diagnostics.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path.")
    parser.add_argument("--config", default=None, help="Optional config override path.")
    parser.add_argument("--output-dir", default=None, help="Diagnostics run directory.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--n-samples", type=int, default=256, help="Samples per time value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    config = load_config(args.config) if args.config else payload["config"]
    output_dir = args.output_dir or _default_field_dir(args.checkpoint)
    config = deep_update(config, {"experiment": {"output_dir": output_dir}})
    seed_everything(int(config.get("experiment", {}).get("seed", 0)))
    run_dir = create_run_dir(config, root=output_dir, unique=args.output_dir is None)
    device = resolve_device(args.device)

    rows = run_field_diagnostics(
        payload=payload,
        config=config,
        device=device,
        n_samples=args.n_samples,
    )
    _write_rows(rows, run_dir / "diagnostics" / "field_stats.csv")
    print(f"Finished field diagnostics: {run_dir}")


def run_field_diagnostics(
    *,
    payload: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    n_samples: int,
) -> list[dict[str, float]]:
    target = build_target(config)
    source = build_source(config)
    coupling = build_coupling(config)
    path = build_path(config)
    model = build_model(config, dim=source.dim)
    if getattr(model, "requires_source_label", False):
        raise ConfigError(
            "Learned-field diagnostics currently assume an Eulerian model v(x,t); "
            "direction-only label-conditioned models are unsupported in v1."
        )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    if isinstance(path, torch.nn.Module):
        if "path_state_dict" in payload:
            path.load_state_dict(payload["path_state_dict"])
        path.to(device)
        path.eval()
    objective = build_objective(config.get("objective", {}))
    model = velocity_model_for_objective(model, path, objective)
    model.eval()
    ambiguity_config = config.get("diagnostics", {}).get("ambiguity", {})
    t_values = [float(value) for value in ambiguity_config.get("t_values", [0.25, 0.5, 0.75])]

    rows: list[dict[str, float]] = []
    for t_value in t_values:
        samples = sample_path_batch(
            source=source,
            target=target,
            coupling=coupling,
            path=path,
            n_samples=n_samples,
            t_value=t_value,
            device=device,
        )
        xt = samples["xt"]
        t = samples["t"]
        curvature = curvature_stats(model, xt, t)
        jacobian = jacobian_stats(model, xt, t)
        rows.append(
            {
                "t": t_value,
                "acceleration_mean": curvature["acceleration_mean"],
                "acceleration_max": curvature["acceleration_max"],
                "acceleration_sq_mean": curvature["acceleration_sq_mean"],
                "jacobian_frobenius_mean": jacobian["frobenius_mean"],
                "jacobian_frobenius_max": jacobian["frobenius_max"],
                "jacobian_spectral_mean": jacobian["spectral_mean"],
                "jacobian_spectral_max": jacobian["spectral_max"],
                "divergence_mean": jacobian["divergence_mean"],
                "divergence_std": jacobian["divergence_std"],
            }
        )
    return rows


def _write_rows(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["t"] + sorted({key for row in rows for key in row.keys()} - {"t"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _default_field_dir(checkpoint: str) -> str:
    return str(Path(checkpoint).parent / "field_diagnostics")


if __name__ == "__main__":
    main()
