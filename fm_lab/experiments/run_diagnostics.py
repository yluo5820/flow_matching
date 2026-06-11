"""Run path-law diagnostics without requiring a trained model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics import bayes_regression_gap_knn, grid_ambiguity, knn_ambiguity
from fm_lab.experiments.factory import (
    build_coupling,
    build_path,
    build_source,
    build_target,
    resolve_device,
)
from fm_lab.experiments.sampling import sample_path_batch
from fm_lab.plotting import plot_heatmap, plot_time_profile
from fm_lab.utils.config import deep_update, load_config
from fm_lab.utils.logging import create_run_dir, write_json
from fm_lab.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fm_lab path-law diagnostics.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--output-dir", default=None, help="Optional diagnostics run directory.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--n-samples", type=int, default=4096, help="Samples per time value.")
    parser.add_argument("--bins", type=int, default=50, help="Grid bins per axis for 2D ambiguity.")
    parser.add_argument("--knn-k", type=int, default=32, help="k for kNN ambiguity.")
    parser.add_argument("--save-raw", action="store_true", help="Save sampled xt and velocities.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or _default_diagnostics_dir(config)
    config = deep_update(config, {"experiment": {"output_dir": output_dir}})

    seed = int(config.get("experiment", {}).get("seed", 0))
    seed_everything(seed)
    run_dir = create_run_dir(config, root=output_dir)
    device = resolve_device(args.device)

    rows = run_path_law_diagnostics(
        config=config,
        run_dir=run_dir,
        device=device,
        n_samples=args.n_samples,
        bins=args.bins,
        knn_k=args.knn_k,
        save_raw=args.save_raw,
    )
    _write_rows(rows, run_dir / "diagnostics" / "ambiguity_time.csv")
    plot_time_profile(
        rows,
        run_dir / "plots" / "ambiguity_time.png",
        value_keys=("grid_ambiguity", "knn_ambiguity", "bayes_gap"),
    )
    write_json({"rows": rows}, run_dir / "diagnostics" / "ambiguity_time.json")
    print(f"Finished diagnostics: {run_dir}")


def run_path_law_diagnostics(
    *,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    n_samples: int,
    bins: int,
    knn_k: int,
    save_raw: bool,
) -> list[dict[str, float]]:
    target = build_target(config)
    source = build_source(config)
    coupling = build_coupling(config)
    path = build_path(config)
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
        velocities = samples["velocities"]

        row: dict[str, float] = {"t": t_value}
        if xt.shape[1] == 2:
            grid_result = grid_ambiguity(xt, velocities, bins=bins)
            row["grid_ambiguity"] = grid_result.ambiguity
            row["grid_valid_bins"] = float(grid_result.valid_bins)
            plot_heatmap(
                grid_result.heatmap,
                run_dir / "plots" / f"ambiguity_heatmap_t{t_value:.3f}.png",
                title=f"grid ambiguity t={t_value:.3f}",
            )
            np.savez_compressed(
                run_dir / "diagnostics" / f"grid_ambiguity_t{t_value:.3f}.npz",
                heatmap=grid_result.heatmap.numpy(),
                counts=grid_result.counts.numpy(),
                x_edges=grid_result.x_edges.numpy(),
                y_edges=grid_result.y_edges.numpy(),
            )

        knn_result = knn_ambiguity(xt, velocities, k=knn_k)
        gap_result = bayes_regression_gap_knn(xt, velocities, k=knn_k)
        row["knn_ambiguity"] = float(knn_result["ambiguity"])
        row["bayes_gap"] = float(gap_result["bayes_gap"])
        rows.append(row)

        if save_raw:
            np.savez_compressed(
                run_dir / "diagnostics" / f"path_law_t{t_value:.3f}.npz",
                xt=xt.detach().cpu().numpy(),
                velocities=velocities.detach().cpu().numpy(),
            )
    return rows


def _write_rows(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["t"] + sorted({key for row in rows for key in row.keys()} - {"t"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _default_diagnostics_dir(config: dict[str, Any]) -> str:
    base = Path(config.get("experiment", {}).get("output_dir", "runs/diagnostics"))
    return str(base.with_name(f"{base.name}_diagnostics"))


if __name__ == "__main__":
    main()
