"""Run path-law geometry diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics import radial_deviation, radial_tangent_velocity_2d
from fm_lab.experiments.factory import (
    build_coupling,
    build_path,
    build_source,
    build_target,
    resolve_device,
)
from fm_lab.plotting import plot_time_profile
from fm_lab.utils.config import deep_update, load_config
from fm_lab.utils.logging import create_run_dir, write_json
from fm_lab.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fm_lab geometry diagnostics.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--output-dir", default=None, help="Optional diagnostics run directory.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cpu, cuda, or mps.",
    )
    parser.add_argument("--n-samples", type=int, default=4096, help="Samples per time value.")
    parser.add_argument("--save-raw", action="store_true", help="Save sampled xt and velocities.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or _default_geometry_dir(config)
    config = deep_update(config, {"experiment": {"output_dir": output_dir}})
    seed_everything(int(config.get("experiment", {}).get("seed", 0)))
    run_dir = create_run_dir(config, root=output_dir)
    rows = run_geometry_diagnostics(
        config=config,
        run_dir=run_dir,
        device=resolve_device(args.device),
        n_samples=args.n_samples,
        save_raw=args.save_raw,
    )
    _write_rows(rows, run_dir / "diagnostics" / "geometry_time.csv")
    plot_time_profile(
        rows,
        run_dir / "plots" / "geometry_time.png",
        value_keys=(
            "radial_deviation_mean",
            "radial_velocity_abs_mean",
            "tangent_velocity_abs_mean",
        ),
    )
    write_json({"rows": rows}, run_dir / "diagnostics" / "geometry_time.json")
    print(f"Finished geometry diagnostics: {run_dir}")


def run_geometry_diagnostics(
    *,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    n_samples: int,
    save_raw: bool,
) -> list[dict[str, float]]:
    target = build_target(config)
    source = build_source(config)
    coupling = build_coupling(config)
    path = build_path(config)
    t_values_config = config.get("diagnostics", {}).get("ambiguity", {}).get(
        "t_values",
        [0.25, 0.5, 0.75],
    )
    t_values = [float(value) for value in t_values_config]
    radii = _infer_radii(config, target.metadata())

    rows: list[dict[str, float]] = []
    for t_value in t_values:
        x0 = source.sample(n_samples, device=device)
        x1 = target.sample(n_samples, device=device)
        x0, x1 = coupling.pair(x0, x1)
        t = torch.full((n_samples,), t_value, device=device)
        xt = path.sample_xt(x0, x1, t)
        velocity = path.target_velocity(x0, x1, t)
        row: dict[str, float] = {"t": t_value}

        radial = radial_deviation(xt, radii)
        row["radial_deviation_mean"] = radial["radial_deviation_mean"]
        row["radial_deviation_max"] = radial["radial_deviation_max"]
        if xt.shape[1] == 2:
            decomposition = radial_tangent_velocity_2d(xt, velocity)
            row["radial_velocity_abs_mean"] = decomposition["radial_velocity_abs_mean"]
            row["tangent_velocity_abs_mean"] = decomposition["tangent_velocity_abs_mean"]
            row["normal_tangent_ratio"] = decomposition["normal_tangent_ratio"]
        rows.append(row)

        if save_raw:
            np.savez_compressed(
                run_dir / "diagnostics" / f"geometry_path_t{t_value:.3f}.npz",
                xt=xt.detach().cpu().numpy(),
                velocity=velocity.detach().cpu().numpy(),
                radial_deviation=radial["radial_deviation"].numpy(),
            )
    return rows


def _infer_radii(config: dict[str, Any], metadata: dict[str, Any]) -> tuple[float, ...]:
    geometry_config = config.get("diagnostics", {}).get("geometry", {})
    if "radii" in geometry_config:
        return tuple(float(value) for value in geometry_config["radii"])
    if "radius" in geometry_config:
        return (float(geometry_config["radius"]),)
    if "radii" in metadata:
        return tuple(float(value) for value in metadata["radii"])
    if "radius" in metadata:
        return (float(metadata["radius"]),)
    if "inner_radius" in metadata and "outer_radius" in metadata:
        return (float(metadata["inner_radius"]), float(metadata["outer_radius"]))
    raise ValueError("Could not infer geometry radii; set diagnostics.geometry.radius or radii.")


def _write_rows(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["t"] + sorted({key for row in rows for key in row.keys()} - {"t"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _default_geometry_dir(config: dict[str, Any]) -> str:
    base = Path(config.get("experiment", {}).get("output_dir", "runs/geometry"))
    return str(base.with_name(f"{base.name}_geometry"))


if __name__ == "__main__":
    main()
