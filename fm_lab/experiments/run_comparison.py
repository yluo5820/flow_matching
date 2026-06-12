"""Run controlled experiment comparisons and aggregate diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from fm_lab.experiments.factory import (
    build_coupling,
    build_model,
    build_path,
    build_solvers,
    build_source,
    build_target,
    resolve_device,
)
from fm_lab.experiments.run_diagnostics import run_path_law_diagnostics
from fm_lab.experiments.run_field_diagnostics import run_field_diagnostics
from fm_lab.experiments.run_geometry import run_geometry_diagnostics
from fm_lab.experiments.run_solver_sensitivity import run_solver_sensitivity
from fm_lab.plotting import plot_time_profile
from fm_lab.training.trainer import train_flow_matching
from fm_lab.utils.checkpoints import load_checkpoint
from fm_lab.utils.config import ConfigError, deep_update, load_config, save_config
from fm_lab.utils.logging import create_run_dir, write_json
from fm_lab.utils.seeding import seed_everything

DEFAULT_STAGES = ("train", "path", "field")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled fm_lab comparison.")
    parser.add_argument("--matrix", required=True, help="Comparison matrix YAML.")
    parser.add_argument("--output-dir", default=None, help="Comparison output directory override.")
    parser.add_argument("--device", default=None, help="Device override: auto, cpu, cuda, or mps.")
    parser.add_argument("--stages", default=None, help="Comma-separated stages to run.")
    parser.add_argument("--steps", type=int, default=None, help="Override training.steps.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override sampling/solver samples.",
    )
    parser.add_argument(
        "--diagnostic-samples",
        type=int,
        default=None,
        help="Override path samples.",
    )
    parser.add_argument(
        "--field-samples",
        type=int,
        default=None,
        help="Override field diagnostic samples.",
    )
    parser.add_argument("--nfe", type=int, default=None, help="Override sampling.nfe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_config(args.matrix)
    cli_overrides = _cli_overrides(args)
    if cli_overrides:
        matrix = deep_update(matrix, cli_overrides)
    output_dir = args.output_dir or matrix.get("experiment", {}).get("output_dir")
    if output_dir is None:
        raise ConfigError(
            "Comparison matrix must define experiment.output_dir or use --output-dir."
        )

    result = run_comparison(matrix=matrix, output_dir=Path(output_dir))
    print(f"Finished comparison: {result['output_dir']}")


def run_comparison(matrix: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run all configured variants and write aggregate outputs."""

    runner = matrix.get("runner", {})
    stages = tuple(runner.get("stages", DEFAULT_STAGES))
    device = resolve_device(runner.get("device", "auto"))
    seed_everything(int(matrix.get("experiment", {}).get("seed", 0)))
    comparison_dir = create_run_dir(matrix, root=output_dir)
    save_config(matrix, comparison_dir / "matrix.yaml")

    summaries = []
    for variant in matrix.get("variants", []):
        summaries.append(
            run_variant(
                variant=variant,
                matrix=matrix,
                comparison_dir=comparison_dir,
                stages=stages,
                device=device,
            )
        )

    summary_path = comparison_dir / "summary.csv"
    _write_summary_csv(summaries, summary_path)
    write_json({"variants": summaries}, comparison_dir / "summary.json")
    _write_report(matrix, summaries, comparison_dir / "report.md")
    return {"output_dir": str(comparison_dir), "variants": summaries}


def run_variant(
    *,
    variant: dict[str, Any],
    matrix: dict[str, Any],
    comparison_dir: Path,
    stages: tuple[str, ...],
    device: torch.device,
) -> dict[str, Any]:
    name = variant["name"]
    variant_dir = comparison_dir / "variants" / name
    config = _variant_config(variant, matrix, variant_dir)
    runner = matrix.get("runner", {})
    checkpoint_path = variant.get("checkpoint")
    summary: dict[str, Any] = {"variant": name, "run_dir": str(variant_dir)}

    if "train" in stages:
        train_dir = variant_dir / "train"
        train_run_dir = create_run_dir(config, root=train_dir)
        model = build_model(config, dim=build_source(config).dim)
        metrics = train_flow_matching(
            config=config,
            run_dir=train_run_dir,
            target=build_target(config),
            source=build_source(config),
            coupling=build_coupling(config),
            path=build_path(config),
            model=model,
            solvers=build_solvers(config),
            device=device,
        )
        checkpoint_path = str(train_run_dir / "checkpoint.pt")
        summary["final_loss"] = metrics["final_loss"]
        summary["trained_steps"] = metrics.get("trained_steps")
        summary["early_stopped"] = metrics.get("early_stopping", {}).get("stopped", False)
        summary["checkpoint"] = checkpoint_path

    if checkpoint_path is None and any(stage in stages for stage in ("field", "solver")):
        raise ConfigError(f"Variant {name} needs training enabled or a checkpoint path.")

    if "path" in stages:
        path_dir = create_run_dir(config, root=variant_dir / "path_diagnostics")
        path_rows = run_path_law_diagnostics(
            config=config,
            run_dir=path_dir,
            device=device,
            n_samples=int(runner.get("path_diagnostic_samples", 4096)),
            bins=int(runner.get("ambiguity_bins", 50)),
            knn_k=int(runner.get("knn_k", 32)),
            save_raw=bool(runner.get("save_raw", False)),
        )
        _write_rows(path_rows, path_dir / "diagnostics" / "ambiguity_time.csv")
        plot_time_profile(
            path_rows,
            path_dir / "plots" / "ambiguity_time.png",
            value_keys=("grid_ambiguity", "knn_ambiguity", "bayes_gap"),
        )
        summary.update(_summarize_rows(path_rows, prefix="path"))

    payload = None
    if checkpoint_path is not None:
        payload = load_checkpoint(checkpoint_path, map_location="cpu")

    if "field" in stages and payload is not None:
        field_dir = create_run_dir(config, root=variant_dir / "field_diagnostics")
        field_rows = run_field_diagnostics(
            payload=payload,
            config=config,
            device=device,
            n_samples=int(runner.get("field_diagnostic_samples", 256)),
        )
        _write_rows(field_rows, field_dir / "diagnostics" / "field_stats.csv")
        summary.update(_summarize_rows(field_rows, prefix="field"))

    if "solver" in stages and payload is not None:
        solver_dir = create_run_dir(config, root=variant_dir / "solver_sensitivity")
        solver_summaries = run_solver_sensitivity(
            payload=payload,
            config=config,
            run_dir=solver_dir,
            device=device,
            n_samples=int(runner.get("solver_samples", 1024)),
            max_metric_samples=int(runner.get("max_metric_samples", 1024)),
        )
        write_json(
            {"summaries": solver_summaries},
            solver_dir / "diagnostics" / "solver_sensitivity.json",
        )
        summary.update(_summarize_rows(solver_summaries, prefix="solver"))

    if "geometry" in stages:
        geometry_dir = create_run_dir(config, root=variant_dir / "geometry")
        geometry_rows = run_geometry_diagnostics(
            config=config,
            run_dir=geometry_dir,
            device=device,
            n_samples=int(runner.get("geometry_samples", 4096)),
            save_raw=bool(runner.get("save_raw", False)),
        )
        _write_rows(geometry_rows, geometry_dir / "diagnostics" / "geometry_time.csv")
        plot_time_profile(
            geometry_rows,
            geometry_dir / "plots" / "geometry_time.png",
            value_keys=(
                "radial_deviation_mean",
                "radial_velocity_abs_mean",
                "tangent_velocity_abs_mean",
            ),
        )
        summary.update(_summarize_rows(geometry_rows, prefix="geometry"))

    return summary


def _variant_config(
    variant: dict[str, Any],
    matrix: dict[str, Any],
    variant_dir: Path,
) -> dict[str, Any]:
    config = load_config(variant["config"])
    config = deep_update(config, matrix.get("runner", {}).get("overrides", {}))
    config = deep_update(config, variant.get("overrides", {}))
    return deep_update(
        config,
        {
            "experiment": {
                "name": variant["name"],
                "output_dir": str(variant_dir / "train"),
            }
        },
    )


def _summarize_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key != "t" and isinstance(value, int | float) and value == value
        }
    )
    summary = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row and row[key] == row[key]]
        if values:
            summary[f"{prefix}_{key}_mean"] = sum(values) / len(values)
            summary[f"{prefix}_{key}_max"] = max(values)
    return summary


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["t"] + sorted({key for row in rows for key in row.keys()} - {"t"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["variant"] + sorted({key for row in rows for key in row.keys()} - {"variant"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(matrix: dict[str, Any], summaries: list[dict[str, Any]], path: Path) -> None:
    title = matrix.get("experiment", {}).get("name", "comparison")
    lines = [
        f"# {title}",
        "",
        "## Variants",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"### {summary['variant']}",
                "",
                f"- Final loss: {_format_metric(summary.get('final_loss'))}",
                f"- Trained steps: {_format_metric(summary.get('trained_steps'))}",
                f"- Early stopped: {_format_metric(summary.get('early_stopped'))}",
                f"- Mean kNN ambiguity: {_format_metric(summary.get('path_knn_ambiguity_mean'))}",
                f"- Mean Bayes gap: {_format_metric(summary.get('path_bayes_gap_mean'))}",
                "- Mean acceleration: "
                f"{_format_metric(summary.get('field_acceleration_mean_mean'))}",
                "- Max solver spread: "
                f"{_format_metric(summary.get('solver_sliced_wasserstein_max_max'))}",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `summary.csv`: aggregate numeric comparison.",
            "- `summary.json`: machine-readable summary.",
            "- `variants/*`: per-variant train and diagnostic run directories.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    runner: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    if args.device is not None:
        runner["device"] = args.device
    if args.stages is not None:
        runner["stages"] = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    if args.diagnostic_samples is not None:
        runner["path_diagnostic_samples"] = args.diagnostic_samples
    if args.field_samples is not None:
        runner["field_diagnostic_samples"] = args.field_samples
    if args.n_samples is not None:
        runner["solver_samples"] = args.n_samples
        runner["max_metric_samples"] = args.n_samples
        overrides["sampling"] = {
            "n_samples": args.n_samples,
            "n_trajectories": min(args.n_samples, 128),
        }
    if args.steps is not None:
        overrides["training"] = {"steps": args.steps}
    if args.nfe is not None:
        overrides["sampling"] = deep_update(overrides.get("sampling", {}), {"nfe": args.nfe})
        overrides["solvers"] = {"nfes": [args.nfe]}
    if overrides:
        runner["overrides"] = overrides
    return {"runner": runner} if runner else {}


if __name__ == "__main__":
    main()
