"""Compare completed training runs side by side."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.plotting import plot_generated_samples, plot_loss_comparison
from fm_lab.utils.config import ConfigError, load_config
from fm_lab.utils.logging import create_run_dir, write_json


@dataclass
class RunArtifacts:
    label: str
    run_dir: Path
    config: dict[str, Any]
    sample_path: Path
    target_path: Path
    history_path: Path
    generated_samples: torch.Tensor
    target_samples: torch.Tensor
    history: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed fm_lab training runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Completed training run directories to compare. Requires at least two.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels, one per run. Defaults to experiment names.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Comparison output directory. Defaults to runs/comparisons/<labels>.",
    )
    parser.add_argument("--nfe", type=int, default=64, help="NFE suffix to compare.")
    parser.add_argument(
        "--solver",
        default="rk4",
        help="Solver sample name, e.g. rk4. Use 'auto' when each run has one *_nfeN file.",
    )
    parser.add_argument(
        "--loss-key",
        default="loss",
        help="Training-history column to overlay in the loss comparison plot.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=10_000,
        help="Maximum target/generated points shown in the sample comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_compare_runs(
        run_dirs=[Path(value) for value in args.runs],
        labels=args.labels,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        nfe=args.nfe,
        solver=args.solver,
        loss_key=args.loss_key,
        max_points=args.max_points,
        unique_output=args.output_dir is None,
    )
    print(f"Finished run comparison: {result['output_dir']}")


def run_compare_runs(
    *,
    run_dirs: list[Path],
    labels: list[str] | None = None,
    output_dir: Path | None = None,
    nfe: int = 64,
    solver: str = "rk4",
    loss_key: str = "loss",
    max_points: int = 10_000,
    unique_output: bool = True,
) -> dict[str, Any]:
    """Compare generated samples and loss curves from completed training runs."""

    if len(run_dirs) < 2:
        raise ConfigError("Compare-runs requires at least two run directories.")

    configs = [_load_run_config(run_dir) for run_dir in run_dirs]
    resolved_labels = _resolve_labels(run_dirs, configs, labels)
    _validate_same_source_and_target(configs, resolved_labels)
    artifacts = [
        _load_run_artifacts(
            run_dir=run_dir,
            label=label,
            config=config,
            solver=solver,
            nfe=nfe,
        )
        for run_dir, label, config in zip(run_dirs, resolved_labels, configs, strict=True)
    ]

    comparison_name = _comparison_name(resolved_labels)
    if output_dir is None:
        output_dir = Path("runs") / "comparisons" / comparison_name

    comparison_config = {
        "experiment": {"name": comparison_name, "output_dir": str(output_dir)},
        "comparison": {
            "type": "completed_runs",
            "runs": [str(artifact.run_dir) for artifact in artifacts],
            "labels": resolved_labels,
            "solver": solver,
            "nfe": nfe,
            "loss_key": loss_key,
            "max_points": max_points,
            "source": artifacts[0].config.get("source", {}),
            "data": artifacts[0].config.get("data", {}),
        },
    }
    comparison_dir = create_run_dir(
        comparison_config,
        root=output_dir,
        unique=unique_output,
    )

    generated = {
        artifact.label: artifact.generated_samples for artifact in artifacts
    }
    generated_plot = comparison_dir / "plots" / f"generated_samples_nfe{nfe}.png"
    plot_generated_samples(
        target_samples=artifacts[0].target_samples,
        generated=generated,
        output_path=generated_plot,
        max_points=max_points,
    )

    histories = {artifact.label: artifact.history for artifact in artifacts}
    loss_plot = comparison_dir / "plots" / "training_loss_comparison.png"
    plot_loss_comparison(histories, loss_plot, value_key=loss_key)

    summary = {
        "output_dir": str(comparison_dir),
        "plots": {
            "generated_samples": str(generated_plot),
            "training_loss_comparison": str(loss_plot),
        },
        "runs": [
            {
                "label": artifact.label,
                "run_dir": str(artifact.run_dir),
                "sample_path": str(artifact.sample_path),
                "target_path": str(artifact.target_path),
                "history_path": str(artifact.history_path),
            }
            for artifact in artifacts
        ],
        "solver": solver,
        "nfe": nfe,
        "loss_key": loss_key,
        "max_points": max_points,
    }
    write_json(summary, comparison_dir / "summary.json")
    return summary


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise ConfigError(f"Run directory is missing config.yaml: {run_dir}")
    return load_config(config_path)


def _load_run_artifacts(
    *,
    run_dir: Path,
    label: str,
    config: dict[str, Any],
    solver: str,
    nfe: int,
) -> RunArtifacts:
    sample_path = _resolve_sample_path(run_dir, solver=solver, nfe=nfe)
    target_path = run_dir / "samples" / "target_reference.npy"
    history_path = run_dir / "diagnostics" / "training_history.csv"
    for path in (target_path, history_path):
        if not path.exists():
            raise ConfigError(f"Required comparison artifact is missing: {path}")

    return RunArtifacts(
        label=label,
        run_dir=run_dir,
        config=config,
        sample_path=sample_path,
        target_path=target_path,
        history_path=history_path,
        generated_samples=torch.as_tensor(np.load(sample_path)),
        target_samples=torch.as_tensor(np.load(target_path)),
        history=_load_history(history_path),
    )


def _resolve_sample_path(run_dir: Path, *, solver: str, nfe: int) -> Path:
    samples_dir = run_dir / "samples"
    if solver != "auto":
        sample_path = samples_dir / f"{solver}_nfe{nfe}.npy"
        if sample_path.exists():
            return sample_path
        raise ConfigError(
            f"Required generated sample file is missing: {sample_path}. "
            "Use --solver to select another solver, or --solver auto when there is one match."
        )

    matches = sorted(samples_dir.glob(f"*_nfe{nfe}.npy"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ConfigError(f"No generated sample files found for nfe={nfe} in {samples_dir}.")
    raise ConfigError(
        f"Found multiple generated sample files for nfe={nfe} in {samples_dir}; "
        "pass --solver explicitly."
    )


def _load_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_same_source_and_target(configs: list[dict[str, Any]], labels: list[str]) -> None:
    baseline = _source_target_signature(configs[0])
    baseline_label = labels[0]
    for config, label in zip(configs[1:], labels[1:], strict=True):
        signature = _source_target_signature(config)
        if signature != baseline:
            raise ConfigError(
                "Run source/target mismatch: "
                f"{label} does not match {baseline_label}."
            )


def _source_target_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": config.get("source", {}),
        "data": config.get("data", {}),
    }


def _resolve_labels(
    run_dirs: list[Path],
    configs: list[dict[str, Any]],
    labels: list[str] | None,
) -> list[str]:
    if labels is not None:
        if len(labels) != len(run_dirs):
            raise ConfigError("--labels must provide exactly one label per --runs entry.")
        if len(set(labels)) != len(labels):
            raise ConfigError("--labels values must be unique.")
        return labels

    raw_labels = [
        str(config.get("experiment", {}).get("name") or run_dir.name)
        for run_dir, config in zip(run_dirs, configs, strict=True)
    ]
    return _make_unique(raw_labels)


def _make_unique(labels: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique = []
    for label in labels:
        count = counts.get(label, 0)
        counts[label] = count + 1
        unique.append(label if count == 0 else f"{label}_{count + 1}")
    return unique


def _comparison_name(labels: list[str]) -> str:
    slug = "_vs_".join(_slugify(label) for label in labels[:3])
    if len(labels) > 3:
        slug = f"{slug}_and_{len(labels) - 3}_more"
    return f"compare_{slug}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return slug or "run"


if __name__ == "__main__":
    main()
