"""Diagnostic plotting helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import torch


def plot_time_profile(
    rows: list[dict[str, float]],
    output_path: str | Path,
    value_keys: tuple[str, ...],
) -> None:
    """Plot one or more diagnostic values against time."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(6, 4))
    times = [row["t"] for row in rows]
    for key in value_keys:
        values = [row[key] for row in rows if key in row]
        if len(values) != len(times):
            continue
        axis.plot(times, values, marker="o", linewidth=1.5, label=key)
    axis.set_xlabel("t")
    axis.set_ylabel("value")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_history(
    history: list[dict[str, Any]],
    output_path: str | Path,
    value_keys: tuple[str, ...] = (
        "loss",
        "flow_matching_loss",
        "straightness_weighted",
        "straightness_loss",
        "kernel_vstar_straightness_weighted",
        "kernel_vstar_straightness_loss",
        "interpolant_acceleration_weighted",
        "interpolant_acceleration_loss",
        "direction_weighted",
        "speed_weighted",
        "direction_loss",
        "speed_loss",
        "direction_speed_vector_mse",
    ),
) -> None:
    """Plot training loss and available objective components over steps."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(6, 4))
    plotted_values: list[float] = []
    for key in value_keys:
        pairs = [
            (int(row["step"]), float(row[key]))
            for row in history
            if key in row and row[key] == row[key]
        ]
        if not pairs:
            continue
        steps, values = zip(*pairs, strict=True)
        plotted_values.extend(values)
        axis.plot(steps, values, marker="o", linewidth=1.4, label=key)

    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    if plotted_values and all(value > 0 for value in plotted_values):
        axis.set_yscale("log")
    axis.grid(alpha=0.25)
    if plotted_values:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_comparison(
    histories: Mapping[str, list[dict[str, Any]]],
    output_path: str | Path,
    value_key: str = "loss",
) -> None:
    """Plot one loss/statistic curve for multiple training runs."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(6.5, 4.25))
    plotted_values: list[float] = []
    for label, rows in histories.items():
        pairs = _numeric_step_pairs(rows, value_key)
        if not pairs:
            continue
        steps, values = zip(*pairs, strict=True)
        plotted_values.extend(values)
        axis.plot(steps, values, linewidth=1.6, label=label)

    axis.set_xlabel("step")
    axis.set_ylabel(value_key)
    if plotted_values and all(value > 0 for value in plotted_values):
        axis.set_yscale("log")
    axis.grid(alpha=0.25)
    if plotted_values:
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmap(
    heatmap: torch.Tensor,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot a 2D ambiguity heatmap."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(5, 4))
    image = axis.imshow(heatmap, origin="lower", interpolation="nearest")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_distance_matrix(
    rows: list[dict],
    labels: list[str],
    metric: str,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot a pairwise solver-distance matrix from long-form rows."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for row in rows:
        i = index[row["solver_i"]]
        j = index[row["solver_j"]]
        matrix[i, j] = matrix[j, i] = float(row[metric])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(5, 4))
    image = axis.imshow(matrix, interpolation="nearest")
    axis.set_title(title)
    axis.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _numeric_step_pairs(rows: list[dict[str, Any]], value_key: str) -> list[tuple[int, float]]:
    pairs = []
    for row in rows:
        if "step" not in row or value_key not in row:
            continue
        try:
            step = int(float(row["step"]))
            value = float(row[value_key])
        except (TypeError, ValueError):
            continue
        if isfinite(value):
            pairs.append((step, value))
    return pairs
