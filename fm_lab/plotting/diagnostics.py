"""Diagnostic plotting helpers."""

from __future__ import annotations

import os
from pathlib import Path

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


def _configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
