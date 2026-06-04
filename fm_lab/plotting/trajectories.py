"""Plot generated samples and trajectories for 2D experiments."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def plot_generated_samples(
    target_samples: torch.Tensor,
    generated: dict[str, torch.Tensor],
    output_path: str | Path,
    max_points: int = 3000,
) -> None:
    """Plot target samples next to solver-generated samples."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_panels = 1 + len(generated)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)
    axes_flat = axes.ravel()

    target_np = _to_numpy(target_samples[:max_points])
    axes_flat[0].scatter(target_np[:, 0], target_np[:, 1], s=4, alpha=0.6, linewidths=0)
    axes_flat[0].set_title("target")
    _format_axis(axes_flat[0])

    for axis, (name, samples) in zip(axes_flat[1:], generated.items(), strict=True):
        samples_np = _to_numpy(samples[:max_points])
        axis.scatter(samples_np[:, 0], samples_np[:, 1], s=4, alpha=0.6, linewidths=0)
        axis.set_title(name)
        _format_axis(axis)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectories(
    trajectory: torch.Tensor,
    output_path: str | Path,
    target_samples: torch.Tensor | None = None,
    max_target_points: int = 1500,
) -> None:
    """Plot sample trajectories with an optional target reference cloud."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_np = _to_numpy(trajectory)
    fig, axis = plt.subplots(1, 1, figsize=(5, 5))
    if target_samples is not None:
        target_np = _to_numpy(target_samples[:max_target_points])
        axis.scatter(target_np[:, 0], target_np[:, 1], s=3, alpha=0.18, linewidths=0, color="black")

    for idx in range(trajectory_np.shape[1]):
        axis.plot(trajectory_np[:, idx, 0], trajectory_np[:, idx, 1], linewidth=0.7, alpha=0.55)
    axis.scatter(trajectory_np[0, :, 0], trajectory_np[0, :, 1], s=8, alpha=0.5, label="source")
    axis.scatter(trajectory_np[-1, :, 0], trajectory_np[-1, :, 1], s=8, alpha=0.7, label="final")
    axis.legend(frameon=False, loc="best")
    _format_axis(axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _format_axis(axis) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18, linewidth=0.5)


def _configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
