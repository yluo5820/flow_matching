"""Plot generated samples and trajectories for 2D and 3D experiments."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def plot_generated_samples(
    target_samples: torch.Tensor,
    generated: dict[str, torch.Tensor],
    output_path: str | Path,
    max_points: int = 10_000,
    image_shape: list[int] | tuple[int, ...] | None = None,
    image_value_range: list[float] | tuple[float, float] = (0.0, 1.0),
) -> None:
    """Plot target samples next to solver-generated samples."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_shape is not None:
        _plot_generated_image_samples(
            plt=plt,
            target_samples=target_samples,
            generated=generated,
            output_path=output_path,
            max_points=max_points,
            image_shape=image_shape,
            image_value_range=image_value_range,
        )
        return

    target_np = _to_numpy(target_samples[:max_points])
    generated_np = {
        name: _to_numpy(samples[:max_points]) for name, samples in generated.items()
    }
    plot_dim = _plot_dim(target_np)
    _validate_sample_dims([target_np, *generated_np.values()], plot_dim)
    bounds = _equal_bounds([target_np, *generated_np.values()], plot_dim)

    n_panels = 1 + len(generated)
    subplot_kwargs = {"projection": "3d"} if plot_dim == 3 else {}
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.5 * n_panels, 4.5),
        squeeze=False,
        subplot_kw=subplot_kwargs,
    )
    axes_flat = axes.ravel()

    _scatter_points(axes_flat[0], target_np, plot_dim, s=4, alpha=0.6)
    axes_flat[0].set_title("target")
    _format_axis(axes_flat[0], plot_dim, bounds)

    for axis, (name, samples_np) in zip(axes_flat[1:], generated_np.items(), strict=True):
        _scatter_points(axis, samples_np, plot_dim, s=4, alpha=0.6)
        axis.set_title(name)
        _format_axis(axis, plot_dim, bounds)

    if plot_dim == 3:
        fig.tight_layout(pad=1.5)
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.9, wspace=0.08)
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectories(
    trajectory: torch.Tensor,
    output_path: str | Path,
    target_samples: torch.Tensor | None = None,
    max_target_points: int = 1500,
    image_shape: list[int] | tuple[int, ...] | None = None,
    image_value_range: list[float] | tuple[float, float] = (0.0, 1.0),
) -> None:
    """Plot sample trajectories with an optional target reference cloud."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_shape is not None:
        _plot_image_trajectories(
            plt=plt,
            trajectory=trajectory,
            output_path=output_path,
            image_shape=image_shape,
            image_value_range=image_value_range,
        )
        return

    trajectory_np = _to_numpy(trajectory)
    plot_dim = _plot_dim(trajectory_np)
    arrays_for_bounds = [trajectory_np]
    target_np = None
    if target_samples is not None:
        target_np = _to_numpy(target_samples[:max_target_points])
        _validate_sample_dims([target_np], plot_dim)
        arrays_for_bounds.append(target_np)
    bounds = _equal_bounds(arrays_for_bounds, plot_dim)

    subplot_kwargs = {"projection": "3d"} if plot_dim == 3 else {}
    fig, axis = plt.subplots(1, 1, figsize=(5.5, 5.5), subplot_kw=subplot_kwargs)
    if target_np is not None:
        _scatter_points(axis, target_np, plot_dim, s=3, alpha=0.18, color="black")

    for idx in range(trajectory_np.shape[1]):
        _plot_line(
            axis,
            trajectory_np[:, idx, :],
            plot_dim,
            color="0.45",
            linewidth=0.6,
            alpha=0.45,
        )
    _scatter_points(axis, trajectory_np[0], plot_dim, s=8, alpha=0.5, label="source")
    _scatter_points(axis, trajectory_np[-1], plot_dim, s=8, alpha=0.7, label="final")
    axis.legend(frameon=False, loc="best")
    _format_axis(axis, plot_dim, bounds)
    if plot_dim == 3:
        fig.tight_layout(pad=1.5)
        fig.subplots_adjust(left=0.02, right=0.9, bottom=0.02, top=0.96)
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_dim(x: np.ndarray) -> int:
    dim = int(x.shape[-1])
    if dim < 2:
        raise ValueError("Plotting requires at least two coordinates.")
    return 3 if dim >= 3 else 2


def _validate_sample_dims(arrays: list[np.ndarray], plot_dim: int) -> None:
    for values in arrays:
        if values.shape[-1] < plot_dim:
            raise ValueError(
                f"Cannot make a {plot_dim}D plot from samples with shape {values.shape}."
            )


def _scatter_points(axis, values: np.ndarray, plot_dim: int, **kwargs) -> None:
    if plot_dim == 3:
        axis.scatter(
            values[:, 0],
            values[:, 1],
            values[:, 2],
            linewidths=0,
            depthshade=True,
            **kwargs,
        )
        return
    axis.scatter(values[:, 0], values[:, 1], linewidths=0, **kwargs)


def _plot_line(axis, values: np.ndarray, plot_dim: int, **kwargs) -> None:
    if plot_dim == 3:
        axis.plot(values[:, 0], values[:, 1], values[:, 2], **kwargs)
        return
    axis.plot(values[:, 0], values[:, 1], **kwargs)


def _format_axis(axis, plot_dim: int, bounds: list[tuple[float, float]]) -> None:
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
    if plot_dim == 3:
        axis.set_zlabel("x2")
        axis.set_xlim3d(*bounds[0])
        axis.set_ylim3d(*bounds[1])
        axis.set_zlim3d(*bounds[2])
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.view_init(elev=22, azim=-55)
        axis.grid(True, alpha=0.18, linewidth=0.5)
        return

    axis.set_xlim(*bounds[0])
    axis.set_ylim(*bounds[1])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18, linewidth=0.5)


def _equal_bounds(arrays: list[np.ndarray], plot_dim: int) -> list[tuple[float, float]]:
    points = np.concatenate(
        [values[..., :plot_dim].reshape(-1, plot_dim) for values in arrays],
        axis=0,
    )
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return [(-1.0, 1.0) for _ in range(plot_dim)]

    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = float(np.max(maxs - mins) * 0.525)
    radius = max(radius, 1e-3)
    return [(float(center - radius), float(center + radius)) for center in centers]


def _configure_matplotlib_cache(output_path: Path) -> None:
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def _plot_generated_image_samples(
    *,
    plt,
    target_samples: torch.Tensor,
    generated: dict[str, torch.Tensor],
    output_path: Path,
    max_points: int,
    image_shape: list[int] | tuple[int, ...],
    image_value_range: list[float] | tuple[float, float],
) -> None:
    n_images = min(max_points, 64)
    panels = {"target": _to_numpy(target_samples[:n_images])}
    panels.update({name: _to_numpy(samples[:n_images]) for name, samples in generated.items()})
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4), squeeze=False)
    vmin, vmax = float(image_value_range[0]), float(image_value_range[1])
    for axis, (name, samples) in zip(axes.ravel(), panels.items(), strict=True):
        grid = _make_image_grid(samples, image_shape=image_shape, value_range=(vmin, vmax))
        axis.imshow(grid, cmap="gray", vmin=vmin, vmax=vmax)
        axis.set_title(name)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_image_trajectories(
    *,
    plt,
    trajectory: torch.Tensor,
    output_path: Path,
    image_shape: list[int] | tuple[int, ...],
    image_value_range: list[float] | tuple[float, float],
) -> None:
    trajectory_np = _to_numpy(trajectory)
    n_steps, n_trajectories = trajectory_np.shape[:2]
    n_rows = min(n_trajectories, 8)
    n_cols = min(n_steps, 6)
    time_indices = np.linspace(0, n_steps - 1, n_cols, dtype=int)
    vmin, vmax = float(image_value_range[0]), float(image_value_range[1])

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.35 * n_cols, 1.35 * n_rows),
        squeeze=False,
    )
    for row in range(n_rows):
        for col, time_idx in enumerate(time_indices):
            image = _reshape_image_samples(
                trajectory_np[time_idx, row : row + 1],
                image_shape=image_shape,
            )[0]
            axes[row, col].imshow(np.clip(image, vmin, vmax), cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"t{time_idx}", fontsize=8)
    fig.tight_layout(pad=0.15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _make_image_grid(
    samples: np.ndarray,
    *,
    image_shape: list[int] | tuple[int, ...],
    value_range: tuple[float, float],
) -> np.ndarray:
    images = _reshape_image_samples(samples, image_shape=image_shape)
    n_images, height, width = images.shape
    grid_cols = int(np.ceil(np.sqrt(n_images)))
    grid_rows = int(np.ceil(n_images / grid_cols))
    vmin, vmax = value_range
    grid = np.full((grid_rows * height, grid_cols * width), vmin, dtype=np.float32)
    for idx, image in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = np.clip(
            image,
            vmin,
            vmax,
        )
    return grid


def _reshape_image_samples(
    samples: np.ndarray,
    *,
    image_shape: list[int] | tuple[int, ...],
) -> np.ndarray:
    height, width = _normalize_image_shape(image_shape)
    if samples.shape[-1] != height * width:
        raise ValueError(
            f"Image samples have dim {samples.shape[-1]}, expected {height * width} "
            f"for image_shape={tuple(image_shape)}."
        )
    return samples.reshape(-1, height, width)


def _normalize_image_shape(image_shape: list[int] | tuple[int, ...]) -> tuple[int, int]:
    shape = tuple(int(value) for value in image_shape)
    if len(shape) == 2:
        return shape
    if len(shape) == 3 and shape[0] == 1:
        return shape[1], shape[2]
    raise ValueError(f"Only grayscale image shapes are supported, got {shape}.")
