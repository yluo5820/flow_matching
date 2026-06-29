"""Plot generated samples and trajectories for 2D and 3D experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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


def plot_umap_projected_trajectories(
    trajectory: torch.Tensor | np.ndarray,
    output_path: str | Path,
    *,
    target_samples: torch.Tensor | np.ndarray | None = None,
    max_target_points: int = 3000,
    max_trajectories: int | None = None,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    coordinates_path: str | Path | None = None,
    interactive_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fit a 3D UMAP embedding to high-dimensional trajectories and plot paths."""

    output_path = Path(output_path)
    _configure_matplotlib_cache(output_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trajectory_np = _as_numpy_array(trajectory)
    if trajectory_np.ndim != 3:
        raise ValueError(
            "UMAP trajectory plotting requires shape "
            f"(n_steps, n_trajectories, dim), got {trajectory_np.shape}."
        )
    if trajectory_np.shape[0] < 1 or trajectory_np.shape[1] < 1:
        raise ValueError("UMAP trajectory plotting requires at least one path state.")
    if max_trajectories is not None:
        if max_trajectories < 1:
            raise ValueError("max_trajectories must be positive when provided.")
        trajectory_np = trajectory_np[:, :max_trajectories, :]

    target_np = None
    if target_samples is not None and max_target_points > 0:
        target_np = _as_numpy_array(target_samples)[:max_target_points]
        if target_np.ndim != 2:
            raise ValueError(f"Target samples must have shape (n, dim), got {target_np.shape}.")
        if target_np.shape[1] != trajectory_np.shape[2]:
            raise ValueError(
                f"Target dim {target_np.shape[1]} does not match trajectory dim "
                f"{trajectory_np.shape[2]}."
            )

    flat_trajectory = trajectory_np.reshape(-1, trajectory_np.shape[-1])
    embedding_inputs = [flat_trajectory] if target_np is None else [target_np, flat_trajectory]
    embedding_input = np.concatenate(embedding_inputs, axis=0).astype(np.float32, copy=False)
    projected = _compute_umap3d(
        embedding_input,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    offset = 0
    target_projected = None
    if target_np is not None:
        target_projected = projected[: len(target_np)]
        offset = len(target_np)
    trajectory_projected = projected[offset:].reshape(
        trajectory_np.shape[0],
        trajectory_np.shape[1],
        3,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(6.0, 5.8), subplot_kw={"projection": "3d"})
    if target_projected is not None and len(target_projected) > 0:
        axis.scatter(
            target_projected[:, 0],
            target_projected[:, 1],
            target_projected[:, 2],
            s=3,
            alpha=0.14,
            color="black",
            linewidths=0,
            depthshade=True,
            label="target",
        )

    for idx in range(trajectory_projected.shape[1]):
        path = trajectory_projected[:, idx, :]
        axis.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color="0.42",
            linewidth=0.7,
            alpha=0.5,
        )
    axis.scatter(
        trajectory_projected[0, :, 0],
        trajectory_projected[0, :, 1],
        trajectory_projected[0, :, 2],
        s=16,
        alpha=0.75,
        color="#2563eb",
        linewidths=0,
        depthshade=True,
        label="source",
    )
    axis.scatter(
        trajectory_projected[-1, :, 0],
        trajectory_projected[-1, :, 1],
        trajectory_projected[-1, :, 2],
        s=18,
        alpha=0.85,
        color="#dc2626",
        linewidths=0,
        depthshade=True,
        label="final",
    )
    axis.legend(frameon=False, loc="best")
    bounds_arrays = [trajectory_projected]
    if target_projected is not None:
        bounds_arrays.append(target_projected)
    _format_umap_axis(axis, bounds_arrays)
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.02, top=0.96)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    saved_coordinates_path = None
    if coordinates_path is not None:
        coordinates_output = Path(coordinates_path)
        coordinates_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            coordinates_output,
            trajectory=trajectory_projected.astype(np.float32, copy=False),
            target=np.empty((0, 3), dtype=np.float32)
            if target_projected is None
            else target_projected.astype(np.float32, copy=False),
            n_neighbors=np.asarray([n_neighbors], dtype=np.int64),
            min_dist=np.asarray([min_dist], dtype=np.float32),
            random_state=np.asarray([random_state], dtype=np.int64),
        )
        saved_coordinates_path = str(coordinates_output)

    saved_interactive_path = None
    if interactive_path is not None:
        interactive_output = Path(interactive_path)
        _write_interactive_umap_trajectory_html(
            interactive_output,
            trajectory=trajectory_projected,
            target=target_projected,
        )
        saved_interactive_path = str(interactive_output)

    return {
        "plot_path": str(output_path),
        "coordinates_path": saved_coordinates_path,
        "interactive_path": saved_interactive_path,
        "n_steps": int(trajectory_projected.shape[0]),
        "n_trajectories": int(trajectory_projected.shape[1]),
        "trajectory_points": int(flat_trajectory.shape[0]),
        "target_points": 0 if target_np is None else int(target_np.shape[0]),
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": metric,
        "random_state": int(random_state),
    }


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


def _format_umap_axis(axis, arrays: list[np.ndarray]) -> None:
    bounds = _equal_bounds(arrays, 3)
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_zlabel("UMAP 3")
    axis.set_xlim3d(*bounds[0])
    axis.set_ylim3d(*bounds[1])
    axis.set_zlim3d(*bounds[2])
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=22, azim=-55)
    axis.grid(True, alpha=0.18, linewidth=0.5)


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


def _as_numpy_array(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = _to_numpy(values)
    return np.asarray(values, dtype=np.float32)


def _compute_umap3d(
    values: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    if len(values) < 3:
        return _pca_projection(values, random_state=random_state, n_components=3)
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP trajectory plots require umap-learn. "
            "Install the image diagnostics extra, e.g. `pip install -e .[image-diagnostics]`."
        ) from exc

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(max(2, int(n_neighbors)), len(values) - 1),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(random_state),
    )
    return np.asarray(reducer.fit_transform(values), dtype=np.float32)


def _pca_projection(values: np.ndarray, *, random_state: int, n_components: int) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0, n_components), dtype=np.float32)
    if len(values) == 1:
        return np.zeros((1, n_components), dtype=np.float32)
    from sklearn.decomposition import PCA

    components = min(n_components, len(values), values.shape[1])
    projected = PCA(n_components=components, random_state=random_state).fit_transform(values)
    if components < n_components:
        projected = np.column_stack(
            [projected, np.zeros((len(projected), n_components - components))]
        )
    return np.asarray(projected, dtype=np.float32)


def _write_interactive_umap_trajectory_html(
    output_path: Path,
    *,
    trajectory: np.ndarray,
    target: np.ndarray | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trajectory": np.round(trajectory, 5).tolist(),
        "target": [] if target is None else np.round(target, 5).tolist(),
    }
    output_path.write_text(_interactive_umap_html(payload), encoding="utf-8")


def _interactive_umap_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UMAP Trajectory Explorer</title>
<style>
  :root {{
    color-scheme: dark;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #111;
    color: #e5e7eb;
  }}
  body {{
    margin: 0;
    min-height: 100vh;
    display: grid;
    grid-template-rows: 1fr auto;
    background: #111;
  }}
  canvas {{
    width: 100vw;
    height: calc(100vh - 96px);
    display: block;
    cursor: grab;
  }}
  canvas:active {{ cursor: grabbing; }}
  .controls {{
    box-sizing: border-box;
    display: grid;
    grid-template-columns: auto 1fr auto auto;
    gap: 14px;
    align-items: center;
    min-height: 96px;
    padding: 14px 18px;
    background: #1f1f1f;
    border-top: 1px solid #333;
  }}
  button {{
    border: 1px solid #4b5563;
    background: #2d3748;
    color: #f9fafb;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 14px;
  }}
  input[type="range"] {{ width: 100%; }}
  .readout {{
    min-width: 128px;
    color: #d1d5db;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }}
  .hint {{
    color: #9ca3af;
    font-size: 13px;
  }}
</style>
</head>
<body>
<canvas id="scene"></canvas>
<div class="controls">
  <button id="play">Play</button>
  <input id="time" type="range" min="0" max="0" value="0" step="1">
  <div id="readout" class="readout">t = 0 / 0</div>
  <div class="hint">slider shows cumulative paths; drag rotate, wheel zoom</div>
</div>
<script>
const data = {data_json};
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("time");
const readout = document.getElementById("readout");
const playButton = document.getElementById("play");
const steps = data.trajectory.length;
const paths = steps ? data.trajectory[0].length : 0;
let step = Math.max(0, steps - 1);
let yaw = -0.78;
let pitch = 0.38;
let zoom = 1.0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let playing = false;
let timer = null;

slider.max = Math.max(0, steps - 1);
slider.value = String(step);

const bounds = computeBounds();
const center = [
  0.5 * (bounds.min[0] + bounds.max[0]),
  0.5 * (bounds.min[1] + bounds.max[1]),
  0.5 * (bounds.min[2] + bounds.max[2])
];
const span = Math.max(
  bounds.max[0] - bounds.min[0],
  bounds.max[1] - bounds.min[1],
  bounds.max[2] - bounds.min[2],
  1e-6
);

function computeBounds() {{
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const visit = (point) => {{
    for (let i = 0; i < 3; i++) {{
      if (point[i] < min[i]) min[i] = point[i];
      if (point[i] > max[i]) max[i] = point[i];
    }}
  }};
  for (const targetPoint of data.target) visit(targetPoint);
  for (const timeSlice of data.trajectory) {{
    for (const point of timeSlice) visit(point);
  }}
  for (let i = 0; i < 3; i++) {{
    if (!Number.isFinite(min[i]) || !Number.isFinite(max[i])) {{
      min[i] = -1;
      max[i] = 1;
    }}
  }}
  return {{min, max}};
}}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function project(point) {{
  const x0 = (point[0] - center[0]) / span;
  const y0 = (point[1] - center[1]) / span;
  const z0 = (point[2] - center[2]) / span;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x0 + sy * z0;
  const z1 = -sy * x0 + cy * z0;
  const y1 = cp * y0 - sp * z1;
  const z2 = sp * y0 + cp * z1;
  const rect = canvas.getBoundingClientRect();
  const scale = Math.min(rect.width, rect.height) * 1.25 * zoom;
  return {{
    x: rect.width * 0.5 + x1 * scale,
    y: rect.height * 0.52 - y1 * scale,
    z: z2
  }};
}}

function drawPoint(point, radius, color, alpha) {{
  const p = project(point);
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fill();
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, rect.width, rect.height);

  if (data.target.length) {{
    for (const point of data.target) drawPoint(point, 1.5, "#d4d4d4", 0.16);
  }}

  ctx.lineWidth = 0.8;
  ctx.strokeStyle = "#8b949e";
  ctx.globalAlpha = 0.46;
  for (let i = 0; i < paths; i++) {{
    ctx.beginPath();
    for (let t = 0; t <= step; t++) {{
      const p = project(data.trajectory[t][i]);
      if (t === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }}
    ctx.stroke();
  }}

  if (steps) {{
    const current = data.trajectory[step];
    const sorted = current.map((point, index) => {{
      const projected = project(point);
      return {{point, index, z: projected.z}};
    }}).sort((a, b) => a.z - b.z);
    for (const entry of sorted) drawPoint(entry.point, 4.2, "#ef4444", 0.9);
  }}

  ctx.globalAlpha = 1;
  ctx.fillStyle = "#e5e7eb";
  ctx.font = "13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.fillText(`${{paths}} particles, ${{steps}} time steps`, 18, 24);
  readout.textContent = `t = ${{step}} / ${{Math.max(0, steps - 1)}}`;
}}

function setStep(value) {{
  step = Math.max(0, Math.min(steps - 1, Number(value)));
  slider.value = String(step);
  draw();
}}

function startPlayback() {{
  if (playing) return;
  playing = true;
  playButton.textContent = "Pause";
  timer = window.setInterval(() => {{
    if (step >= steps - 1) setStep(0);
    else setStep(step + 1);
  }}, 140);
}}

function stopPlayback() {{
  playing = false;
  playButton.textContent = "Play";
  if (timer !== null) window.clearInterval(timer);
  timer = null;
}}

slider.addEventListener("input", () => {{
  stopPlayback();
  setStep(slider.value);
}});
playButton.addEventListener("click", () => {{
  if (playing) stopPlayback();
  else startPlayback();
}});
canvas.addEventListener("pointerdown", (event) => {{
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", (event) => {{
  if (!dragging) return;
  yaw += (event.clientX - lastX) * 0.008;
  pitch += (event.clientY - lastY) * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
  lastX = event.clientX;
  lastY = event.clientY;
  draw();
}});
canvas.addEventListener("pointerup", () => {{ dragging = false; }});
canvas.addEventListener("wheel", (event) => {{
  event.preventDefault();
  zoom *= Math.exp(-event.deltaY * 0.001);
  zoom = Math.max(0.25, Math.min(6.0, zoom));
  draw();
}}, {{passive: false}});
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""


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
