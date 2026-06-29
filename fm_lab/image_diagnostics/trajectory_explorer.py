"""Explorer-style HTML renderer for projected sampling trajectories."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fm_lab.image_diagnostics.canvas_explorer import (
    AtlasBundle,
    atlas_data_url,
    prepare_array_sprite_atlases,
)
from fm_lab.image_diagnostics.config import ExplorerConfig
from fm_lab.image_diagnostics.palette import LABEL_PALETTE


def write_trajectory_explorer_html(
    output_path: str | Path,
    *,
    trajectory: np.ndarray,
    target: np.ndarray | None,
    generated: np.ndarray | None,
    target_images: np.ndarray | None,
    generated_images: np.ndarray | None,
    target_labels: np.ndarray | None = None,
    generated_labels: np.ndarray | None = None,
    image_shape: list[int] | tuple[int, ...] | None = None,
    image_value_range: list[float] | tuple[float, float] = (0.0, 1.0),
    dataset_name: str = "mnist",
    height: int = 760,
    config: ExplorerConfig | None = None,
) -> dict[str, Any]:
    """Write a self-contained trajectory explorer HTML page."""

    trajectory = _as_projected_trajectory(trajectory)
    target = _as_projected_points(target)
    generated = _as_projected_points(generated)
    rows, endpoint_images, endpoint_coordinates = _endpoint_rows(
        target=target,
        generated=generated,
        target_images=target_images,
        generated_images=generated_images,
        target_labels=target_labels,
        generated_labels=generated_labels,
        dataset_name=dataset_name,
    )
    if endpoint_images is None or endpoint_coordinates is None:
        raise ValueError("Trajectory explorer requires target or generated endpoint images.")

    explorer_config = config or ExplorerConfig(height=height)
    output_path = Path(output_path)
    frame = pd.DataFrame(rows)
    bundle = prepare_array_sprite_atlases(
        frame,
        endpoint_images,
        output_dir=output_path.parent / "assets" / "trajectory_atlases",
        image_shape=image_shape,
        image_value_range=image_value_range,
        tile_size=explorer_config.atlas_tile_size,
        max_atlas_size=explorer_config.atlas_size,
    )
    trajectory_labels = _trajectory_labels(
        trajectory=trajectory,
        target=target,
        target_labels=target_labels,
    )
    palette = _palette_with_trajectory_labels(bundle.palette, trajectory_labels)
    payload = {
        "points": _point_payload(bundle, endpoint_coordinates),
        "atlases": [atlas_data_url(path) for path in bundle.atlas_paths],
        "palette": {
            label: f"rgb({color[0]}, {color[1]}, {color[2]})"
            for label, color in palette.items()
        },
        "tileSize": bundle.tile_size,
        "atlasColumns": bundle.atlas_columns,
        "trajectory": np.round(trajectory, 5).tolist(),
        "trajectoryLabels": trajectory_labels,
        "options": {
            "pointSize": explorer_config.point_size,
            "hoverSize": explorer_config.hover_size,
            "previewMode": explorer_config.preview_mode,
            "targetAlpha": 0.28,
            "generatedAlpha": 0.9,
            "lineAlpha": 0.34,
        },
        "counts": {
            "targets": 0 if target is None else int(len(target)),
            "generated": 0 if generated is None else int(len(generated)),
            "trajectorySteps": int(trajectory.shape[0]),
            "trajectories": int(trajectory.shape[1]),
        },
    }
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _trajectory_explorer_template(payload_json, height=height, config=explorer_config),
        encoding="utf-8",
    )
    return {
        "endpoint_points": int(len(frame)),
        "target_endpoint_points": int(payload["counts"]["targets"]),
        "generated_endpoint_points": int(payload["counts"]["generated"]),
        "atlas_count": int(len(bundle.atlas_paths)),
    }


def _as_projected_trajectory(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Trajectory explorer requires shape (steps, paths, 3), got {array.shape}.")
    return array


def _as_projected_points(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] != 3:
        raise ValueError(f"Endpoint coordinates require shape (n, 3), got {array.shape}.")
    return array


def _endpoint_rows(
    *,
    target: np.ndarray | None,
    generated: np.ndarray | None,
    target_images: np.ndarray | None,
    generated_images: np.ndarray | None,
    target_labels: np.ndarray | None,
    generated_labels: np.ndarray | None,
    dataset_name: str,
) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None]:
    rows: list[dict[str, Any]] = []
    image_blocks: list[np.ndarray] = []
    coordinate_blocks: list[np.ndarray] = []

    if target is not None and target_images is not None and len(target) > 0:
        target_images = np.asarray(target_images)
        count = min(len(target), len(target_images))
        labels = _string_labels(target_labels, count=count, fallback="target")
        for index in range(count):
            rows.append(
                {
                    "row_id": len(rows),
                    "source_index": index,
                    "label": labels[index],
                    "dataset": dataset_name,
                    "kind": "target",
                    "label_source": "target label" if target_labels is not None else "none",
                }
            )
        image_blocks.append(target_images[:count])
        coordinate_blocks.append(target[:count])

    if generated is not None and generated_images is not None and len(generated) > 0:
        generated_images = np.asarray(generated_images)
        count = min(len(generated), len(generated_images))
        labels = _string_labels(generated_labels, count=count, fallback="generated")
        label_source = "predicted/inferred" if generated_labels is not None else "none"
        for index in range(count):
            rows.append(
                {
                    "row_id": len(rows),
                    "source_index": index,
                    "label": labels[index],
                    "dataset": dataset_name,
                    "kind": "generated",
                    "label_source": label_source,
                }
            )
        image_blocks.append(generated_images[:count])
        coordinate_blocks.append(generated[:count])

    if not image_blocks:
        return rows, None, None
    return rows, np.concatenate(image_blocks, axis=0), np.concatenate(coordinate_blocks, axis=0)


def _string_labels(
    labels: np.ndarray | None,
    *,
    count: int,
    fallback: str,
) -> list[str]:
    if labels is None:
        return [fallback] * count
    labels = np.asarray(labels)[:count]
    return [str(value.item() if isinstance(value, np.generic) else value) for value in labels]


def _trajectory_labels(
    *,
    trajectory: np.ndarray,
    target: np.ndarray | None,
    target_labels: np.ndarray | None,
) -> list[str]:
    if target is None or target_labels is None or len(target) == 0:
        return ["trajectory"] * trajectory.shape[1]
    labels = _nearest_projected_labels(
        query=trajectory[-1],
        reference=target,
        labels=target_labels,
    )
    return _string_labels(labels, count=trajectory.shape[1], fallback="trajectory")


def infer_generated_labels_from_target(
    *,
    generated: np.ndarray | None,
    target: np.ndarray | None,
    target_labels: np.ndarray | None,
) -> np.ndarray | None:
    """Assign generated endpoints the nearest target label in projected space."""

    if generated is None or target is None or target_labels is None:
        return None
    if len(generated) == 0 or len(target) == 0:
        return None
    return _nearest_projected_labels(
        query=np.asarray(generated, dtype=np.float32),
        reference=np.asarray(target, dtype=np.float32),
        labels=np.asarray(target_labels),
    )


def _nearest_projected_labels(
    *,
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors

        model = NearestNeighbors(n_neighbors=1, algorithm="auto")
        model.fit(reference)
        indices = model.kneighbors(query, return_distance=False)[:, 0]
    except ImportError:
        indices = _nearest_indices_numpy(query=query, reference=reference)
    return np.asarray(labels)[indices]


def _nearest_indices_numpy(
    *,
    query: np.ndarray,
    reference: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    reference = np.asarray(reference, dtype=np.float32)
    for start in range(0, len(query), chunk_size):
        chunk = np.asarray(query[start : start + chunk_size], dtype=np.float32)
        distances = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        chunks.append(np.argmin(distances, axis=1))
    return np.concatenate(chunks, axis=0)


def _palette_with_trajectory_labels(
    palette: dict[str, tuple[int, int, int]],
    trajectory_labels: list[str],
) -> dict[str, tuple[int, int, int]]:
    updated = dict(palette)
    for label in trajectory_labels:
        if label not in updated:
            updated[label] = LABEL_PALETTE[len(updated) % len(LABEL_PALETTE)]
    return updated


def _point_payload(
    bundle: AtlasBundle,
    coordinates: np.ndarray,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for position, row in bundle.frame.iterrows():
        points.append(
            {
                "rowId": int(row.get("row_id", position)),
                "sourceIndex": int(row.get("source_index", position)),
                "label": str(row.get("label", "")),
                "dataset": str(row.get("dataset", "")),
                "kind": str(row.get("kind", "")),
                "labelSource": str(row.get("label_source", "")),
                "atlas": int(row["atlas_index"]),
                "column": int(row["atlas_column"]),
                "row": int(row["atlas_row"]),
                "coordinates": [float(value) for value in coordinates[position]],
            }
        )
    return points


def _trajectory_explorer_template(
    payload_json: str,
    *,
    height: int,
    config: ExplorerConfig,
) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trajectory UMAP Explorer</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; overflow: hidden; background: #111; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #f2f2f2; }
  #app { display: grid; grid-template-columns: __SIDEBAR_WIDTH__px 1fr; height: __HEIGHT__px; background: #111; }
  #sidebar { background: #222; padding: 16px; display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .control { display: grid; grid-template-columns: 72px 1fr; align-items: center; gap: 10px; }
  label, .muted { color: #c8c8c8; font-size: 14px; }
  button { border: 1px solid #444; background: #1b1b1b; color: #ddd; cursor: pointer; }
  button:hover { background: #292929; }
  #play { height: 32px; border-radius: 2px; font-size: 13px; }
  input[type="range"] { width: 100%; accent-color: #d8d8d8; }
  .time-grid { display: grid; grid-template-columns: 58px 1fr; gap: 8px; align-items: center; }
  #time-readout { color: #aaa; font-size: 12px; font-variant-numeric: tabular-nums; text-align: right; }
  .class-menu { position: relative; min-width: 0; }
  .class-menu summary { height: 32px; display: flex; align-items: center; padding: 0 8px; background: #f3f3f3; color: #111; border-radius: 2px; cursor: pointer; list-style: none; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .class-menu summary::-webkit-details-marker { display: none; }
  .class-menu summary::after { content: "v"; margin-left: auto; padding-left: 8px; color: #555; }
  .class-options { position: absolute; z-index: 20; top: 36px; left: 0; right: 0; max-height: 250px; overflow-y: auto; padding: 6px; background: #f3f3f3; color: #111; border: 1px solid #bbb; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35); }
  .class-option { min-height: 28px; display: flex; align-items: center; gap: 7px; padding: 2px 4px; color: #111; font-size: 12px; cursor: pointer; }
  .class-option:hover { background: #ddd; }
  .class-option input { margin: 0; }
  .class-count { margin-left: auto; color: #666; font-variant-numeric: tabular-nums; }
  #preview-wrap { width: 100%; aspect-ratio: 1; background: #191919; display: grid; place-items: center; }
  #preview { width: 100%; height: 100%; image-rendering: pixelated; }
  #sample-info { min-height: 118px; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 10px; align-content: start; font-size: 12px; }
  #sample-label { grid-column: 1 / -1; font-size: 24px; font-weight: 650; }
  #sample-index { grid-column: 1 / -1; color: #a9a9a9; font-variant-numeric: tabular-nums; }
  .metric-key { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #bdbdbd; }
  .metric-value { color: #eee; font-variant-numeric: tabular-nums; text-align: right; }
  #legend { display: flex; flex-wrap: wrap; gap: 6px 10px; }
  .legend-item { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; color: #cfcfcf; }
  .swatch { width: 9px; height: 9px; }
  #sidebar-footer { margin-top: auto; color: #aaa; font-size: 12px; line-height: 1.45; }
  #main { position: relative; min-width: 0; overflow: hidden; background: #111; }
  #plot { position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }
  #plot.dragging { cursor: grabbing; }
  #status { position: absolute; right: 14px; bottom: 12px; color: #777; font-size: 12px; pointer-events: none; }
  #view-controls { position: absolute; right: 14px; top: 12px; display: grid; gap: 5px; }
  #view-controls button { width: 34px; height: 34px; font-size: 20px; }
  @media (max-width: 760px) {
    #app { grid-template-columns: 1fr; grid-template-rows: 1fr 190px; }
    #sidebar { grid-row: 2; display: grid; grid-template-columns: 120px minmax(150px, 220px) minmax(0, 1fr); grid-template-rows: auto auto 1fr; gap: 8px 10px; padding: 10px; overflow: hidden; }
    #preview-wrap { grid-column: 1; grid-row: 1 / span 3; }
    #sidebar > .control:nth-of-type(1) { grid-column: 2; grid-row: 1; }
    #sidebar > .control:nth-of-type(2) { grid-column: 2; grid-row: 2; }
    #sample-info { grid-column: 3; grid-row: 1 / span 3; overflow-y: auto; }
    #legend, #sidebar-footer { display: none; }
    #main { grid-row: 1; }
    .control { grid-template-columns: 1fr; gap: 4px; align-content: start; }
  }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <div class="control">
      <label for="time">Time</label>
      <div class="time-grid">
        <button id="play">Play</button>
        <input id="time" type="range" min="0" max="0" value="0" step="1">
        <div id="time-readout">t = 0 / 0</div>
      </div>
    </div>
    <div class="control">
      <span class="muted">Class</span>
      <details id="class-filter" class="class-menu">
        <summary id="class-filter-summary">All classes</summary>
        <div id="class-options" class="class-options"></div>
      </details>
    </div>
    <div id="preview-wrap"><canvas id="preview"></canvas></div>
    <div id="sample-info">
      <div class="muted">Label</div>
      <div id="sample-label">-</div>
      <div id="sample-index">Index: -</div>
    </div>
    <div id="legend"></div>
    <div id="sidebar-footer">
      Drag to rotate. Scroll to zoom. Hover to inspect endpoint images. Click to pin a sample.
    </div>
  </aside>
  <main id="main">
    <canvas id="plot"></canvas>
    <div id="view-controls">
      <button id="zoom-in" title="Zoom in" aria-label="Zoom in">+</button>
      <button id="zoom-out" title="Zoom out" aria-label="Zoom out">&minus;</button>
      <button id="reset" title="Reset view" aria-label="Reset view">&#8634;</button>
    </div>
    <div id="status"></div>
  </main>
</div>
<script>
const DATA = __PAYLOAD_JSON__;
const canvas = document.getElementById("plot");
const context = canvas.getContext("2d");
const preview = document.getElementById("preview");
const previewContext = preview.getContext("2d");
const previewTile = document.createElement("canvas");
previewTile.width = DATA.tileSize;
previewTile.height = DATA.tileSize;
const previewTileContext = previewTile.getContext("2d");
const main = document.getElementById("main");
const slider = document.getElementById("time");
const timeReadout = document.getElementById("time-readout");
const playButton = document.getElementById("play");
const classFilterSummary = document.getElementById("class-filter-summary");
const classOptions = document.getElementById("class-options");
const labelElement = document.getElementById("sample-label");
const indexElement = document.getElementById("sample-index");
const sampleInfo = document.getElementById("sample-info");
const statusElement = document.getElementById("status");
const resetButton = document.getElementById("reset");
const zoomInButton = document.getElementById("zoom-in");
const zoomOutButton = document.getElementById("zoom-out");
const atlasImages = [];
const steps = DATA.trajectory.length;
const pathCount = steps ? DATA.trajectory[0].length : 0;
let width = 1;
let height = 1;
let pixelRatio = Math.max(1, window.devicePixelRatio || 1);
let step = Math.max(0, steps - 1);
let yaw = -0.78;
let pitch = 0.38;
let zoom = 1.0;
let dragging = false;
let moved = false;
let dragX = 0;
let dragY = 0;
let hoverIndex = null;
let pinnedIndex = null;
let frameRequested = false;
let playing = false;
let timer = null;
let selectedLabels = new Set(DATA.points.map(point => point.label));
let visibleIndices = DATA.points.map((_, index) => index);
let bounds = computeBounds();
let center = bounds.center;
let span = bounds.span;

slider.max = Math.max(0, steps - 1);
slider.value = String(step);

for (const [label, color] of Object.entries(DATA.palette)) {
  const item = document.createElement("span");
  item.className = "legend-item";
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = color;
  const text = document.createElement("span");
  text.textContent = label;
  item.append(swatch, text);
  document.getElementById("legend").appendChild(item);
}

function initializeClassFilter() {
  const counts = new Map();
  for (const point of DATA.points) {
    counts.set(point.label, (counts.get(point.label) || 0) + 1);
  }
  const labels = Array.from(counts.keys()).sort((left, right) =>
    left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" })
  );
  selectedLabels = new Set(labels);
  const allInput = document.createElement("input");
  allInput.type = "checkbox";
  allInput.checked = true;
  classOptions.appendChild(createClassOption(allInput, "All classes", DATA.points.length));
  const classInputs = labels.map(label => {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = label;
    input.checked = true;
    classOptions.appendChild(createClassOption(input, label || "(empty)", counts.get(label)));
    input.addEventListener("change", () => {
      allInput.checked = classInputs.every(value => value.checked);
      syncClassFilter(classInputs);
    });
    return input;
  });
  allInput.addEventListener("change", () => {
    for (const input of classInputs) input.checked = allInput.checked;
    syncClassFilter(classInputs);
  });
}

function createClassOption(input, text, count) {
  const row = document.createElement("label");
  row.className = "class-option";
  const name = document.createElement("span");
  name.textContent = text;
  const total = document.createElement("span");
  total.className = "class-count";
  total.textContent = count.toLocaleString();
  row.append(input, name, total);
  return row;
}

function syncClassFilter(classInputs) {
  selectedLabels = new Set(
    classInputs.filter(input => input.checked).map(input => input.value)
  );
  visibleIndices = DATA.points
    .map((point, index) => selectedLabels.has(point.label) ? index : -1)
    .filter(index => index >= 0);
  const selectedCount = selectedLabels.size;
  const totalCount = classInputs.length;
  classFilterSummary.textContent = selectedCount === totalCount
    ? "All classes"
    : selectedCount === 0
      ? "No classes"
      : selectedCount === 1
        ? Array.from(selectedLabels)[0] || "(empty)"
        : `${selectedCount} classes`;
  hoverIndex = null;
  pinnedIndex = null;
  showPoint(null);
  requestDraw();
}

function loadAtlases() {
  return Promise.all(DATA.atlases.map(source => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => { atlasImages.push(image); resolve(); };
    image.onerror = reject;
    image.src = source;
  })));
}

function computeBounds() {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const visit = (point) => {
    for (let i = 0; i < 3; i++) {
      if (point[i] < min[i]) min[i] = point[i];
      if (point[i] > max[i]) max[i] = point[i];
    }
  };
  for (const point of DATA.points) visit(point.coordinates);
  for (const timeSlice of DATA.trajectory) {
    for (const point of timeSlice) visit(point);
  }
  for (let i = 0; i < 3; i++) {
    if (!Number.isFinite(min[i]) || !Number.isFinite(max[i])) {
      min[i] = -1;
      max[i] = 1;
    }
  }
  const center = [
    0.5 * (min[0] + max[0]),
    0.5 * (min[1] + max[1]),
    0.5 * (min[2] + max[2])
  ];
  const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 1e-6);
  return { min, max, center, span };
}

function resize() {
  const rect = main.getBoundingClientRect();
  width = Math.max(1, rect.width);
  height = Math.max(1, rect.height);
  pixelRatio = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  preview.width = Math.round(preview.clientWidth * pixelRatio);
  preview.height = Math.round(preview.clientHeight * pixelRatio);
  previewContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  requestDraw();
}

function project(point) {
  const x0 = (point[0] - center[0]) / span;
  const y0 = (point[1] - center[1]) / span;
  const z0 = (point[2] - center[2]) / span;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x0 + sy * z0;
  const z1 = -sy * x0 + cy * z0;
  const y1 = cp * y0 - sp * z1;
  const z2 = sp * y0 + cp * z1;
  const scale = Math.min(width, height) * 1.25 * zoom;
  return {
    x: width * 0.5 + x1 * scale,
    y: height * 0.52 - y1 * scale,
    z: z2
  };
}

function endpointSize(point) {
  return point.kind === "target" ? Math.max(5, DATA.options.pointSize * 0.72) : DATA.options.pointSize;
}

function draw() {
  frameRequested = false;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111";
  context.fillRect(0, 0, width, height);
  drawTrajectory();
  drawEndpoints();
  const selectedIndex = pinnedIndex !== null ? pinnedIndex : hoverIndex;
  if (selectedIndex !== null && selectedLabels.has(DATA.points[selectedIndex].label)) {
    drawHighlight(selectedIndex);
  }
  timeReadout.textContent = `t = ${step} / ${Math.max(0, steps - 1)}`;
  statusElement.textContent = `${visibleIndices.length.toLocaleString()} endpoint images - ${pathCount.toLocaleString()} trajectories`;
}

function drawTrajectory() {
  if (!steps) return;
  context.lineWidth = 0.8;
  for (let i = 0; i < pathCount; i++) {
    const label = DATA.trajectoryLabels[i] || "trajectory";
    context.strokeStyle = DATA.palette[label] || "#8b949e";
    context.globalAlpha = DATA.options.lineAlpha;
    context.beginPath();
    for (let t = 0; t <= step; t++) {
      const p = project(DATA.trajectory[t][i]);
      if (t === 0) context.moveTo(p.x, p.y);
      else context.lineTo(p.x, p.y);
    }
    context.stroke();
  }
  const current = DATA.trajectory[step];
  const sorted = current.map((point, index) => {
    const projected = project(point);
    return { point, index, projected };
  }).sort((a, b) => a.projected.z - b.projected.z);
  for (const entry of sorted) {
    const label = DATA.trajectoryLabels[entry.index] || "trajectory";
    context.globalAlpha = 0.92;
    context.fillStyle = DATA.palette[label] || "#ef4444";
    context.strokeStyle = "#111";
    context.lineWidth = 1.2;
    context.beginPath();
    context.arc(entry.projected.x, entry.projected.y, 4.2, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }
  context.globalAlpha = 1;
}

function drawEndpoints() {
  const sorted = visibleIndices.map(index => {
    const point = DATA.points[index];
    const projected = project(point.coordinates);
    return { index, point, projected };
  }).sort((a, b) => a.projected.z - b.projected.z);
  for (const entry of sorted) {
    const size = endpointSize(entry.point);
    if (entry.projected.x < -size || entry.projected.x > width + size || entry.projected.y < -size || entry.projected.y > height + size) continue;
    context.globalAlpha = entry.point.kind === "target" ? DATA.options.targetAlpha : DATA.options.generatedAlpha;
    drawTile(context, entry.point, entry.projected.x - size / 2, entry.projected.y - size / 2, size);
  }
  context.globalAlpha = 1;
}

function drawHighlight(index) {
  const point = DATA.points[index];
  const projected = project(point.coordinates);
  const size = Math.max(DATA.options.hoverSize, endpointSize(point) * 2.2);
  context.globalAlpha = 1;
  context.strokeStyle = "#fff";
  context.lineWidth = 2;
  context.strokeRect(projected.x - size / 2 - 2, projected.y - size / 2 - 2, size + 4, size + 4);
  drawTile(context, point, projected.x - size / 2, projected.y - size / 2, size);
}

function drawTile(targetContext, point, x, y, size) {
  const sourceX = point.column * DATA.tileSize;
  const sourceY = point.row * DATA.tileSize;
  targetContext.imageSmoothingEnabled = false;
  targetContext.drawImage(
    atlasImages[point.atlas],
    sourceX,
    sourceY,
    DATA.tileSize,
    DATA.tileSize,
    x,
    y,
    size,
    size
  );
}

function drawPreviewTile(point, x, y, size) {
  const grayscaleDataset = ["mnist", "fashion_mnist"].includes(point.dataset.toLowerCase());
  if (DATA.options.previewMode !== "original" || !grayscaleDataset) {
    drawTile(previewContext, point, x, y, size);
    return;
  }
  const sourceX = point.column * DATA.tileSize;
  const sourceY = point.row * DATA.tileSize;
  previewTileContext.clearRect(0, 0, DATA.tileSize, DATA.tileSize);
  previewTileContext.drawImage(
    atlasImages[point.atlas],
    sourceX,
    sourceY,
    DATA.tileSize,
    DATA.tileSize,
    0,
    0,
    DATA.tileSize,
    DATA.tileSize
  );
  const image = previewTileContext.getImageData(0, 0, DATA.tileSize, DATA.tileSize);
  for (let offset = 0; offset < image.data.length; offset += 4) {
    image.data[offset] = 255;
    image.data[offset + 1] = 255;
    image.data[offset + 2] = 255;
  }
  previewTileContext.putImageData(image, 0, 0);
  previewContext.imageSmoothingEnabled = false;
  previewContext.drawImage(previewTile, x, y, size, size);
}

function nearestPoint(mouseX, mouseY) {
  const threshold = Math.max(12, DATA.options.pointSize * 1.2);
  let best = null;
  let bestDistance = threshold * threshold;
  for (const index of visibleIndices) {
    const screen = project(DATA.points[index].coordinates);
    const dx = screen.x - mouseX;
    const dy = screen.y - mouseY;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best;
}

function showPoint(index) {
  previewContext.clearRect(0, 0, preview.clientWidth, preview.clientHeight);
  previewContext.fillStyle = "#191919";
  previewContext.fillRect(0, 0, preview.clientWidth, preview.clientHeight);
  labelElement.textContent = "-";
  indexElement.textContent = "Index: -";
  clearMetrics();
  if (index === null) return;
  const point = DATA.points[index];
  const margin = 12;
  drawPreviewTile(
    point,
    margin,
    margin,
    preview.clientWidth - margin * 2,
    preview.clientHeight - margin * 2
  );
  labelElement.textContent = point.label || "-";
  indexElement.textContent = `${point.kind} - Index: ${point.sourceIndex} - Row: ${point.rowId}`;
  appendMetric("UMAP X", point.coordinates[0]);
  appendMetric("UMAP Y", point.coordinates[1]);
  appendMetric("UMAP Z", point.coordinates[2]);
  appendMetric("Label source", point.labelSource || "-");
}

function clearMetrics() {
  for (const element of Array.from(sampleInfo.querySelectorAll(".metric-key, .metric-value"))) {
    element.remove();
  }
}

function appendMetric(name, value) {
  const key = document.createElement("span");
  key.className = "metric-key";
  key.title = name;
  key.textContent = name;
  const metric = document.createElement("span");
  metric.className = "metric-value";
  if (typeof value === "number") {
    metric.textContent = value.toFixed(3);
  } else {
    metric.textContent = String(value);
  }
  sampleInfo.append(key, metric);
}

function requestDraw() {
  if (!frameRequested) {
    frameRequested = true;
    requestAnimationFrame(draw);
  }
}

function setStep(value) {
  step = Math.max(0, Math.min(steps - 1, Number(value)));
  slider.value = String(step);
  requestDraw();
}

function startPlayback() {
  if (playing) return;
  playing = true;
  playButton.textContent = "Pause";
  timer = window.setInterval(() => {
    if (step >= steps - 1) setStep(0);
    else setStep(step + 1);
  }, 140);
}

function stopPlayback() {
  playing = false;
  playButton.textContent = "Play";
  if (timer !== null) window.clearInterval(timer);
  timer = null;
}

function resetView() {
  yaw = -0.78;
  pitch = 0.38;
  zoom = 1.0;
  requestDraw();
}

slider.addEventListener("input", () => {
  stopPlayback();
  setStep(slider.value);
});
playButton.addEventListener("click", () => {
  if (playing) stopPlayback();
  else startPlayback();
});
canvas.addEventListener("pointerdown", event => {
  dragging = true;
  moved = false;
  dragX = event.clientX;
  dragY = event.clientY;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", event => {
  const rect = canvas.getBoundingClientRect();
  if (dragging) {
    const dx = event.clientX - dragX;
    const dy = event.clientY - dragY;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    yaw += dx * 0.008;
    pitch += dy * 0.008;
    pitch = Math.max(-1.45, Math.min(1.45, pitch));
    dragX = event.clientX;
    dragY = event.clientY;
    requestDraw();
    return;
  }
  if (pinnedIndex !== null) return;
  hoverIndex = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
  showPoint(hoverIndex);
  requestDraw();
});
canvas.addEventListener("pointerup", event => {
  dragging = false;
  canvas.classList.remove("dragging");
  if (!moved) {
    const rect = canvas.getBoundingClientRect();
    const selected = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
    pinnedIndex = pinnedIndex === selected ? null : selected;
    hoverIndex = selected;
    showPoint(pinnedIndex !== null ? pinnedIndex : hoverIndex);
  }
  requestDraw();
});
canvas.addEventListener("pointerleave", () => {
  dragging = false;
  canvas.classList.remove("dragging");
  if (pinnedIndex === null) {
    hoverIndex = null;
    showPoint(null);
    requestDraw();
  }
});
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  zoom *= Math.exp(-event.deltaY * 0.001);
  zoom = Math.max(0.25, Math.min(6.0, zoom));
  requestDraw();
}, { passive: false });
resetButton.addEventListener("click", resetView);
zoomInButton.addEventListener("click", () => {
  zoom = Math.min(6.0, zoom * 1.25);
  requestDraw();
});
zoomOutButton.addEventListener("click", () => {
  zoom = Math.max(0.25, zoom / 1.25);
  requestDraw();
});
window.addEventListener("resize", resize);
initializeClassFilter();
loadAtlases().then(() => {
  resize();
  showPoint(null);
});
</script>
</body>
</html>
"""
    return (
        template.replace("__PAYLOAD_JSON__", payload_json)
        .replace("__HEIGHT__", str(int(height)))
        .replace("__SIDEBAR_WIDTH__", str(int(config.sidebar_width)))
    )
