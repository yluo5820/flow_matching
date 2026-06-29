"""Explorer-style HTML renderer for projected sampling trajectories."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fm_lab.image_diagnostics.canvas_explorer import prepare_array_sprite_atlases
from fm_lab.image_diagnostics.config import ExplorerConfig
from fm_lab.image_diagnostics.explorer_payload import (
    atlas_data_url,
    atlas_point_payload,
    palette_payload,
)
from fm_lab.image_diagnostics.explorer_viewer import (
    ExplorerDocument,
    build_explorer_document,
    class_filter_html,
    shared_explorer_script,
)
from fm_lab.image_diagnostics.palette import LABEL_PALETTE


def write_trajectory_explorer_html(
    output_path: str | Path,
    *,
    trajectory: np.ndarray,
    target: np.ndarray | None,
    generated: np.ndarray | None,
    target_images: np.ndarray | None,
    generated_images: np.ndarray | None,
    trajectory_images: np.ndarray | None = None,
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
    trajectory_labels = _trajectory_labels(
        trajectory=trajectory,
        target=target,
        target_labels=target_labels,
    )
    (
        rows,
        atlas_images,
        endpoint_coordinates,
        n_endpoint_rows,
        n_trajectory_preview_rows,
    ) = _atlas_rows(
        target=target,
        generated=generated,
        target_images=target_images,
        generated_images=generated_images,
        trajectory_images=trajectory_images,
        target_labels=target_labels,
        generated_labels=generated_labels,
        trajectory_labels=trajectory_labels,
        dataset_name=dataset_name,
    )
    if atlas_images is None or endpoint_coordinates is None:
        raise ValueError("Trajectory explorer requires target or generated endpoint images.")

    explorer_config = config or ExplorerConfig(height=height)
    output_path = Path(output_path)
    frame = pd.DataFrame(rows)
    bundle = prepare_array_sprite_atlases(
        frame,
        atlas_images,
        output_dir=output_path.parent / "assets" / "trajectory_atlases",
        image_shape=image_shape,
        image_value_range=image_value_range,
        tile_size=explorer_config.atlas_tile_size,
        max_atlas_size=explorer_config.atlas_size,
    )
    palette = _palette_with_trajectory_labels(bundle.palette, trajectory_labels)
    payload = {
        "points": atlas_point_payload(
            bundle.frame,
            coordinates=endpoint_coordinates,
            start=0,
            count=n_endpoint_rows,
        ),
        "trajectoryPreviews": atlas_point_payload(
            bundle.frame,
            coordinates=trajectory[-1],
            start=n_endpoint_rows,
            count=n_trajectory_preview_rows,
        ),
        "atlases": [atlas_data_url(path) for path in bundle.atlas_paths],
        "palette": palette_payload(palette),
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
            "drawThumbnailsDefault": False,
        },
        "counts": {
            "targets": 0 if target is None else int(len(target)),
            "generated": 0 if generated is None else int(len(generated)),
            "trajectorySteps": int(trajectory.shape[0]),
            "trajectories": int(trajectory.shape[1]),
            "trajectoryPreviews": int(n_trajectory_preview_rows),
        },
    }
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _trajectory_explorer_template(payload_json, height=height, config=explorer_config),
        encoding="utf-8",
    )
    return {
        "endpoint_points": int(n_endpoint_rows),
        "target_endpoint_points": int(payload["counts"]["targets"]),
        "generated_endpoint_points": int(payload["counts"]["generated"]),
        "trajectory_preview_points": int(n_trajectory_preview_rows),
        "atlas_points": int(len(frame)),
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


def _atlas_rows(
    *,
    target: np.ndarray | None,
    generated: np.ndarray | None,
    target_images: np.ndarray | None,
    generated_images: np.ndarray | None,
    trajectory_images: np.ndarray | None,
    target_labels: np.ndarray | None,
    generated_labels: np.ndarray | None,
    trajectory_labels: list[str],
    dataset_name: str,
) -> tuple[list[dict[str, Any]], np.ndarray | None, np.ndarray | None, int, int]:
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

    endpoint_rows = len(rows)
    if trajectory_images is not None:
        trajectory_images = np.asarray(trajectory_images)
        count = min(len(trajectory_images), len(trajectory_labels))
        labels = trajectory_labels[:count]
        for index in range(count):
            rows.append(
                {
                    "row_id": len(rows),
                    "source_index": index,
                    "label": labels[index],
                    "dataset": dataset_name,
                    "kind": "trajectory",
                    "label_source": "nearest target final"
                    if labels[index] != "trajectory"
                    else "none",
                }
            )
        image_blocks.append(trajectory_images[:count])

    if not image_blocks:
        return rows, None, None, 0, 0
    endpoint_coordinates = (
        np.concatenate(coordinate_blocks, axis=0)
        if coordinate_blocks
        else np.empty((0, 3), dtype=np.float32)
    )
    return (
        rows,
        np.concatenate(image_blocks, axis=0),
        endpoint_coordinates,
        endpoint_rows,
        len(rows) - endpoint_rows,
    )


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


def _trajectory_explorer_template(
    payload_json: str,
    *,
    height: int,
    config: ExplorerConfig,
) -> str:
    controls_html = (
        '<div class="control">\n'
        '      <label for="time">Time</label>\n'
        '      <div class="time-grid">\n'
        '        <button id="play">Play</button>\n'
        '        <input id="time" type="range" min="0" max="0" value="0" step="1">\n'
        '        <div id="time-readout">t = 0 / 0</div>\n'
        '      </div>\n'
        '    </div>\n'
        + class_filter_html()
        + '\n'
        '<div class="control">\n'
        '      <span class="muted">View</span>\n'
        '      <label class="toggle-row">\n'
        '        <input id="show-thumbnails" type="checkbox">\n'
        '        <span>Thumbnails</span>\n'
        '      </label>\n'
        '    </div>'
    )
    sample_info_html = (
        '<div class="muted">Label</div>\n'
        '      <div id="sample-label">-</div>\n'
        '      <div id="sample-index">Index: -</div>\n'
        '      <div id="metrics"></div>'
    )
    footer_html = (
        'Drag to rotate. Scroll to zoom. Hover to inspect endpoint images. '
        'Click to pin a sample.'
    )
    extra_css = (
        '  #play { height: 32px; border-radius: 2px; font-size: 13px; }\n'
        '  input[type="range"] { width: 100%; accent-color: #d8d8d8; }\n'
        '  .time-grid { display: grid; grid-template-columns: 58px 1fr; gap: 8px; align-items: center; }\n'
        '  #time-readout { color: #aaa; font-size: 12px; font-variant-numeric: tabular-nums; text-align: right; }\n'
        '  #sample-info { min-height: 118px; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 10px; font-size: 12px; }\n'
        '  #sample-label { grid-column: 1 / -1; }\n'
        '  #sample-index { grid-column: 1 / -1; }\n'
        '  .toggle-row { display: inline-flex; align-items: center; gap: 7px; color: #ddd; font-size: 13px; }\n'
        '  .toggle-row input { margin: 0; }\n'
        '  @media (max-width: 760px) {\n'
        '    #sidebar > .control:nth-of-type(3) { grid-column: 2; grid-row: 3; }\n'
        '    #sample-label { font-size: 20px; }\n'
        '  }\n'
    )
    return build_explorer_document(
        ExplorerDocument(
            title="Trajectory UMAP Explorer",
            controls_html=controls_html,
            sample_info_html=sample_info_html,
            footer_html=footer_html,
            script=_trajectory_script(payload_json),
            height=height,
            config=config,
            extra_css=extra_css,
            control_label_width=72,
            mobile_height=190,
            mobile_columns="120px minmax(150px, 220px) minmax(0, 1fr)",
            mobile_rows="auto auto 1fr",
            preview_row_span=3,
        )
    )


def _trajectory_script(payload_json: str) -> str:
    script = r'''__SHARED_SCRIPT__
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
const thumbnailToggle = document.getElementById("show-thumbnails");
const classFilterSummary = document.getElementById("class-filter-summary");
const classOptions = document.getElementById("class-options");
const labelElement = document.getElementById("sample-label");
const indexElement = document.getElementById("sample-index");
const metricsElement = document.getElementById("metrics");
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
let showThumbnails = Boolean(DATA.options.drawThumbnailsDefault);
let selectedLabels = new Set(DATA.points.map(point => point.label));
let visibleIndices = DATA.points.map((_, index) => index);
let bounds = computeBounds();
let center = bounds.center;
let span = bounds.span;

slider.max = Math.max(0, steps - 1);
slider.value = String(step);
thumbnailToggle.checked = showThumbnails;
populateLegend(DATA.palette);

function onClassFilterChange() {
  hoverIndex = null;
  pinnedIndex = null;
  showPoint(null);
  requestDraw();
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
  resizeExplorerCanvases(requestDraw);
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

function isTrajectoryLabelVisible(label) {
  return DATA.points.some(point => point.label === label) ? selectedLabels.has(label) : true;
}

function draw() {
  frameRequested = false;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111";
  context.fillRect(0, 0, width, height);
  drawTrajectory();
  drawEndpoints();
  const selectedIndex = pinnedIndex !== null ? pinnedIndex : hoverIndex;
  if (selectedIndex !== null) {
    drawSelectionHighlight(selectedIndex);
  }
  timeReadout.textContent = `t = ${step} / ${Math.max(0, steps - 1)}`;
  statusElement.textContent = `${visibleIndices.length.toLocaleString()} endpoints - ${pathCount.toLocaleString()} trajectories`;
}

function drawTrajectory() {
  if (!steps) return;
  context.lineWidth = 0.8;
  for (let i = 0; i < pathCount; i++) {
    const label = DATA.trajectoryLabels[i] || "trajectory";
    if (!isTrajectoryLabelVisible(label)) continue;
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
    if (!isTrajectoryLabelVisible(label)) continue;
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
    if (showThumbnails) {
      drawTile(context, entry.point, entry.projected.x - size / 2, entry.projected.y - size / 2, size);
    } else {
      drawCirclePoint(
        entry.projected.x,
        entry.projected.y,
        size * 0.42,
        DATA.palette[entry.point.label] || "#d4d4d4",
        entry.point.kind === "target" ? DATA.options.targetAlpha : DATA.options.generatedAlpha
      );
    }
  }
  context.globalAlpha = 1;
}

function drawCirclePoint(x, y, radius, color, alpha) {
  context.globalAlpha = alpha;
  context.fillStyle = color;
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fill();
  context.globalAlpha = 1;
}

function drawSelectionHighlight(selection) {
  const parsed = parseSelection(selection);
  if (parsed.kind === "point") drawEndpointHighlight(parsed.index);
  else drawTrajectoryHighlight(parsed.index);
}

function drawEndpointHighlight(index) {
  const point = DATA.points[index];
  if (!selectedLabels.has(point.label)) return;
  const projected = project(point.coordinates);
  const size = Math.max(DATA.options.hoverSize, endpointSize(point) * 2.2);
  context.globalAlpha = 1;
  context.strokeStyle = "#fff";
  context.lineWidth = 2;
  context.strokeRect(projected.x - size / 2 - 2, projected.y - size / 2 - 2, size + 4, size + 4);
  if (showThumbnails) {
    drawTile(context, point, projected.x - size / 2, projected.y - size / 2, size);
  } else {
    drawCirclePoint(projected.x, projected.y, Math.max(5, size * 0.18), DATA.palette[point.label] || "#fff", 1);
  }
}

function drawTrajectoryHighlight(index) {
  const label = DATA.trajectoryLabels[index] || "trajectory";
  if (!isTrajectoryLabelVisible(label)) return;
  const projected = project(DATA.trajectory[step][index]);
  context.globalAlpha = 1;
  context.strokeStyle = "#fff";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(projected.x, projected.y, 8, 0, Math.PI * 2);
  context.stroke();
}

function nearestPoint(mouseX, mouseY) {
  const trajectorySelection = nearestTrajectoryPoint(mouseX, mouseY);
  if (trajectorySelection !== null) return trajectorySelection;
  const endpointIndex = nearestVisiblePoint(
    mouseX,
    mouseY,
    Math.max(12, DATA.options.pointSize * 1.2),
    index => project(DATA.points[index].coordinates)
  );
  return endpointIndex === null ? null : pointSelection(endpointIndex);
}

function nearestTrajectoryPoint(mouseX, mouseY) {
  const threshold = Math.max(12, DATA.options.pointSize * 1.25);
  let best = null;
  let bestDistance = threshold * threshold;
  if (!steps) return null;
  for (let index = 0; index < pathCount; index++) {
    const label = DATA.trajectoryLabels[index] || "trajectory";
    if (!isTrajectoryLabelVisible(label)) continue;
    const screen = project(DATA.trajectory[step][index]);
    const dx = screen.x - mouseX;
    const dy = screen.y - mouseY;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best === null ? null : trajectorySelection(best);
}

function pointSelection(index) {
  return `p:${index}`;
}

function trajectorySelection(index) {
  return `t:${index}`;
}

function parseSelection(selection) {
  const [kind, rawIndex] = String(selection).split(":");
  return {
    kind: kind === "t" ? "trajectory" : "point",
    index: Number(rawIndex),
  };
}

function showPoint(index) {
  clearPreview();
  metricsElement.replaceChildren();
  labelElement.textContent = "-";
  indexElement.textContent = "Index: -";
  if (index === null) return;
  const parsed = parseSelection(index);
  if (parsed.kind === "trajectory") {
    showTrajectoryPoint(parsed.index);
    return;
  }
  const point = DATA.points[parsed.index];
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
  appendMetric(metricsElement, "UMAP X", point.coordinates[0]);
  appendMetric(metricsElement, "UMAP Y", point.coordinates[1]);
  appendMetric(metricsElement, "UMAP Z", point.coordinates[2]);
  appendMetric(metricsElement, "Label source", point.labelSource || "-");
}

function showTrajectoryPoint(index) {
  const previewPoint = DATA.trajectoryPreviews[index];
  const label = DATA.trajectoryLabels[index] || "trajectory";
  const current = DATA.trajectory[step][index];
  const final = DATA.trajectory[DATA.trajectory.length - 1][index];
  const margin = 12;
  if (previewPoint) {
    drawPreviewTile(
      previewPoint,
      margin,
      margin,
      preview.clientWidth - margin * 2,
      preview.clientHeight - margin * 2
    );
  }
  labelElement.textContent = label || "-";
  indexElement.textContent = `trajectory - Index: ${index} - final image`;
  appendMetric(metricsElement, "Current UMAP X", current[0]);
  appendMetric(metricsElement, "Current UMAP Y", current[1]);
  appendMetric(metricsElement, "Current UMAP Z", current[2]);
  appendMetric(metricsElement, "Final UMAP X", final[0]);
  appendMetric(metricsElement, "Final UMAP Y", final[1]);
  appendMetric(metricsElement, "Final UMAP Z", final[2]);
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
thumbnailToggle.addEventListener("change", () => {
  showThumbnails = thumbnailToggle.checked;
  requestDraw();
});
installCanvasPointerHandlers((_event, dx, dy) => {
  yaw += dx * 0.008;
  pitch += dy * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch));
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
initializeClassFilter(DATA.points, onClassFilterChange);
loadAtlases(DATA.atlases).then(() => {
  resize();
  showPoint(null);
});
'''
    return script.replace("__SHARED_SCRIPT__", shared_explorer_script()).replace(
        "__PAYLOAD_JSON__",
        payload_json,
    )
