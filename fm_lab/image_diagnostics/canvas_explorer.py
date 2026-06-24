"""Thumbnail-atlas canvas renderer for projected datasets."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from fm_lab.image_diagnostics.config import ExplorerConfig
from fm_lab.image_diagnostics.palette import LABEL_PALETTE


@dataclass(frozen=True)
class AtlasBundle:
    frame: pd.DataFrame
    atlas_paths: list[Path]
    palette: dict[str, tuple[int, int, int]]
    tile_size: int
    atlas_columns: int


def render_thumbnail_canvas(
    frame: pd.DataFrame,
    *,
    data_path: str | Path,
    config: ExplorerConfig | None = None,
    projection_names: dict[str, str] | None = None,
) -> None:
    """Render the reference-style thumbnail projection inside Streamlit."""

    import streamlit as st

    explorer_config = config or ExplorerConfig()
    output_dir = Path(data_path).resolve().parent.parent
    bundle = prepare_sprite_atlases(
        frame,
        output_dir=output_dir / "assets" / "atlases",
        tile_size=explorer_config.atlas_tile_size,
        max_atlas_size=explorer_config.atlas_size,
    )
    html = build_canvas_html(
        bundle,
        height=explorer_config.height,
        config=explorer_config,
        projection_names=projection_names,
    )
    st.iframe(
        html,
        height=explorer_config.height,
        width="stretch",
        tab_index=0,
    )


def prepare_sprite_atlases(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    tile_size: int = 28,
    max_atlas_size: int = 2048,
) -> AtlasBundle:
    """Pack sample previews into colored PNG atlases and return tile locations."""

    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    prepacked = _prepacked_atlas_bundle(frame)
    if prepacked is not None:
        return prepacked
    atlas_columns = max(1, max_atlas_size // tile_size)
    atlas_capacity = atlas_columns * atlas_columns
    prepared = frame.reset_index(drop=True).copy()
    labels = sorted(
        {str(value) for value in prepared.get("label", pd.Series(dtype=str))},
        key=_natural_sort_key,
    )
    palette = {
        label: LABEL_PALETTE[index % len(LABEL_PALETTE)]
        for index, label in enumerate(labels)
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    digest = _atlas_digest(
        prepared,
        tile_size=tile_size,
        atlas_size=max_atlas_size,
    )
    atlas_count = max(1, math.ceil(len(prepared) / atlas_capacity))
    atlas_paths = [
        output_path / f"atlas_{digest[:12]}_{index:02d}.png"
        for index in range(atlas_count)
    ]

    if not all(path.exists() for path in atlas_paths):
        _remove_old_atlases(output_path, keep=set(atlas_paths))
        _build_atlases(
            prepared,
            atlas_paths=atlas_paths,
            palette=palette,
            tile_size=tile_size,
            atlas_columns=atlas_columns,
            atlas_capacity=atlas_capacity,
            atlas_size=max_atlas_size,
        )

    positions = np.arange(len(prepared), dtype=int)
    prepared["atlas_index"] = positions // atlas_capacity
    local_positions = positions % atlas_capacity
    prepared["atlas_column"] = local_positions % atlas_columns
    prepared["atlas_row"] = local_positions // atlas_columns
    return AtlasBundle(
        frame=prepared,
        atlas_paths=atlas_paths,
        palette=palette,
        tile_size=tile_size,
        atlas_columns=atlas_columns,
    )


def build_canvas_html(
    bundle: AtlasBundle,
    *,
    height: int,
    config: ExplorerConfig | None = None,
    projection_names: dict[str, str] | None = None,
) -> str:
    """Build the complete HTML document for the canvas explorer."""

    explorer_config = config or ExplorerConfig(height=height)
    projections = _projection_columns(bundle.frame, projection_names=projection_names)
    if not projections:
        raise ValueError("Explorer data contains no UMAP, PCA, or t-SNE coordinates.")
    points = _point_payload(bundle.frame, projections)
    atlases = [
        f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        for path in bundle.atlas_paths
    ]
    payload = {
        "points": points,
        "atlases": atlases,
        "projections": list(projections),
        "projectionDiagnostics": _projection_diagnostics_payload(
            bundle.frame,
            projections,
        ),
        "palette": {
            label: f"rgb({color[0]}, {color[1]}, {color[2]})"
            for label, color in bundle.palette.items()
        },
        "tileSize": bundle.tile_size,
        "atlasColumns": bundle.atlas_columns,
        "options": {
            "pointSize": explorer_config.point_size,
            "hoverSize": explorer_config.hover_size,
            "transitionMs": explorer_config.transition_ms,
            "transitionEasing": explorer_config.transition_easing,
            "scalePointSizeWithZoom": explorer_config.scale_point_size_with_zoom,
            "previewMode": explorer_config.preview_mode,
        },
    }
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return _html_template(
        payload_json,
        height=height,
        config=explorer_config,
    )


def _build_atlases(
    frame: pd.DataFrame,
    *,
    atlas_paths: list[Path],
    palette: dict[str, tuple[int, int, int]],
    tile_size: int,
    atlas_columns: int,
    atlas_capacity: int,
    atlas_size: int,
) -> None:
    atlases = [
        Image.new(
            "RGBA",
            (atlas_size, atlas_size),
            (0, 0, 0, 0),
        )
        for _ in atlas_paths
    ]
    for position, row in frame.iterrows():
        image = _load_preview(row.get("image_path"), tile_size)
        label = str(row.get("label", ""))
        if str(row.get("dataset", "")).lower() == "mnist":
            image = _tint_grayscale(image, palette.get(label, (255, 255, 255)))
        atlas_index = position // atlas_capacity
        local_position = position % atlas_capacity
        column = local_position % atlas_columns
        row_index = local_position // atlas_columns
        atlases[atlas_index].alpha_composite(
            image,
            (column * tile_size, row_index * tile_size),
        )
    for atlas, path in zip(atlases, atlas_paths, strict=False):
        atlas.save(path, optimize=True)


def _load_preview(path_value: Any, tile_size: int) -> Image.Image:
    path = Path(str(path_value)) if str(path_value).strip() else None
    if path is None or not path.is_file():
        return Image.new("RGBA", (tile_size, tile_size), (255, 255, 255, 180))
    with Image.open(path) as source:
        image = source.convert("RGBA")
        image.thumbnail((tile_size, tile_size), Image.Resampling.NEAREST)
        output = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        x = (tile_size - image.width) // 2
        y = (tile_size - image.height) // 2
        output.alpha_composite(image, (x, y))
        return output


def _tint_grayscale(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    rgba = np.zeros((*grayscale.shape, 4), dtype=np.uint8)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = grayscale
    return Image.fromarray(rgba, mode="RGBA")


def _projection_columns(
    frame: pd.DataFrame,
    *,
    projection_names: dict[str, str] | None = None,
) -> dict[str, tuple[str, str]]:
    projections: dict[str, tuple[str, str]] = {}
    columns = set(frame.columns)
    x_columns = [column for column in frame.columns if str(column).endswith("_x")]
    for x_column in x_columns:
        key = str(x_column)[:-2]
        y_column = f"{key}_y"
        if y_column not in columns:
            continue
        display_name = (
            projection_names.get(key, _projection_display_name(key))
            if projection_names
            else _projection_display_name(key)
        )
        projections[display_name] = (str(x_column), y_column)
    return projections


def _point_payload(
    frame: pd.DataFrame,
    projections: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    normalized = {
        name: _normalized_coordinates(frame, x_column, y_column)
        for name, (x_column, y_column) in projections.items()
    }
    diagnostic_columns = [
        column
        for column in (
            "knn_radius_k15",
            "participation_ratio_k15",
            "two_nn_lid",
            "outlier_score",
            "distance_to_label_centroid",
        )
        if column in frame
    ]
    points: list[dict[str, Any]] = []
    for position, row in frame.iterrows():
        details = {
            column: _json_scalar(row.get(column))
            for column in diagnostic_columns
        }
        points.append(
            {
                "rowId": int(row.get("row_id", position)),
                "sourceIndex": _json_scalar(row.get("source_index", position)),
                "label": str(row.get("label", "")),
                "dataset": str(row.get("dataset", "")),
                "atlas": int(row["atlas_index"]),
                "column": int(row["atlas_column"]),
                "row": int(row["atlas_row"]),
                "coordinates": {
                    name: [
                        float(normalized[name][position, 0]),
                        float(normalized[name][position, 1]),
                    ]
                    for name in projections
                },
                "details": details,
            }
        )
    return points


def _projection_diagnostics_payload(
    frame: pd.DataFrame,
    projections: dict[str, tuple[str, str]],
) -> dict[str, dict[str, dict[str, str]]]:
    payload: dict[str, dict[str, dict[str, str]]] = {}
    for name, (x_column, _) in projections.items():
        projection_key = x_column[:-2]
        details: dict[str, dict[str, str]] = {}
        prefixes = (
            ("knn_radius_k", "kNN radius"),
            ("label_agreement_k", "Local label agreement"),
        )
        for suffix_prefix, label in prefixes:
            column = next(
                (
                    str(value)
                    for value in frame.columns
                    if str(value).startswith(f"{projection_key}_{suffix_prefix}")
                ),
                None,
            )
            if column:
                k_value = column.rsplit("k", 1)[-1]
                details[f"{label} (k={k_value})"] = _float32_payload(frame[column])
        centroid_column = f"{projection_key}_distance_to_label_centroid"
        if centroid_column in frame:
            details["Distance to label centroid"] = _float32_payload(
                frame[centroid_column]
            )
        payload[name] = details
    return payload


def _float32_payload(series: pd.Series) -> dict[str, str]:
    values = np.asarray(series, dtype="<f4")
    return {
        "encoding": "float32-base64",
        "data": base64.b64encode(values.tobytes()).decode("ascii"),
    }


def _normalized_coordinates(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> np.ndarray:
    values = frame[[x_column, y_column]].to_numpy(dtype=np.float64)
    center = np.nanmean(values, axis=0, keepdims=True)
    values = values - center
    maximum = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0
    return np.nan_to_num(values / maximum * 20.0)


def _atlas_digest(frame: pd.DataFrame, *, tile_size: int, atlas_size: int) -> str:
    digest = hashlib.sha256(f"{tile_size}:{atlas_size}".encode())
    for row in frame.itertuples(index=False):
        path = Path(str(getattr(row, "image_path", "")))
        digest.update(str(path).encode())
        digest.update(str(getattr(row, "label", "")).encode())
        if path.is_file():
            stat = path.stat()
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _prepacked_atlas_bundle(frame: pd.DataFrame) -> AtlasBundle | None:
    required = {
        "sprite_atlas_path",
        "sprite_atlas_index",
        "sprite_atlas_column",
        "sprite_atlas_row",
        "sprite_tile_size",
        "sprite_atlas_columns",
    }
    if not required <= set(frame.columns) or frame.empty:
        return None
    prepared = frame.reset_index(drop=True).copy()
    atlas_rows = (
        prepared[["sprite_atlas_index", "sprite_atlas_path"]]
        .drop_duplicates()
        .sort_values("sprite_atlas_index")
    )
    expected = list(range(len(atlas_rows)))
    actual = atlas_rows["sprite_atlas_index"].astype(int).tolist()
    if actual != expected:
        raise ValueError(f"Prepacked atlas indices must be contiguous: {actual}")
    atlas_paths = [Path(value) for value in atlas_rows["sprite_atlas_path"]]
    missing = [path for path in atlas_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepacked sprite atlas does not exist: {missing[0]}")
    prepared["atlas_index"] = prepared["sprite_atlas_index"].astype(int)
    prepared["atlas_column"] = prepared["sprite_atlas_column"].astype(int)
    prepared["atlas_row"] = prepared["sprite_atlas_row"].astype(int)
    labels = sorted(
        {str(value) for value in prepared.get("label", pd.Series(dtype=str))},
        key=_natural_sort_key,
    )
    palette = {
        label: LABEL_PALETTE[index % len(LABEL_PALETTE)]
        for index, label in enumerate(labels)
    }
    return AtlasBundle(
        frame=prepared,
        atlas_paths=atlas_paths,
        palette=palette,
        tile_size=int(prepared["sprite_tile_size"].iloc[0]),
        atlas_columns=int(prepared["sprite_atlas_columns"].iloc[0]),
    )


def _projection_display_name(key: str) -> str:
    if key == "tsne":
        return "T-SNE"
    if key == "umap":
        return "UMAP"
    if key == "pca":
        return "PCA"
    return key.replace("_", " ").title()


def _remove_old_atlases(directory: Path, *, keep: set[Path]) -> None:
    for path in directory.glob("atlas_*.png"):
        if path not in keep:
            path.unlink()


def _natural_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _json_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _html_template(
    payload_json: str,
    *,
    height: int,
    config: ExplorerConfig,
) -> str:
    metrics_display = "grid" if config.show_metrics else "none"
    legend_display = "flex" if config.show_legend else "none"
    controls_display = "grid" if config.show_view_controls else "none"
    instructions_display = "block" if config.show_instructions else "none"
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; overflow: hidden; background: #111; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #f2f2f2; }}
  #app {{ display: grid; grid-template-columns: {config.sidebar_width}px 1fr; height: {height}px; background: #111; }}
  #sidebar {{ background: #222; padding: 16px; display: flex; flex-direction: column; gap: 14px; min-width: 0; }}
  .control {{ display: grid; grid-template-columns: 88px 1fr; align-items: center; gap: 10px; }}
  label, .muted {{ color: #c8c8c8; font-size: 14px; }}
  select {{ width: 100%; height: 32px; background: #f3f3f3; color: #111; border: 0; border-radius: 2px; padding: 0 8px; }}
  #preview-wrap {{ width: 100%; aspect-ratio: 1; background: #191919; display: grid; place-items: center; }}
  #preview {{ width: 100%; height: 100%; image-rendering: pixelated; }}
  #sample-info {{ min-height: 88px; display: grid; gap: 5px; align-content: start; }}
  #sample-label {{ font-size: 24px; font-weight: 650; }}
  #sample-index {{ color: #a9a9a9; font-variant-numeric: tabular-nums; }}
  #metrics {{ display: {metrics_display}; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 10px; margin-top: 8px; padding-top: 10px; border-top: 1px solid #3a3a3a; font-size: 12px; color: #bdbdbd; }}
  .metrics-heading {{ grid-column: 1 / -1; color: #f0f0f0; font-size: 13px; font-weight: 600; margin-bottom: 2px; }}
  .metric-key {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .metric-value {{ color: #eee; font-variant-numeric: tabular-nums; text-align: right; }}
  #legend {{ display: {legend_display}; flex-wrap: wrap; gap: 6px 10px; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 5px; font-size: 12px; color: #cfcfcf; }}
  .swatch {{ width: 9px; height: 9px; }}
  #sidebar-footer {{ display: {instructions_display}; margin-top: auto; color: #aaa; font-size: 12px; line-height: 1.45; }}
  #main {{ position: relative; min-width: 0; overflow: hidden; background: #111; }}
  #plot {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }}
  #plot.dragging {{ cursor: grabbing; }}
  #status {{ position: absolute; right: 14px; bottom: 12px; color: #777; font-size: 12px; pointer-events: none; }}
  #view-controls {{ position: absolute; right: 14px; top: 12px; display: {controls_display}; gap: 5px; }}
  #view-controls button {{ width: 34px; height: 34px; border: 1px solid #444; background: #1b1b1b; color: #ddd; cursor: pointer; font-size: 20px; }}
  #view-controls button:hover {{ background: #292929; }}
  @media (max-width: 760px) {{
    #app {{ grid-template-columns: 1fr; grid-template-rows: 1fr 150px; }}
    #sidebar {{ grid-row: 2; display: grid; grid-template-columns: 120px 1fr 1fr; gap: 10px; padding: 10px; overflow: hidden; }}
    #preview-wrap {{ grid-row: 1 / span 3; }}
    #legend, #sidebar-footer {{ display: none; }}
    #main {{ grid-row: 1; }}
    .control {{ grid-template-columns: 1fr; gap: 4px; align-content: start; }}
    #sample-label {{ font-size: 20px; }}
  }}
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <div class="control">
      <label for="projection">{config.selector_label}</label>
      <select id="projection"></select>
    </div>
    <div id="preview-wrap"><canvas id="preview"></canvas></div>
    <div id="sample-info">
      <div class="muted">Label</div>
      <div id="sample-label">-</div>
      <div id="sample-index">Index: -</div>
      <div id="metrics"></div>
    </div>
    <div id="legend"></div>
    <div id="sidebar-footer">
      Drag to pan. Scroll to zoom. Hover to inspect. Click to pin a sample.
      Double-click or press the reset control to refit the projection.
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
const DATA = {payload_json};
for (const details of Object.values(DATA.projectionDiagnostics)) {{
  for (const [name, encoded] of Object.entries(details)) {{
    const binary = atob(encoded.data);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) {{
      bytes[index] = binary.charCodeAt(index);
    }}
    details[name] = new Float32Array(bytes.buffer);
  }}
}}
const canvas = document.getElementById("plot");
const context = canvas.getContext("2d");
const preview = document.getElementById("preview");
const previewContext = preview.getContext("2d");
const previewTile = document.createElement("canvas");
previewTile.width = DATA.tileSize;
previewTile.height = DATA.tileSize;
const previewTileContext = previewTile.getContext("2d");
const main = document.getElementById("main");
const projectionSelect = document.getElementById("projection");
const labelElement = document.getElementById("sample-label");
const indexElement = document.getElementById("sample-index");
const metricsElement = document.getElementById("metrics");
const statusElement = document.getElementById("status");
const resetButton = document.getElementById("reset");
const zoomInButton = document.getElementById("zoom-in");
const zoomOutButton = document.getElementById("zoom-out");
const atlasImages = [];
let width = 1;
let height = 1;
let pixelRatio = Math.max(1, window.devicePixelRatio || 1);
let centerX = 0;
let centerY = 0;
let scale = 1;
let fitScale = 1;
let dragging = false;
let moved = false;
let dragX = 0;
let dragY = 0;
let hoverIndex = null;
let pinnedIndex = null;
let projection = DATA.projections[0];
let currentCoordinates = DATA.points.map(point => point.coordinates[projection].slice());
let animation = null;
let frameRequested = false;

for (const name of DATA.projections) {{
  const option = document.createElement("option");
  option.value = name;
  option.textContent = name;
  projectionSelect.appendChild(option);
}}
for (const [label, color] of Object.entries(DATA.palette)) {{
  const item = document.createElement("span");
  item.className = "legend-item";
  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = color;
  const text = document.createElement("span");
  text.textContent = label;
  item.append(swatch, text);
  document.getElementById("legend").appendChild(item);
}}

function loadAtlases() {{
  return Promise.all(DATA.atlases.map(source => new Promise((resolve, reject) => {{
    const image = new Image();
    image.onload = () => {{ atlasImages.push(image); resolve(); }};
    image.onerror = reject;
    image.src = source;
  }})));
}}

function resize() {{
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
  fitView();
}}

function fitView() {{
  const coordinates = DATA.points.map(point => point.coordinates[projection]);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const coordinate of coordinates) {{
    minX = Math.min(minX, coordinate[0]);
    maxX = Math.max(maxX, coordinate[0]);
    minY = Math.min(minY, coordinate[1]);
    maxY = Math.max(maxY, coordinate[1]);
  }}
  centerX = (minX + maxX) / 2;
  centerY = (minY + maxY) / 2;
  const dataWidth = Math.max(1, maxX - minX);
  const dataHeight = Math.max(1, maxY - minY);
  fitScale = Math.min(width / (dataWidth * 1.12), height / (dataHeight * 1.12));
  scale = fitScale;
  requestDraw();
}}

function pointSize() {{
  if (!DATA.options.scalePointSizeWithZoom) return DATA.options.pointSize;
  const zoom = Math.max(1, scale / fitScale);
  return Math.max(
    DATA.options.pointSize * 0.9,
    Math.min(
      DATA.options.pointSize * 3.1,
      DATA.options.pointSize + Math.log2(zoom) * 5
    )
  );
}}

function zoomBy(factor) {{
  scale = Math.max(fitScale * 0.5, Math.min(fitScale * 60, scale * factor));
  requestDraw();
}}

function worldToScreen(coordinate) {{
  return [
    (coordinate[0] - centerX) * scale + width / 2,
    (centerY - coordinate[1]) * scale + height / 2,
  ];
}}

function draw() {{
  frameRequested = false;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111";
  context.fillRect(0, 0, width, height);
  const size = pointSize();
  for (let index = 0; index < DATA.points.length; index++) {{
    const point = DATA.points[index];
    const screen = worldToScreen(currentCoordinates[index]);
    if (screen[0] < -size || screen[0] > width + size || screen[1] < -size || screen[1] > height + size) continue;
    drawTile(context, point, screen[0] - size / 2, screen[1] - size / 2, size);
  }}
  const selectedIndex = pinnedIndex !== null ? pinnedIndex : hoverIndex;
  if (selectedIndex !== null) {{
    const screen = worldToScreen(currentCoordinates[selectedIndex]);
    const highlightSize = Math.max(DATA.options.hoverSize, size * 2.0);
    context.strokeStyle = "#fff";
    context.lineWidth = 2;
    context.strokeRect(
      screen[0] - highlightSize / 2 - 2,
      screen[1] - highlightSize / 2 - 2,
      highlightSize + 4,
      highlightSize + 4
    );
    drawTile(
      context,
      DATA.points[selectedIndex],
      screen[0] - highlightSize / 2,
      screen[1] - highlightSize / 2,
      highlightSize
    );
  }}
  statusElement.textContent = `${{DATA.points.length.toLocaleString()}} samples`;
  if (animation) requestAnimationFrame(stepAnimation);
}}

function drawTile(targetContext, point, x, y, size) {{
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
}}

function drawPreviewTile(point, x, y, size) {{
  if (DATA.options.previewMode !== "original" || point.dataset.toLowerCase() !== "mnist") {{
    drawTile(previewContext, point, x, y, size);
    return;
  }}
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
  const image = previewTileContext.getImageData(
    0,
    0,
    DATA.tileSize,
    DATA.tileSize
  );
  for (let offset = 0; offset < image.data.length; offset += 4) {{
    image.data[offset] = 255;
    image.data[offset + 1] = 255;
    image.data[offset + 2] = 255;
  }}
  previewTileContext.putImageData(image, 0, 0);
  previewContext.imageSmoothingEnabled = false;
  previewContext.drawImage(previewTile, x, y, size, size);
}}

function requestDraw() {{
  if (!frameRequested) {{
    frameRequested = true;
    requestAnimationFrame(draw);
  }}
}}

function nearestPoint(mouseX, mouseY) {{
  const threshold = Math.max(10, pointSize() * 0.75);
  let best = null;
  let bestDistance = threshold * threshold;
  for (let index = 0; index < currentCoordinates.length; index++) {{
    const screen = worldToScreen(currentCoordinates[index]);
    const dx = screen[0] - mouseX;
    const dy = screen[1] - mouseY;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {{
      best = index;
      bestDistance = distance;
    }}
  }}
  return best;
}}

function showPoint(index) {{
  previewContext.clearRect(0, 0, preview.clientWidth, preview.clientHeight);
  previewContext.fillStyle = "#191919";
  previewContext.fillRect(0, 0, preview.clientWidth, preview.clientHeight);
  metricsElement.replaceChildren();
  if (index === null) {{
    labelElement.textContent = "-";
    indexElement.textContent = "Index: -";
    return;
  }}
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
  indexElement.textContent = `Index: ${{point.sourceIndex}}  Row: ${{point.rowId}}`;
  appendMetricHeading(`Diagnostics · ${{projection}}`);
  const normalizedCoordinate = point.coordinates[projection];
  appendMetric("Map X", normalizedCoordinate[0]);
  appendMetric("Map Y", normalizedCoordinate[1]);
  const selectedDetails = DATA.projectionDiagnostics[projection] || {{}};
  for (const [name, values] of Object.entries(selectedDetails)) {{
    appendMetric(name, values[index]);
  }}
  if (Object.keys(point.details).length) appendMetricHeading("Sample");
  for (const [name, value] of Object.entries(point.details)) {{
    if (value === null) continue;
    appendMetric(name.replaceAll("_", " "), value);
  }}
}}

function appendMetricHeading(text) {{
  const heading = document.createElement("div");
  heading.className = "metrics-heading";
  heading.textContent = text;
  metricsElement.appendChild(heading);
}}

function appendMetric(name, value) {{
  if (value === null) return;
  const key = document.createElement("span");
  key.className = "metric-key";
  key.title = name;
  key.textContent = name;
  const metric = document.createElement("span");
  metric.className = "metric-value";
  if (typeof value === "number") {{
    metric.textContent = name.includes("agreement")
      ? `${{(value * 100).toFixed(1)}}%`
      : Number.isInteger(value)
        ? value.toLocaleString()
        : value.toFixed(3);
  }} else {{
    metric.textContent = String(value);
  }}
  metricsElement.append(key, metric);
}}

function stepAnimation(timestamp) {{
  if (!animation) return;
  const progress = Math.min(1, (timestamp - animation.start) / animation.duration);
  const eased = DATA.options.transitionEasing === "linear"
    ? progress
    : progress < 0.5
      ? 2 * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 2) / 2;
  currentCoordinates = animation.from.map((coordinate, index) => [
    coordinate[0] + (animation.to[index][0] - coordinate[0]) * eased,
    coordinate[1] + (animation.to[index][1] - coordinate[1]) * eased,
  ]);
  if (progress >= 1) {{
    currentCoordinates = animation.to;
    animation = null;
    fitView();
  }}
  requestDraw();
}}

projectionSelect.addEventListener("change", event => {{
  projection = event.target.value;
  animation = {{
    from: currentCoordinates.map(coordinate => coordinate.slice()),
    to: DATA.points.map(point => point.coordinates[projection].slice()),
    start: performance.now(),
    duration: DATA.options.transitionMs,
  }};
  const selectedIndex = pinnedIndex !== null ? pinnedIndex : hoverIndex;
  showPoint(selectedIndex);
  requestDraw();
}});

canvas.addEventListener("pointerdown", event => {{
  dragging = true;
  moved = false;
  dragX = event.clientX;
  dragY = event.clientY;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", event => {{
  const rect = canvas.getBoundingClientRect();
  if (dragging) {{
    const dx = event.clientX - dragX;
    const dy = event.clientY - dragY;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    centerX -= dx / scale;
    centerY += dy / scale;
    dragX = event.clientX;
    dragY = event.clientY;
    requestDraw();
    return;
  }}
  if (pinnedIndex !== null) return;
  hoverIndex = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
  showPoint(hoverIndex);
  requestDraw();
}});
canvas.addEventListener("pointerup", event => {{
  dragging = false;
  canvas.classList.remove("dragging");
  if (!moved) {{
    const rect = canvas.getBoundingClientRect();
    const selected = nearestPoint(event.clientX - rect.left, event.clientY - rect.top);
    pinnedIndex = pinnedIndex === selected ? null : selected;
    hoverIndex = selected;
    showPoint(pinnedIndex !== null ? pinnedIndex : hoverIndex);
  }}
  requestDraw();
}});
canvas.addEventListener("pointerleave", () => {{
  dragging = false;
  canvas.classList.remove("dragging");
  if (pinnedIndex === null) {{
    hoverIndex = null;
    showPoint(null);
    requestDraw();
  }}
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  const worldX = centerX + (mouseX - width / 2) / scale;
  const worldY = centerY - (mouseY - height / 2) / scale;
  const factor = Math.exp(-event.deltaY * 0.0012);
  const nextScale = Math.max(fitScale * 0.5, Math.min(fitScale * 60, scale * factor));
  centerX = worldX - (mouseX - width / 2) / nextScale;
  centerY = worldY + (mouseY - height / 2) / nextScale;
  scale = nextScale;
  requestDraw();
}}, {{ passive: false }});
canvas.addEventListener("dblclick", fitView);
resetButton.addEventListener("click", fitView);
zoomInButton.addEventListener("click", () => zoomBy(1.5));
zoomOutButton.addEventListener("click", () => zoomBy(1 / 1.5));
window.addEventListener("resize", resize);

loadAtlases().then(() => {{
  resize();
  showPoint(null);
}}).catch(error => {{
  statusElement.textContent = `Failed to load thumbnail atlas: ${{error}}`;
}});
</script>
</body>
</html>"""
