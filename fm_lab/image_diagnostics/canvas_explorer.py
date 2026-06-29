"""Thumbnail-atlas canvas renderer for projected datasets."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from fm_lab.image_diagnostics.config import ExplorerConfig
from fm_lab.image_diagnostics.explorer_payload import (
    atlas_data_url,
    palette_payload,
    projection_columns,
    projection_diagnostics_payload,
    projection_point_payload,
)
from fm_lab.image_diagnostics.explorer_viewer import (
    ExplorerDocument,
    build_explorer_document,
    class_filter_html,
    shared_explorer_script,
)
from fm_lab.image_diagnostics.palette import LABEL_PALETTE

ATLAS_COMPACTION_THRESHOLD = 32 * 1024 * 1024
ATLAS_WEBP_QUALITY = 90


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
        return _compact_atlas_bundle(prepacked)
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


def prepare_array_sprite_atlases(
    frame: pd.DataFrame,
    images: np.ndarray,
    *,
    output_dir: str | Path,
    image_shape: list[int] | tuple[int, ...] | None,
    image_value_range: list[float] | tuple[float, float] = (0.0, 1.0),
    tile_size: int = 28,
    max_atlas_size: int = 2048,
) -> AtlasBundle:
    """Pack in-memory image arrays into the same atlas layout as file previews."""

    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    prepared = frame.reset_index(drop=True).copy()
    images_np = np.asarray(images)
    if images_np.shape[0] != len(prepared):
        raise ValueError(
            f"Image count {images_np.shape[0]} does not match frame rows {len(prepared)}."
        )

    atlas_columns = max(1, max_atlas_size // tile_size)
    atlas_capacity = atlas_columns * atlas_columns
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
    digest = _array_atlas_digest(
        prepared,
        images_np,
        image_shape=image_shape,
        image_value_range=image_value_range,
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
        _build_array_atlases(
            prepared,
            images_np,
            atlas_paths=atlas_paths,
            palette=palette,
            image_shape=image_shape,
            image_value_range=image_value_range,
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
    return _compact_atlas_bundle(
        AtlasBundle(
            frame=prepared,
            atlas_paths=atlas_paths,
            palette=palette,
            tile_size=tile_size,
            atlas_columns=atlas_columns,
        )
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
    projections = projection_columns(bundle.frame, projection_names=projection_names)
    if not projections:
        raise ValueError("Explorer data contains no UMAP, PCA, or t-SNE coordinates.")
    points = projection_point_payload(bundle.frame, projections)
    atlases = [atlas_data_url(path) for path in bundle.atlas_paths]
    payload = {
        "points": points,
        "atlases": atlases,
        "projections": list(projections),
        "projectionDiagnostics": projection_diagnostics_payload(
            bundle.frame,
            projections,
        ),
        "palette": palette_payload(bundle.palette),
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
        if str(row.get("dataset", "")).lower() in {"mnist", "fashion_mnist"}:
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


def _build_array_atlases(
    frame: pd.DataFrame,
    images: np.ndarray,
    *,
    atlas_paths: list[Path],
    palette: dict[str, tuple[int, int, int]],
    image_shape: list[int] | tuple[int, ...] | None,
    image_value_range: list[float] | tuple[float, float],
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
        image = _array_preview(
            images[position],
            image_shape=image_shape,
            image_value_range=image_value_range,
            tile_size=tile_size,
        )
        label = str(row.get("label", ""))
        if str(row.get("dataset", "")).lower() in {"mnist", "fashion_mnist"}:
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


def _array_preview(
    values: np.ndarray,
    *,
    image_shape: list[int] | tuple[int, ...] | None,
    image_value_range: list[float] | tuple[float, float],
    tile_size: int,
) -> Image.Image:
    array = np.asarray(values)
    if image_shape is not None:
        shape = tuple(int(value) for value in image_shape)
        expected = int(np.prod(shape))
        if array.size == expected:
            array = array.reshape(shape)
    if array.ndim == 1:
        side = int(round(math.sqrt(array.size)))
        if side * side != array.size:
            return Image.new("RGBA", (tile_size, tile_size), (255, 255, 255, 180))
        array = array.reshape(side, side)

    low = float(image_value_range[0])
    high = float(image_value_range[1])
    denom = high - low if high > low else 1.0
    scaled = np.clip((array.astype(np.float32, copy=False) - low) / denom, 0.0, 1.0)
    scaled = np.rint(scaled * 255.0).astype(np.uint8)
    if scaled.ndim == 2:
        image = Image.fromarray(scaled, mode="L").convert("RGBA")
    elif scaled.ndim == 3 and scaled.shape[-1] == 1:
        image = Image.fromarray(scaled[..., 0], mode="L").convert("RGBA")
    elif scaled.ndim == 3 and scaled.shape[-1] in {3, 4}:
        mode = "RGB" if scaled.shape[-1] == 3 else "RGBA"
        image = Image.fromarray(scaled, mode=mode).convert("RGBA")
    else:
        return Image.new("RGBA", (tile_size, tile_size), (255, 255, 255, 180))

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


def _array_atlas_digest(
    frame: pd.DataFrame,
    images: np.ndarray,
    *,
    image_shape: list[int] | tuple[int, ...] | None,
    image_value_range: list[float] | tuple[float, float],
    tile_size: int,
    atlas_size: int,
) -> str:
    digest = hashlib.sha256(
        f"{tile_size}:{atlas_size}:{image_shape}:{tuple(image_value_range)}".encode()
    )
    digest.update(str(images.shape).encode())
    digest.update(str(images.dtype).encode())
    digest.update(np.ascontiguousarray(images).view(np.uint8))
    for row in frame.itertuples(index=False):
        digest.update(str(getattr(row, "label", "")).encode())
        digest.update(str(getattr(row, "dataset", "")).encode())
        digest.update(str(getattr(row, "kind", "")).encode())
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


def _compact_atlas_bundle(
    bundle: AtlasBundle,
    *,
    threshold_bytes: int = ATLAS_COMPACTION_THRESHOLD,
) -> AtlasBundle:
    total_bytes = sum(path.stat().st_size for path in bundle.atlas_paths)
    if total_bytes <= threshold_bytes:
        return bundle
    compact_paths = [
        _compact_atlas_path(path)
        for path in bundle.atlas_paths
    ]
    return AtlasBundle(
        frame=bundle.frame,
        atlas_paths=compact_paths,
        palette=bundle.palette,
        tile_size=bundle.tile_size,
        atlas_columns=bundle.atlas_columns,
    )


def _compact_atlas_path(path: Path) -> Path:
    if path.suffix.lower() == ".webp":
        return path
    compact_path = path.with_name(f"{path.stem}_q{ATLAS_WEBP_QUALITY}.webp")
    if (
        compact_path.exists()
        and compact_path.stat().st_mtime_ns >= path.stat().st_mtime_ns
    ):
        return compact_path
    compact_path.with_name(f".{compact_path.name}.tmp").unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{compact_path.name}.",
        suffix=".tmp",
        dir=compact_path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with Image.open(path) as image:
            image.save(
                temporary,
                format="WEBP",
                quality=ATLAS_WEBP_QUALITY,
                method=6,
                exact=True,
            )
        temporary.replace(compact_path)
    finally:
        temporary.unlink(missing_ok=True)
    return compact_path


def _remove_old_atlases(directory: Path, *, keep: set[Path]) -> None:
    for path in directory.glob("atlas_*.png"):
        if path not in keep:
            path.unlink()


def _natural_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)




def _html_template(
    payload_json: str,
    *,
    height: int,
    config: ExplorerConfig,
) -> str:
    controls_html = (
        '<div class="control">\n'
        '      <label for="projection">' + config.selector_label + '</label>\n'
        '      <select id="projection"></select>\n'
        '    </div>\n'
        + class_filter_html()
    )
    sample_info_html = (
        '<div class="muted">Label</div>\n'
        '      <div id="sample-label">-</div>\n'
        '      <div id="sample-index">Index: -</div>\n'
        '      <div id="metrics"></div>'
    )
    footer_html = (
        'Drag to pan. Scroll to zoom. Hover to inspect. Click to pin a sample.\n'
        '      Double-click or press the reset control to refit the projection.'
    )
    return build_explorer_document(
        ExplorerDocument(
            title="Dataset Explorer",
            controls_html=controls_html,
            sample_info_html=sample_info_html,
            footer_html=footer_html,
            script=_canvas_script(payload_json),
            height=height,
            config=config,
            control_label_width=88,
            mobile_height=150,
            mobile_columns="120px 140px minmax(0, 1fr)",
            mobile_rows="1fr 1fr",
            preview_row_span=2,
        )
    )


def _canvas_script(payload_json: str) -> str:
    script = r'''__SHARED_SCRIPT__
const DATA = __PAYLOAD_JSON__;
for (const details of Object.values(DATA.projectionDiagnostics)) {
  for (const [name, encoded] of Object.entries(details)) {
    const binary = atob(encoded.data);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) {
      bytes[index] = binary.charCodeAt(index);
    }
    details[name] = new Float32Array(bytes.buffer);
  }
}
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
let visibleIndices = DATA.points.map((_, index) => index);
let selectedLabels = new Set(DATA.points.map(point => point.label));
let animation = null;
let frameRequested = false;

for (const name of DATA.projections) {
  const option = document.createElement("option");
  option.value = name;
  option.textContent = name;
  projectionSelect.appendChild(option);
}
populateLegend(DATA.palette);

function onClassFilterChange() {
  animation = null;
  currentCoordinates = DATA.points.map(
    point => point.coordinates[projection].slice()
  );
  hoverIndex = null;
  pinnedIndex = null;
  showPoint(null);
  fitView();
}

function resize() {
  resizeExplorerCanvases(fitView);
}

function fitView() {
  const coordinates = visibleIndices.map(index => currentCoordinates[index]);
  if (!coordinates.length) {
    centerX = 0;
    centerY = 0;
    fitScale = 1;
    scale = 1;
    requestDraw();
    return;
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const coordinate of coordinates) {
    minX = Math.min(minX, coordinate[0]);
    maxX = Math.max(maxX, coordinate[0]);
    minY = Math.min(minY, coordinate[1]);
    maxY = Math.max(maxY, coordinate[1]);
  }
  centerX = (minX + maxX) / 2;
  centerY = (minY + maxY) / 2;
  const dataWidth = Math.max(1, maxX - minX);
  const dataHeight = Math.max(1, maxY - minY);
  fitScale = Math.min(width / (dataWidth * 1.12), height / (dataHeight * 1.12));
  scale = fitScale;
  requestDraw();
}

function pointSize() {
  if (!DATA.options.scalePointSizeWithZoom) return DATA.options.pointSize;
  const zoom = Math.max(1, scale / fitScale);
  return Math.max(
    DATA.options.pointSize * 0.9,
    Math.min(
      DATA.options.pointSize * 3.1,
      DATA.options.pointSize + Math.log2(zoom) * 5
    )
  );
}

function zoomBy(factor) {
  scale = Math.max(fitScale * 0.5, Math.min(fitScale * 60, scale * factor));
  requestDraw();
}

function worldToScreen(coordinate) {
  return [
    (coordinate[0] - centerX) * scale + width / 2,
    (centerY - coordinate[1]) * scale + height / 2,
  ];
}

function draw() {
  frameRequested = false;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111";
  context.fillRect(0, 0, width, height);
  const size = pointSize();
  for (const index of visibleIndices) {
    const point = DATA.points[index];
    const screen = worldToScreen(currentCoordinates[index]);
    if (screen[0] < -size || screen[0] > width + size || screen[1] < -size || screen[1] > height + size) continue;
    drawTile(context, point, screen[0] - size / 2, screen[1] - size / 2, size);
  }
  const selectedIndex = pinnedIndex !== null ? pinnedIndex : hoverIndex;
  if (selectedIndex !== null && selectedLabels.has(DATA.points[selectedIndex].label)) {
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
  }
  statusElement.textContent = `${visibleIndices.length.toLocaleString()} samples`;
  if (animation) requestAnimationFrame(stepAnimation);
}

function nearestPoint(mouseX, mouseY) {
  return nearestVisiblePoint(
    mouseX,
    mouseY,
    Math.max(10, pointSize() * 0.75),
    index => worldToScreen(currentCoordinates[index])
  );
}

function showPoint(index) {
  clearPreview();
  metricsElement.replaceChildren();
  if (index === null) {
    labelElement.textContent = "-";
    indexElement.textContent = "Index: -";
    return;
  }
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
  indexElement.textContent = `Index: ${point.sourceIndex}  Row: ${point.rowId}`;
  appendMetricHeading(`Diagnostics - ${projection}`);
  const normalizedCoordinate = point.coordinates[projection];
  appendMetric(metricsElement, "Map X", normalizedCoordinate[0]);
  appendMetric(metricsElement, "Map Y", normalizedCoordinate[1]);
  const selectedDetails = DATA.projectionDiagnostics[projection] || {};
  for (const [name, values] of Object.entries(selectedDetails)) {
    appendMetric(metricsElement, name, values[index]);
  }
  if (Object.keys(point.details).length) appendMetricHeading("Sample");
  for (const [name, value] of Object.entries(point.details)) {
    if (value === null) continue;
    appendMetric(metricsElement, name.replaceAll("_", " "), value);
  }
}

function appendMetricHeading(text) {
  const heading = document.createElement("div");
  heading.className = "metrics-heading";
  heading.textContent = text;
  metricsElement.appendChild(heading);
}

function stepAnimation(timestamp) {
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
  if (progress >= 1) {
    currentCoordinates = animation.to;
    animation = null;
    fitView();
  }
  requestDraw();
}

projectionSelect.addEventListener("change", event => {
  projection = event.target.value;
  animation = {
    from: currentCoordinates.map(coordinate => coordinate.slice()),
    to: DATA.points.map(point => point.coordinates[projection].slice()),
    start: performance.now(),
    duration: DATA.options.transitionMs,
  };
  const selectedIndex = pinnedIndex !== null ? pinnedIndex : hoverIndex;
  showPoint(selectedIndex);
  requestDraw();
});

installCanvasPointerHandlers((_event, dx, dy) => {
  centerX -= dx / scale;
  centerY += dy / scale;
});
canvas.addEventListener("wheel", event => {
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
}, { passive: false });
canvas.addEventListener("dblclick", fitView);
resetButton.addEventListener("click", fitView);
zoomInButton.addEventListener("click", () => zoomBy(1.5));
zoomOutButton.addEventListener("click", () => zoomBy(1 / 1.5));
window.addEventListener("resize", resize);

initializeClassFilter(DATA.points, onClassFilterChange);
loadAtlases(DATA.atlases).then(() => {
  resize();
  showPoint(null);
}).catch(error => {
  statusElement.textContent = `Failed to load thumbnail atlas: ${error}`;
});
'''
    return script.replace("__SHARED_SCRIPT__", shared_explorer_script()).replace(
        "__PAYLOAD_JSON__",
        payload_json,
    )
