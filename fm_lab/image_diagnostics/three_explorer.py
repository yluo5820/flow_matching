"""Three.js renderer for mixed two- and three-dimensional projections."""

# ruff: noqa: E501

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
from PIL import Image

from fm_lab.image_diagnostics.canvas_explorer import (
    AtlasBundle,
    prepare_sprite_atlases,
)
from fm_lab.image_diagnostics.config import ExplorerConfig
from fm_lab.image_diagnostics.explorer_payload import (
    atlas_data_url,
    projection_columns,
    projection_diagnostics_payload,
    projection_dimensions,
    projection_point_payload,
)

THREE_VERSION = "0.159.0"
THREE_URL = (
    f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.min.js"
)


def render_thumbnail_three(
    frame: pd.DataFrame,
    *,
    data_path: str | Path,
    config: ExplorerConfig,
    projection_names: dict[str, str] | None = None,
) -> None:
    """Render flat and spatial projections with a Three.js camera."""

    import streamlit as st

    output_dir = Path(data_path).resolve().parent.parent
    bundle = prepare_sprite_atlases(
        frame,
        output_dir=output_dir / "assets" / "atlases",
        tile_size=config.atlas_tile_size,
        max_atlas_size=config.atlas_size,
    )
    three_source = _load_three_source(output_dir / "assets" / "vendor")
    html = build_three_html(
        bundle,
        height=config.height,
        config=config,
        projection_names=projection_names,
        three_source=three_source,
    )
    st.iframe(html, height=config.height, width="stretch", tab_index=0)


def build_three_html(
    bundle: AtlasBundle,
    *,
    height: int,
    config: ExplorerConfig,
    projection_names: dict[str, str] | None = None,
    three_source: str,
) -> str:
    """Build a standalone Three.js document for 2D and 3D projections."""

    projections = projection_columns(
        bundle.frame,
        projection_names=projection_names,
        include_z=True,
    )
    if not projections:
        raise ValueError("Three.js explorer data contains no x/y projection.")
    atlas_size = _atlas_size(bundle)
    payload = {
        "points": projection_point_payload(
            bundle.frame,
            projections,
            output_dimensions=3,
        ),
        "atlases": [atlas_data_url(path) for path in bundle.atlas_paths],
        "projections": list(projections),
        "projectionDimensions": projection_dimensions(projections),
        "projectionDiagnostics": projection_diagnostics_payload(
            bundle.frame,
            projections,
        ),
        "tileSize": bundle.tile_size,
        "atlasSize": atlas_size,
        "options": {
            "pointSize": config.point_size,
            "hoverSize": config.hover_size,
            "previewMode": config.preview_mode,
        },
    }
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    safe_three_source = three_source.replace("</script", "<\\/script")
    return _html_template(
        payload_json,
        three_source=safe_three_source,
        height=height,
        config=config,
    )


def _atlas_size(bundle: AtlasBundle) -> int:
    with Image.open(bundle.atlas_paths[0]) as image:
        return image.width


def _load_three_source(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"three-{THREE_VERSION}.min.js"
    if not path.exists():
        with NamedTemporaryFile(dir=directory, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            urllib.request.urlretrieve(THREE_URL, temporary_path)  # noqa: S310
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path.read_text(encoding="utf-8")


def _html_template(
    payload_json: str,
    *,
    three_source: str,
    height: int,
    config: ExplorerConfig,
) -> str:
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
  .class-menu {{ position: relative; min-width: 0; }}
  .class-menu summary {{ height: 32px; display: flex; align-items: center; padding: 0 8px; background: #f3f3f3; color: #111; border-radius: 2px; cursor: pointer; list-style: none; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .class-menu summary::-webkit-details-marker {{ display: none; }}
  .class-menu summary::after {{ content: "▾"; margin-left: auto; padding-left: 8px; color: #555; }}
  .class-options {{ position: absolute; z-index: 20; top: 36px; left: 0; right: 0; max-height: 250px; overflow-y: auto; padding: 6px; background: #f3f3f3; color: #111; border: 1px solid #bbb; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35); }}
  .class-option {{ min-height: 28px; display: flex; align-items: center; gap: 7px; padding: 2px 4px; color: #111; font-size: 12px; cursor: pointer; }}
  .class-option:hover {{ background: #ddd; }}
  .class-option input {{ margin: 0; }}
  .class-count {{ margin-left: auto; color: #666; font-variant-numeric: tabular-nums; }}
  #preview-wrap {{ width: 100%; aspect-ratio: 1; background: #191919; display: grid; place-items: center; }}
  #preview {{ width: 100%; height: 100%; image-rendering: pixelated; }}
  #sample-info {{ min-height: 88px; display: grid; gap: 5px; align-content: start; }}
  #sample-label {{ font-size: 24px; font-weight: 650; }}
  #sample-index {{ color: #a9a9a9; font-variant-numeric: tabular-nums; }}
  #metrics {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 10px; margin-top: 8px; padding-top: 10px; border-top: 1px solid #3a3a3a; font-size: 12px; color: #bdbdbd; }}
  .metrics-heading {{ grid-column: 1 / -1; color: #f0f0f0; font-size: 13px; font-weight: 600; margin-bottom: 2px; }}
  .metric-key {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .metric-value {{ color: #eee; font-variant-numeric: tabular-nums; text-align: right; }}
  #main {{ position: relative; min-width: 0; overflow: hidden; background: #111; }}
  #main canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }}
  #main canvas.dragging {{ cursor: grabbing; }}
  #highlight {{ display: none; position: absolute; width: {config.hover_size}px; height: {config.hover_size}px; border: 2px solid #fff; pointer-events: none; transform: translate(-50%, -50%); }}
  #status {{ position: absolute; right: 14px; bottom: 12px; color: #777; font-size: 12px; pointer-events: none; }}
  #view-controls {{ position: absolute; right: 14px; top: 12px; display: grid; gap: 5px; }}
  #view-controls button {{ width: 34px; height: 34px; border: 1px solid #444; background: #1b1b1b; color: #ddd; cursor: pointer; font-size: 20px; }}
  #view-controls button:hover {{ background: #292929; }}
  @media (max-width: 760px) {{
    #app {{ grid-template-columns: 1fr; grid-template-rows: 1fr 230px; }}
    #sidebar {{ grid-row: 2; display: grid; grid-template-columns: 100px 140px minmax(0, 1fr); grid-template-rows: 1fr 1fr; gap: 8px 10px; padding: 10px; overflow: hidden; }}
    #sidebar > .control:nth-of-type(1) {{ grid-column: 2; grid-row: 1; }}
    #sidebar > .control:nth-of-type(2) {{ grid-column: 2; grid-row: 2; }}
    #preview-wrap {{ grid-column: 1; grid-row: 1 / span 2; }}
    #sample-info {{ grid-column: 3; grid-row: 1 / span 2; }}
    #main {{ grid-row: 1; }}
    .control {{ grid-template-columns: 1fr; gap: 4px; align-content: start; }}
    #sample-label {{ font-size: 20px; }}
    #sample-info {{ overflow-y: auto; padding-right: 3px; }}
    #sample-index {{ font-size: 12px; }}
    #metrics {{ font-size: 11px; gap: 4px 8px; }}
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
      <div id="metrics"></div>
    </div>
  </aside>
  <main id="main">
    <div id="highlight"></div>
    <div id="view-controls">
      <button id="zoom-in" title="Zoom in" aria-label="Zoom in">+</button>
      <button id="zoom-out" title="Zoom out" aria-label="Zoom out">&minus;</button>
      <button id="reset" title="Reset view" aria-label="Reset view">&#8634;</button>
    </div>
    <div id="status"></div>
  </main>
</div>
<script>{three_source}</script>
<script>
const DATA = {payload_json};
for (const details of Object.values(DATA.projectionDiagnostics)) {{
  for (const [name, encoded] of Object.entries(details)) {{
    const binary = atob(encoded.data);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    details[name] = new Float32Array(bytes.buffer);
  }}
}}

const main = document.getElementById("main");
const preview = document.getElementById("preview");
const previewContext = preview.getContext("2d");
const previewTile = document.createElement("canvas");
previewTile.width = DATA.tileSize;
previewTile.height = DATA.tileSize;
const previewTileContext = previewTile.getContext("2d");
const projectionSelect = document.getElementById("projection");
const classFilterSummary = document.getElementById("class-filter-summary");
const classOptions = document.getElementById("class-options");
const labelElement = document.getElementById("sample-label");
const indexElement = document.getElementById("sample-index");
const metricsElement = document.getElementById("metrics");
const statusElement = document.getElementById("status");
const highlight = document.getElementById("highlight");
const atlasImages = [];

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 500);
const renderer = new THREE.WebGLRenderer({{
  antialias: true,
  alpha: false,
  preserveDrawingBuffer: true,
}});
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.outputColorSpace = THREE.SRGBColorSpace;
main.prepend(renderer.domElement);
const group = new THREE.Group();
scene.add(group);
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.55;
const mouse = new THREE.Vector2();
const projected = new THREE.Vector3();

let projection = DATA.projections[0];
let radius = 52;
let theta = 0.7;
let phi = 1.05;
let focusX = 0;
let focusY = 0;
let focusZ = 0;
let panX = 0;
let panY = 0;
let dragging = false;
let moved = false;
let pointerX = 0;
let pointerY = 0;
let hoverIndex = null;
let pinnedIndex = null;
let renderRequested = false;
let loadedTextures = [];
let visibleIndices = DATA.points.map((_, index) => index);
let selectedLabels = new Set(DATA.points.map(point => point.label));

for (const name of DATA.projections) {{
  const option = document.createElement("option");
  option.value = name;
  option.textContent = name;
  projectionSelect.appendChild(option);
}}

function initializeClassFilter() {{
  const counts = new Map();
  for (const point of DATA.points) {{
    counts.set(point.label, (counts.get(point.label) || 0) + 1);
  }}
  const labels = Array.from(counts.keys()).sort((left, right) =>
    left.localeCompare(right, undefined, {{ numeric: true, sensitivity: "base" }})
  );
  selectedLabels = new Set(labels);
  const allInput = document.createElement("input");
  allInput.type = "checkbox";
  allInput.checked = true;
  classOptions.appendChild(createClassOption(allInput, "All classes", DATA.points.length));
  const classInputs = labels.map(label => {{
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = label;
    input.checked = true;
    classOptions.appendChild(createClassOption(input, label || "(empty)", counts.get(label)));
    input.addEventListener("change", () => {{
      allInput.checked = classInputs.every(value => value.checked);
      syncClassFilter(classInputs);
    }});
    return input;
  }});
  allInput.addEventListener("change", () => {{
    for (const input of classInputs) input.checked = allInput.checked;
    syncClassFilter(classInputs);
  }});
}}

function createClassOption(input, text, count) {{
  const row = document.createElement("label");
  row.className = "class-option";
  const name = document.createElement("span");
  name.textContent = text;
  const total = document.createElement("span");
  total.className = "class-count";
  total.textContent = count.toLocaleString();
  row.append(input, name, total);
  return row;
}}

function syncClassFilter(classInputs) {{
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
        : `${{selectedCount}} classes`;
  hoverIndex = null;
  pinnedIndex = null;
  showPoint(null);
  if (loadedTextures.length) buildPointClouds(loadedTextures);
  resetView();
}}

function loadAssets() {{
  const textureLoader = new THREE.TextureLoader();
  statusElement.textContent = `Loading ${{DATA.points.length.toLocaleString()}} samples...`;
  return Promise.all(DATA.atlases.map((source, atlas) => new Promise((resolve, reject) => {{
    textureLoader.load(source, texture => {{
      texture.flipY = false;
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.magFilter = THREE.NearestFilter;
      texture.minFilter = THREE.LinearFilter;
      atlasImages[atlas] = texture.image;
      resolve(texture);
    }}, undefined, reject);
  }})));
}}

function buildPointClouds(textures) {{
  for (const cloud of group.children) {{
    cloud.geometry.dispose();
    cloud.material.dispose();
  }}
  group.clear();
  for (let atlas = 0; atlas < textures.length; atlas++) {{
    const indices = [];
    for (const index of visibleIndices) {{
      if (DATA.points[index].atlas === atlas) indices.push(index);
    }}
    if (!indices.length) continue;
    const positions = new Float32Array(indices.length * 3);
    const offsets = new Float32Array(indices.length * 2);
    for (let local = 0; local < indices.length; local++) {{
      const point = DATA.points[indices[local]];
      const coordinate = point.coordinates[projection];
      positions[local * 3] = coordinate[0];
      positions[local * 3 + 1] = coordinate[1];
      positions[local * 3 + 2] = coordinate[2];
      offsets[local * 2] = point.column * DATA.tileSize / DATA.atlasSize;
      offsets[local * 2 + 1] = point.row * DATA.tileSize / DATA.atlasSize;
    }}
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("offset", new THREE.BufferAttribute(offsets, 2));
    const material = new THREE.ShaderMaterial({{
      uniforms: {{
        textureAtlas: {{ value: textures[atlas] }},
        repeat: {{ value: new THREE.Vector2(DATA.tileSize / DATA.atlasSize, DATA.tileSize / DATA.atlasSize) }},
        pointSize: {{ value: DATA.options.pointSize }},
      }},
      vertexShader: `
        attribute vec2 offset;
        varying vec2 vOffset;
        uniform float pointSize;
        void main() {{
          vOffset = offset;
          gl_PointSize = pointSize;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }}
      `,
      fragmentShader: `
        uniform sampler2D textureAtlas;
        uniform vec2 repeat;
        varying vec2 vOffset;
        void main() {{
          vec2 uv = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
          vec4 texel = texture2D(textureAtlas, vOffset + uv * repeat);
          if (texel.a < 0.04) discard;
          gl_FragColor = texel;
        }}
      `,
      transparent: true,
      depthTest: true,
      depthWrite: true,
    }});
    const cloud = new THREE.Points(geometry, material);
    cloud.userData.indices = indices;
    group.add(cloud);
  }}
  requestRender();
}}

function updateProjection() {{
  for (const cloud of group.children) {{
    const positions = cloud.geometry.attributes.position.array;
    const indices = cloud.userData.indices;
    for (let local = 0; local < indices.length; local++) {{
      const coordinate = DATA.points[indices[local]].coordinates[projection];
      positions[local * 3] = coordinate[0];
      positions[local * 3 + 1] = coordinate[1];
      positions[local * 3 + 2] = coordinate[2];
    }}
    cloud.geometry.attributes.position.needsUpdate = true;
    cloud.geometry.computeBoundingSphere();
  }}
  resetView();
  showPoint(pinnedIndex !== null ? pinnedIndex : hoverIndex);
}}

function updateCamera() {{
  if (DATA.projectionDimensions[projection] === 2) {{
    camera.position.set(panX, panY, radius);
    camera.lookAt(panX, panY, 0);
    return;
  }}
  camera.position.set(
    focusX + radius * Math.sin(phi) * Math.cos(theta),
    focusY + radius * Math.cos(phi),
    focusZ + radius * Math.sin(phi) * Math.sin(theta)
  );
  camera.lookAt(focusX, focusY, focusZ);
}}

function resetView() {{
  if (!visibleIndices.length) {{
    focusX = 0;
    focusY = 0;
    focusZ = 0;
    radius = 52;
  }} else {{
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (const index of visibleIndices) {{
      const coordinate = DATA.points[index].coordinates[projection];
      minX = Math.min(minX, coordinate[0]);
      maxX = Math.max(maxX, coordinate[0]);
      minY = Math.min(minY, coordinate[1]);
      maxY = Math.max(maxY, coordinate[1]);
      minZ = Math.min(minZ, coordinate[2]);
      maxZ = Math.max(maxZ, coordinate[2]);
    }}
    focusX = (minX + maxX) / 2;
    focusY = (minY + maxY) / 2;
    focusZ = (minZ + maxZ) / 2;
    const span = Math.max(1, maxX - minX, maxY - minY, maxZ - minZ);
    radius = Math.max(6, Math.min(140, span * 1.45));
  }}
  theta = 0.7;
  phi = 1.05;
  panX = focusX;
  panY = focusY;
  updateCamera();
  requestRender();
}}

function resize() {{
  const rect = main.getBoundingClientRect();
  renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
  camera.aspect = Math.max(1, rect.width) / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
  preview.width = Math.round(preview.clientWidth * Math.max(1, window.devicePixelRatio || 1));
  preview.height = Math.round(preview.clientHeight * Math.max(1, window.devicePixelRatio || 1));
  previewContext.setTransform(
    Math.max(1, window.devicePixelRatio || 1), 0, 0,
    Math.max(1, window.devicePixelRatio || 1), 0, 0
  );
  requestRender();
}}

function requestRender() {{
  if (renderRequested) return;
  renderRequested = true;
  requestAnimationFrame(render);
}}

function render() {{
  renderRequested = false;
  renderer.render(scene, camera);
  updateHighlight();
  const dimensions = DATA.projectionDimensions[projection];
  statusElement.textContent = `${{dimensions}}D · ${{visibleIndices.length.toLocaleString()}} samples`;
}}

function pick(event) {{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersections = raycaster.intersectObjects(group.children, false);
  if (!intersections.length) return null;
  intersections.sort((left, right) => (left.distanceToRay || 0) - (right.distanceToRay || 0));
  const intersection = intersections[0];
  return intersection.object.userData.indices[intersection.index];
}}

function updateHighlight() {{
  const index = pinnedIndex !== null ? pinnedIndex : hoverIndex;
  if (index === null) {{
    highlight.style.display = "none";
    return;
  }}
  const coordinate = DATA.points[index].coordinates[projection];
  projected.set(coordinate[0], coordinate[1], coordinate[2]).project(camera);
  const rect = renderer.domElement.getBoundingClientRect();
  highlight.style.display = "block";
  highlight.style.left = `${{(projected.x * 0.5 + 0.5) * rect.width}}px`;
  highlight.style.top = `${{(-projected.y * 0.5 + 0.5) * rect.height}}px`;
}}

function drawPreview(point) {{
  const width = preview.clientWidth;
  const height = preview.clientHeight;
  previewContext.clearRect(0, 0, width, height);
  previewContext.fillStyle = "#191919";
  previewContext.fillRect(0, 0, width, height);
  const sourceX = point.column * DATA.tileSize;
  const sourceY = point.row * DATA.tileSize;
  previewTileContext.clearRect(0, 0, DATA.tileSize, DATA.tileSize);
  previewTileContext.drawImage(
    atlasImages[point.atlas],
    sourceX, sourceY, DATA.tileSize, DATA.tileSize,
    0, 0, DATA.tileSize, DATA.tileSize
  );
  const grayscaleDataset = ["mnist", "fashion_mnist"].includes(point.dataset.toLowerCase());
  if (DATA.options.previewMode === "original" && grayscaleDataset) {{
    const image = previewTileContext.getImageData(0, 0, DATA.tileSize, DATA.tileSize);
    for (let offset = 0; offset < image.data.length; offset += 4) {{
      image.data[offset] = 255;
      image.data[offset + 1] = 255;
      image.data[offset + 2] = 255;
    }}
    previewTileContext.putImageData(image, 0, 0);
  }}
  const margin = 12;
  previewContext.imageSmoothingEnabled = false;
  previewContext.drawImage(previewTile, margin, margin, width - margin * 2, height - margin * 2);
}}

function showPoint(index) {{
  metricsElement.replaceChildren();
  if (index === null) {{
    previewContext.clearRect(0, 0, preview.clientWidth, preview.clientHeight);
    previewContext.fillStyle = "#191919";
    previewContext.fillRect(0, 0, preview.clientWidth, preview.clientHeight);
    labelElement.textContent = "-";
    indexElement.textContent = "Index: -";
    requestRender();
    return;
  }}
  const point = DATA.points[index];
  drawPreview(point);
  labelElement.textContent = point.label || "-";
  indexElement.textContent = `Index: ${{point.sourceIndex}}  Row: ${{point.rowId}}`;
  appendMetricHeading(`Diagnostics · ${{projection}}`);
  const coordinate = point.coordinates[projection];
  appendMetric("Map X", coordinate[0]);
  appendMetric("Map Y", coordinate[1]);
  if (DATA.projectionDimensions[projection] === 3) appendMetric("Map Z", coordinate[2]);
  for (const [name, values] of Object.entries(DATA.projectionDiagnostics[projection] || {{}})) {{
    appendMetric(name, values[index]);
  }}
  if (Object.keys(point.details).length) appendMetricHeading("Sample");
  for (const [name, value] of Object.entries(point.details)) {{
    if (value === null) continue;
    appendMetric(name.replaceAll("_", " "), value);
  }}
  requestRender();
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
  metric.textContent = name.includes("agreement")
    ? `${{(value * 100).toFixed(1)}}%`
    : value.toFixed(3);
  metricsElement.append(key, metric);
}}

projectionSelect.addEventListener("change", event => {{
  projection = event.target.value;
  updateProjection();
}});
renderer.domElement.addEventListener("pointerdown", event => {{
  dragging = true;
  moved = false;
  pointerX = event.clientX;
  pointerY = event.clientY;
  renderer.domElement.classList.add("dragging");
  renderer.domElement.setPointerCapture(event.pointerId);
}});
renderer.domElement.addEventListener("pointermove", event => {{
  if (dragging) {{
    const dx = event.clientX - pointerX;
    const dy = event.clientY - pointerY;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    if (DATA.projectionDimensions[projection] === 3) {{
      theta -= dx * 0.006;
      phi = Math.max(0.12, Math.min(Math.PI - 0.12, phi + dy * 0.006));
    }} else {{
      const worldPerPixel = radius * 0.0018;
      panX -= dx * worldPerPixel;
      panY += dy * worldPerPixel;
    }}
    pointerX = event.clientX;
    pointerY = event.clientY;
    updateCamera();
    requestRender();
    return;
  }}
  if (pinnedIndex !== null) return;
  hoverIndex = pick(event);
  showPoint(hoverIndex);
}});
renderer.domElement.addEventListener("pointerup", event => {{
  dragging = false;
  renderer.domElement.classList.remove("dragging");
  if (!moved) {{
    const selected = pick(event);
    pinnedIndex = pinnedIndex === selected ? null : selected;
    hoverIndex = selected;
    showPoint(pinnedIndex !== null ? pinnedIndex : hoverIndex);
  }}
}});
renderer.domElement.addEventListener("pointerleave", () => {{
  dragging = false;
  renderer.domElement.classList.remove("dragging");
  if (pinnedIndex === null) {{
    hoverIndex = null;
    showPoint(null);
  }}
}});
renderer.domElement.addEventListener("wheel", event => {{
  event.preventDefault();
  radius = Math.max(4, Math.min(140, radius * Math.exp(event.deltaY * 0.001)));
  updateCamera();
  requestRender();
}}, {{ passive: false }});
renderer.domElement.addEventListener("dblclick", resetView);
document.getElementById("zoom-in").addEventListener("click", () => {{
  radius = Math.max(4, radius / 1.3);
  updateCamera();
  requestRender();
}});
document.getElementById("zoom-out").addEventListener("click", () => {{
  radius = Math.min(140, radius * 1.3);
  updateCamera();
  requestRender();
}});
document.getElementById("reset").addEventListener("click", resetView);
window.addEventListener("resize", resize);

initializeClassFilter();
loadAssets().then(textures => {{
  loadedTextures = textures;
  buildPointClouds(loadedTextures);
  resetView();
  resize();
  showPoint(null);
}}).catch(error => {{
  statusElement.textContent = `Failed to load 3D assets: ${{error}}`;
}});
</script>
</body>
</html>"""
