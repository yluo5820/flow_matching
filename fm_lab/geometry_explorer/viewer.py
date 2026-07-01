"""Unified Three.js geometry viewer for datasets and trajectories."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from fm_lab.image_diagnostics.explorer_viewer import shared_explorer_script
from fm_lab.image_diagnostics.three_explorer import _load_three_source


def build_geometry_html(
    payload: dict,
    *,
    height: int = 760,
    three_source: str | None = None,
    vendor_dir: str | Path | None = None,
) -> str:
    """Build one self-contained unified geometry explorer document."""

    if three_source is None:
        three_source = _load_three_source(Path(vendor_dir or "outputs/geometry_explorer/vendor"))
    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    safe_three = three_source.replace("</script", "<\\/script")
    script = _geometry_script(payload_json)
    return _html_template(
        payload_json=payload_json,
        script=script,
        three_source=safe_three,
        height=height,
    )


def _html_template(
    *,
    payload_json: str,
    script: str,
    three_source: str,
    height: int,
) -> str:
    del payload_json
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Geometry Explorer</title>
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; height: 100%; overflow: hidden; background: #111; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #f2f2f2; }}
  #app {{ --sidebar-width: 320px; --dock-height: 190px; display: grid; grid-template-columns: var(--sidebar-width) 7px minmax(0, 1fr); grid-template-rows: minmax(0, 1fr) 7px var(--dock-height); height: {height}px; min-height: 0; background: #111; overflow: hidden; }}
  #sidebar {{ grid-column: 1; grid-row: 1 / span 3; background: #222; padding: 16px; display: flex; flex-direction: column; gap: 13px; min-width: 0; min-height: 0; overflow: hidden; }}
  .control {{ display: grid; grid-template-columns: 92px 1fr; align-items: center; gap: 9px; }}
  label, .muted {{ color: #c8c8c8; font-size: 13px; }}
  select {{ width: 100%; height: 32px; background: #f3f3f3; color: #111; border: 0; border-radius: 2px; padding: 0 8px; }}
  button {{ border: 1px solid #444; background: #1b1b1b; color: #ddd; cursor: pointer; }}
  button:hover {{ background: #292929; }}
  .toggle-row {{ display: flex; align-items: center; gap: 7px; min-height: 24px; color: #ddd; font-size: 12px; }}
  .toggle-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px; }}
  .class-menu {{ position: relative; min-width: 0; }}
  .class-menu summary {{ height: 32px; display: flex; align-items: center; padding: 0 8px; background: #f3f3f3; color: #111; border-radius: 2px; cursor: pointer; list-style: none; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .class-menu summary::-webkit-details-marker {{ display: none; }}
  .class-menu summary::after {{ content: "v"; margin-left: auto; padding-left: 8px; color: #555; }}
  .class-options {{ position: absolute; z-index: 20; top: 36px; left: 0; right: 0; max-height: 250px; overflow-y: auto; padding: 6px; background: #f3f3f3; color: #111; border: 1px solid #bbb; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35); }}
  .class-option {{ min-height: 28px; display: flex; align-items: center; gap: 7px; padding: 2px 4px; color: #111; font-size: 12px; cursor: pointer; }}
  .class-option:hover {{ background: #ddd; }}
  .class-option input {{ margin: 0; }}
  .class-count {{ margin-left: auto; color: #666; font-variant-numeric: tabular-nums; }}
  #preview-wrap {{ width: 100%; aspect-ratio: 1; flex: 0 0 auto; background: #191919; display: grid; place-items: center; }}
  #preview {{ width: 100%; height: 100%; image-rendering: pixelated; }}
  #sample-info {{ flex: 0 0 auto; display: grid; gap: 5px; align-content: start; min-height: 88px; }}
  #sample-label {{ font-size: 24px; font-weight: 650; }}
  #sample-index {{ color: #a9a9a9; font-variant-numeric: tabular-nums; }}
  .splitter {{ position: relative; z-index: 8; background: #171717; transition: background 120ms ease; }}
  .splitter::after {{ content: ""; position: absolute; inset: 0; background: transparent; }}
  .splitter:hover, .splitter.active {{ background: #3a3a3a; }}
  #sidebar-splitter {{ grid-column: 2; grid-row: 1 / span 3; cursor: col-resize; }}
  #dock-splitter {{ grid-column: 3; grid-row: 2; cursor: row-resize; }}
  body.resizing-layout, body.resizing-layout * {{ user-select: none; }}
  #diagnostics-dock {{ grid-column: 3; grid-row: 3; min-height: 0; background: #202020; border-top: 1px solid #343434; padding: 12px 16px; overflow-y: auto; overscroll-behavior: contain; }}
  #metrics {{ display: grid; grid-template-columns: repeat(3, minmax(140px, 1fr) max-content); gap: 6px 14px; font-size: 12px; color: #bdbdbd; align-content: start; }}
  .metrics-heading {{ grid-column: 1 / -1; color: #f0f0f0; font-size: 13px; font-weight: 600; margin-bottom: 2px; }}
  .metric-key {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #bdbdbd; }}
  .metric-value {{ color: #eee; font-variant-numeric: tabular-nums; text-align: right; }}
  #legend {{ flex: 1 1 auto; min-height: 0; display: flex; align-content: start; align-items: flex-start; flex-wrap: wrap; gap: 6px 10px; overflow-y: auto; padding-top: 10px; border-top: 1px solid #333; }}
  .legend-item {{ display: inline-flex; align-items: center; gap: 5px; max-width: 100%; font-size: 12px; color: #cfcfcf; }}
  .legend-item span:last-child {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .swatch {{ width: 9px; height: 9px; flex: 0 0 auto; }}
  #main {{ grid-column: 3; grid-row: 1; position: relative; min-width: 0; min-height: 0; overflow: hidden; background: #111; }}
  #main canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }}
  #main canvas.dragging {{ cursor: grabbing; }}
  #status {{ position: absolute; right: 14px; bottom: 12px; color: #777; font-size: 12px; pointer-events: none; }}
  #view-controls {{ position: absolute; right: 14px; top: 12px; display: grid; gap: 5px; }}
  #view-controls button {{ width: 34px; height: 34px; font-size: 20px; }}
  @media (max-width: 1180px) {{
    #metrics {{ grid-template-columns: repeat(2, minmax(130px, 1fr) max-content); }}
  }}
  @media (max-width: 760px) {{
    #app {{ grid-template-columns: 1fr; grid-template-rows: 1fr 7px 230px 7px var(--dock-height); }}
    #sidebar {{ grid-column: 1; grid-row: 3; display: grid; grid-template-columns: 105px 145px minmax(0, 1fr); grid-template-rows: 1fr 1fr; gap: 8px 10px; padding: 10px; overflow: hidden; }}
    #preview-wrap {{ grid-column: 1; grid-row: 1 / span 2; }}
    #sample-info {{ grid-column: 3; grid-row: 1 / span 2; overflow-y: auto; }}
    #sidebar-splitter {{ display: none; }}
    #dock-splitter {{ grid-column: 1; grid-row: 4; }}
    #diagnostics-dock {{ grid-column: 1; grid-row: 5; }}
    #metrics {{ grid-template-columns: minmax(0, 1fr) max-content; }}
    #main {{ grid-column: 1; grid-row: 1; }}
    .control {{ grid-template-columns: 1fr; gap: 4px; align-content: start; }}
    .toggle-grid, #legend {{ display: none; }}
  }}
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <div class="control">
      <label for="projection">Projection</label>
      <select id="projection"></select>
    </div>
    <div class="control">
      <span class="muted">Class</span>
      <details id="class-filter" class="class-menu">
        <summary id="class-filter-summary">All classes</summary>
        <div id="class-options" class="class-options"></div>
      </details>
    </div>
    <div class="control" id="time-control">
      <label for="time">Time</label>
      <div>
        <input id="time" type="range" min="0" max="0" value="0" style="width:100%">
        <div id="time-readout" class="muted">t = 0 / 0</div>
      </div>
    </div>
    <div class="control">
      <span class="muted">Layers</span>
      <div class="toggle-grid">
        <label class="toggle-row"><input id="show-target" type="checkbox" checked>Target</label>
        <label class="toggle-row"><input id="show-generated" type="checkbox" checked>Generated</label>
        <label class="toggle-row"><input id="show-trajectory" type="checkbox" checked>Path</label>
        <label class="toggle-row"><input id="show-thumbnails" type="checkbox" checked>Thumbnails</label>
      </div>
    </div>
    <div id="preview-wrap"><canvas id="preview"></canvas></div>
    <div id="sample-info">
      <div class="muted">Label</div>
      <div id="sample-label">-</div>
      <div id="sample-index">Index: -</div>
    </div>
    <div id="legend"></div>
  </aside>
  <div id="sidebar-splitter" class="splitter" role="separator" aria-orientation="vertical" title="Drag to resize sidebar"></div>
  <main id="main">
    <div id="view-controls">
      <button id="zoom-in" title="Zoom in" aria-label="Zoom in">+</button>
      <button id="zoom-out" title="Zoom out" aria-label="Zoom out">&minus;</button>
      <button id="reset" title="Reset view" aria-label="Reset view">&#8634;</button>
    </div>
    <div id="status"></div>
  </main>
  <div id="dock-splitter" class="splitter" role="separator" aria-orientation="horizontal" title="Drag to resize diagnostics"></div>
  <section id="diagnostics-dock">
    <div id="metrics"></div>
  </section>
</div>
<script>{three_source}</script>
<script>{script}</script>
</body>
</html>"""


def _geometry_script(payload_json: str) -> str:
    script = r'''__SHARED_SCRIPT__
const DATA = __PAYLOAD_JSON__;
for (const details of Object.values(DATA.projectionDiagnostics || {})) {
  for (const [name, encoded] of Object.entries(details)) {
    const binary = atob(encoded.data);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    details[name] = new Float32Array(bytes.buffer);
  }
}

const app = document.getElementById("app");
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
const slider = document.getElementById("time");
const timeReadout = document.getElementById("time-readout");
const timeControl = document.getElementById("time-control");
const showTarget = document.getElementById("show-target");
const showGenerated = document.getElementById("show-generated");
const showTrajectory = document.getElementById("show-trajectory");
const thumbnailToggle = document.getElementById("show-thumbnails");
const resetButton = document.getElementById("reset");
const zoomInButton = document.getElementById("zoom-in");
const zoomOutButton = document.getElementById("zoom-out");
const sidebarSplitter = document.getElementById("sidebar-splitter");
const dockSplitter = document.getElementById("dock-splitter");
const atlasImages = [];
let atlasTextures = [];

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(58, 1, 0.1, 800);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
renderer.outputColorSpace = THREE.SRGBColorSpace;
main.prepend(renderer.domElement);
const group = new THREE.Group();
scene.add(group);
const endpointGroup = new THREE.Group();
const lineGroup = new THREE.Group();
const particleGroup = new THREE.Group();
const hoverGroup = new THREE.Group();
group.add(lineGroup, endpointGroup, particleGroup, hoverGroup);
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.75;
const mouse = new THREE.Vector2();
const projected = new THREE.Vector3();

let projection = DATA.projections[0];
let step = 0;
let radius = 55;
let theta = 0.72;
let phi = 1.05;
let dragging = false;
let moved = false;
let pointerX = 0;
let pointerY = 0;
let hoverSelection = null;
let pinnedSelection = null;
let selectedLabels = new Set(DATA.points.map(point => point.label));
let visibleIndices = DATA.points.map((_, index) => index);
let renderRequested = false;
let layoutDragMode = null;

for (const name of DATA.projections) {
  const option = document.createElement("option");
  option.value = name;
  option.textContent = name;
  projectionSelect.appendChild(option);
}
projection = defaultProjectionName(DATA.projections);
projectionSelect.value = projection;
slider.max = Math.max(0, (DATA.trajectory || []).length - 1);
timeControl.style.display = DATA.mode === "trajectory" ? "grid" : "none";
thumbnailToggle.checked = Boolean(DATA.options.drawThumbnailsDefault);
populateLegend(DATA.palette || {});

function defaultProjectionName(projections) {
  return projections.find(name => name.toLowerCase().includes("umap")) || projections[0] || "";
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function startLayoutDrag(mode, event) {
  layoutDragMode = mode;
  event.preventDefault();
  event.stopPropagation();
  document.body.classList.add("resizing-layout");
  (mode === "sidebar" ? sidebarSplitter : dockSplitter).classList.add("active");
  updateLayoutDrag(event);
}

function updateLayoutDrag(event) {
  const rect = app.getBoundingClientRect();
  if (layoutDragMode === "sidebar") {
    const maximum = Math.max(260, Math.min(560, rect.width - 420));
    const width = clamp(event.clientX - rect.left, 240, maximum);
    app.style.setProperty("--sidebar-width", `${width}px`);
  } else if (layoutDragMode === "dock") {
    const maximum = Math.max(150, rect.height - 260);
    const height = clamp(rect.bottom - event.clientY, 120, Math.min(420, maximum));
    app.style.setProperty("--dock-height", `${height}px`);
  }
  resize();
}

function finishLayoutDrag() {
  if (!layoutDragMode) return;
  layoutDragMode = null;
  document.body.classList.remove("resizing-layout");
  sidebarSplitter.classList.remove("active");
  dockSplitter.classList.remove("active");
  resize();
}

function parseColor(label) {
  const fallback = "rgb(220, 220, 220)";
  const value = DATA.palette[label] || fallback;
  const match = value.match(/\d+/g) || [220, 220, 220];
  return [Number(match[0]) / 255, Number(match[1]) / 255, Number(match[2]) / 255];
}

function pointCoordinate(point) {
  if (Array.isArray(point.coordinates)) return point.coordinates;
  return point.coordinates[projection] || [0, 0, 0];
}

function pointVisible(point) {
  if (!selectedLabels.has(point.label)) return false;
  if (point.kind === "target" && !showTarget.checked) return false;
  if (point.kind === "generated" && !showGenerated.checked) return false;
  return true;
}

function trajectoryLabelVisible(label) {
  return DATA.points.some(point => point.label === label) ? selectedLabels.has(label) : true;
}

function clearObject(object) {
  while (object.children.length) {
    const child = object.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (child.material.map) child.material.map.dispose();
      child.material.dispose();
    }
  }
}

function rebuildScene() {
  clearObject(endpointGroup);
  clearObject(lineGroup);
  clearObject(particleGroup);
  buildEndpoints();
  if (DATA.mode === "trajectory" && showTrajectory.checked) {
    buildTrajectoryLines();
    buildTrajectoryParticles();
  }
  updateStatus();
  updateHoverMarker();
  requestRender();
}

function buildEndpoints() {
  visibleIndices = DATA.points
    .map((point, index) => pointVisible(point) ? index : -1)
    .filter(index => index >= 0);
  if (!visibleIndices.length) return;
  if (thumbnailToggle.checked && canUseAtlasPointCloud()) {
    buildAtlasPointClouds(visibleIndices);
    return;
  }
  if (thumbnailToggle.checked && visibleIndices.length <= 2000) {
    for (const pointIndex of visibleIndices) {
      endpointGroup.add(endpointSprite(DATA.points[pointIndex], pointIndex));
    }
    return;
  }
  const positions = new Float32Array(visibleIndices.length * 3);
  const colors = new Float32Array(visibleIndices.length * 3);
  visibleIndices.forEach((pointIndex, visiblePosition) => {
    const point = DATA.points[pointIndex];
    const coordinate = pointCoordinate(point);
    positions.set([coordinate[0] || 0, coordinate[1] || 0, coordinate[2] || 0], visiblePosition * 3);
    colors.set(parseColor(point.label), visiblePosition * 3);
  });
  endpointGroup.add(coloredPointCloud(positions, colors, {
    pointSize: Math.max(5.5, DATA.options.pointSize * 0.72),
    opacity: 0.88,
    kind: "endpoint",
    indices: visibleIndices,
  }));
}

function canUseAtlasPointCloud() {
  return atlasTextures.length > 0 && Number(DATA.atlasSize || 0) > 0;
}

function buildAtlasPointClouds(indices) {
  const byAtlas = new Map();
  for (const pointIndex of indices) {
    const point = DATA.points[pointIndex];
    if (!byAtlas.has(point.atlas)) byAtlas.set(point.atlas, []);
    byAtlas.get(point.atlas).push(pointIndex);
  }
  for (const [atlas, atlasIndices] of byAtlas.entries()) {
    const texture = atlasTextures[atlas];
    if (!texture || !atlasIndices.length) continue;
    const positions = new Float32Array(atlasIndices.length * 3);
    const offsets = new Float32Array(atlasIndices.length * 2);
    atlasIndices.forEach((pointIndex, local) => {
      const point = DATA.points[pointIndex];
      const coordinate = pointCoordinate(point);
      positions.set([coordinate[0] || 0, coordinate[1] || 0, coordinate[2] || 0], local * 3);
      offsets[local * 2] = point.column * DATA.tileSize / DATA.atlasSize;
      offsets[local * 2 + 1] = point.row * DATA.tileSize / DATA.atlasSize;
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("offset", new THREE.BufferAttribute(offsets, 2));
    const material = new THREE.ShaderMaterial({
      uniforms: {
        textureAtlas: { value: texture },
        repeat: { value: new THREE.Vector2(DATA.tileSize / DATA.atlasSize, DATA.tileSize / DATA.atlasSize) },
        pointSize: { value: DATA.options.pointSize },
      },
      vertexShader: `
        attribute vec2 offset;
        varying vec2 vOffset;
        uniform float pointSize;
        void main() {
          vOffset = offset;
          gl_PointSize = pointSize;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D textureAtlas;
        uniform vec2 repeat;
        varying vec2 vOffset;
        void main() {
          vec2 uv = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
          vec4 texel = texture2D(textureAtlas, vOffset + uv * repeat);
          if (texel.a < 0.04) discard;
          gl_FragColor = texel;
        }
      `,
      transparent: true,
      depthTest: true,
      depthWrite: true,
    });
    const points = new THREE.Points(geometry, material);
    points.userData = { kind: "endpoint", indices: atlasIndices };
    endpointGroup.add(points);
  }
}

function endpointSprite(point, pointIndex) {
  const coordinate = pointCoordinate(point);
  const tile = document.createElement("canvas");
  tile.width = DATA.tileSize;
  tile.height = DATA.tileSize;
  const tileContext = tile.getContext("2d");
  tileContext.imageSmoothingEnabled = false;
  tileContext.drawImage(
    atlasImages[point.atlas],
    point.column * DATA.tileSize,
    point.row * DATA.tileSize,
    DATA.tileSize,
    DATA.tileSize,
    0,
    0,
    DATA.tileSize,
    DATA.tileSize
  );
  const texture = new THREE.CanvasTexture(tile);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, alphaTest: 0.05 });
  const sprite = new THREE.Sprite(material);
  sprite.position.set(coordinate[0] || 0, coordinate[1] || 0, coordinate[2] || 0);
  sprite.scale.set(1.15, 1.15, 1.15);
  sprite.userData = { kind: "endpoint", index: pointIndex };
  return sprite;
}

function initializeAtlasTextures() {
  atlasTextures = atlasImages.map(image => {
    const texture = new THREE.Texture(image);
    texture.flipY = false;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.LinearFilter;
    texture.needsUpdate = true;
    return texture;
  });
}

function buildTrajectoryLines() {
  const limit = Math.max(1, step + 1);
  for (let index = 0; index < DATA.counts.trajectories; index++) {
    const label = DATA.trajectoryLabels[index] || "trajectory";
    if (!trajectoryLabelVisible(label)) continue;
    const positions = new Float32Array(limit * 3);
    for (let t = 0; t < limit; t++) {
      const coordinate = DATA.trajectory[t][index];
      positions.set([coordinate[0], coordinate[1], coordinate[2]], t * 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const color = new THREE.Color(...parseColor(label));
    const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: DATA.options.lineAlpha || 0.32 });
    lineGroup.add(new THREE.Line(geometry, material));
  }
}

function buildTrajectoryParticles() {
  if (!DATA.trajectory.length) return;
  const indices = [];
  for (let index = 0; index < DATA.counts.trajectories; index++) {
    const label = DATA.trajectoryLabels[index] || "trajectory";
    if (trajectoryLabelVisible(label)) indices.push(index);
  }
  const positions = new Float32Array(indices.length * 3);
  const colors = new Float32Array(indices.length * 3);
  indices.forEach((trajectoryIndex, visiblePosition) => {
    const coordinate = DATA.trajectory[step][trajectoryIndex];
    const label = DATA.trajectoryLabels[trajectoryIndex] || "trajectory";
    positions.set([coordinate[0], coordinate[1], coordinate[2]], visiblePosition * 3);
    colors.set(parseColor(label), visiblePosition * 3);
  });
  const points = coloredPointCloud(positions, colors, {
    pointSize: Math.max(5.0, DATA.options.pointSize * 0.62),
    opacity: 0.92,
    kind: "trajectory",
    indices,
  });
  particleGroup.add(points);
}

function coloredPointCloud(positions, colors, options) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.ShaderMaterial({
    uniforms: {
      pointSize: { value: options.pointSize },
      opacity: { value: options.opacity },
    },
    vertexShader: `
      attribute vec3 color;
      varying vec3 vColor;
      uniform float pointSize;
      void main() {
        vColor = color;
        gl_PointSize = pointSize;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      uniform float opacity;
      void main() {
        vec2 centered = gl_PointCoord * 2.0 - 1.0;
        float distance = length(centered);
        if (distance > 1.0) discard;
        float edge = smoothstep(0.78, 1.0, distance);
        float alpha = opacity * (1.0 - smoothstep(0.86, 1.0, distance));
        vec3 color = mix(vColor, vec3(0.03, 0.03, 0.03), edge * 0.28);
        gl_FragColor = vec4(color, alpha);
      }
    `,
    transparent: true,
    depthTest: true,
    depthWrite: false,
  });
  const points = new THREE.Points(geometry, material);
  points.userData = { kind: options.kind, indices: options.indices };
  return points;
}

function updateCamera() {
  phi = Math.max(0.08, Math.min(Math.PI - 0.08, phi));
  camera.position.set(
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta)
  );
  camera.lookAt(0, 0, 0);
}

function render() {
  renderRequested = false;
  updateCamera();
  renderer.render(scene, camera);
}

function requestRender() {
  if (renderRequested) return;
  renderRequested = true;
  window.requestAnimationFrame(render);
}

function resize() {
  const rect = main.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = Math.max(1e-3, width / height);
  camera.updateProjectionMatrix();
  preview.width = Math.max(1, Math.round(preview.clientWidth * window.devicePixelRatio));
  preview.height = Math.max(1, Math.round(preview.clientHeight * window.devicePixelRatio));
  previewContext.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
  requestRender();
}

function selectionLabel(selection) {
  if (!selection) return "-";
  if (selection.kind === "trajectory") return DATA.trajectoryLabels[selection.index] || "trajectory";
  return DATA.points[selection.index].label || "-";
}

function sameSelection(left, right) {
  if (!left && !right) return true;
  if (!left || !right) return false;
  return left.kind === right.kind && left.index === right.index;
}

function showSelection(selection) {
  updateHoverMarker();
  clearPreview();
  metricsElement.replaceChildren();
  labelElement.textContent = "-";
  indexElement.textContent = "Index: -";
  if (!selection) {
    showGroupDiagnostics(null);
    return;
  }
  if (selection.kind === "trajectory") {
    const previewPoint = DATA.trajectoryPreviews[selection.index];
    if (previewPoint) drawPreview(previewPoint);
    const current = DATA.trajectory[step][selection.index];
    const final = DATA.trajectory[DATA.trajectory.length - 1][selection.index];
    labelElement.textContent = selectionLabel(selection);
    indexElement.textContent = `trajectory - Index: ${selection.index} - final image`;
    appendMetric(metricsElement, "Current UMAP X", current[0]);
    appendMetric(metricsElement, "Current UMAP Y", current[1]);
    appendMetric(metricsElement, "Current UMAP Z", current[2]);
    appendMetric(metricsElement, "Final UMAP X", final[0]);
    appendMetric(metricsElement, "Final UMAP Y", final[1]);
    appendMetric(metricsElement, "Final UMAP Z", final[2]);
    showGroupDiagnostics(selectionLabel(selection));
    return;
  }
  const point = DATA.points[selection.index];
  drawPreview(point);
  labelElement.textContent = point.label || "-";
  indexElement.textContent = `${point.kind || "sample"} - Index: ${point.sourceIndex} - Row: ${point.rowId}`;
  const coordinate = pointCoordinate(point);
  appendMetricHeading(`Diagnostics · ${projection}`);
  appendMetric(metricsElement, "Map X", coordinate[0]);
  appendMetric(metricsElement, "Map Y", coordinate[1]);
  if ((DATA.projectionDimensions || {})[projection] === 3) appendMetric(metricsElement, "Map Z", coordinate[2] || 0);
  const diagnostics = (DATA.projectionDiagnostics || {})[projection] || {};
  for (const [key, values] of Object.entries(diagnostics)) appendMetric(metricsElement, metricLabel(key), values[selection.index]);
  showGroupDiagnostics(point.label);
  if (Object.keys(point.details || {}).length) appendMetricHeading("Sample");
  for (const [key, value] of Object.entries(point.details || {})) appendMetric(metricsElement, metricLabel(key), value);
}

function metricLabel(key) {
  return (DATA.metricLabels || {})[key] || key.replaceAll("_", " ");
}

function appendMetricHeading(text) {
  const heading = document.createElement("div");
  heading.className = "metrics-heading";
  heading.textContent = text;
  metricsElement.appendChild(heading);
}

function showGroupDiagnostics(label) {
  const diagnostics = DATA.groupDiagnostics || {};
  const groups = diagnostics.groups || {};
  const metrics = diagnostics.metrics || [];
  if (!metrics.length) return;
  const labelKey = label === null || label === undefined ? null : String(label);
  if (labelKey && groups[labelKey]) {
    appendMetricHeading(`Class ID · ${labelKey}`);
    appendGroupMetricRows(groups[labelKey], metrics);
    return;
  }
  const filteredLabel = selectedGroupLabel(groups);
  if (filteredLabel) {
    appendMetricHeading(`Class ID · ${filteredLabel}`);
    appendGroupMetricRows(groups[filteredLabel], metrics);
    return;
  }
  if (diagnostics.overall) {
    appendMetricHeading("Global ID · All classes");
    appendGroupMetricRows(diagnostics.overall, metrics);
  }
  const primary = diagnostics.primaryMetric || metrics[0];
  const labels = Object.keys(groups)
    .filter(value => selectedLabels.has(value))
    .sort((left, right) => left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" }));
  if (!labels.length) return;
  appendMetricHeading(`Class ID · ${metricLabel(primary)}`);
  for (const value of labels) appendMetric(metricsElement, value || "(empty)", groups[value][primary]);
}

function selectedGroupLabel(groups) {
  if (selectedLabels.size !== 1) return null;
  const [label] = Array.from(selectedLabels);
  return groups[label] ? label : null;
}

function appendGroupMetricRows(row, metrics) {
  appendMetric(metricsElement, "Samples", row.n_samples);
  if (row.class_share !== null && row.class_share !== undefined && row.class_share < 0.9999) {
    appendMetric(metricsElement, "Dataset share", row.class_share);
  }
  for (const metric of metrics) appendMetric(metricsElement, metricLabel(metric), row[metric]);
}

function drawPreview(point) {
  const margin = 12;
  drawPreviewTile(point, margin, margin, preview.clientWidth - margin * 2, preview.clientHeight - margin * 2);
}

function updateHoverMarker() {
  clearObject(hoverGroup);
  const selection = pinnedSelection || hoverSelection;
  if (!selection) {
    requestRender();
    return;
  }
  if (selection.kind === "trajectory") {
    const previewPoint = DATA.trajectoryPreviews[selection.index];
    const coordinate = (DATA.trajectory[step] || [])[selection.index];
    if (previewPoint && coordinate && thumbnailToggle.checked && addHoverAtlasThumbnail(previewPoint, coordinate)) {
      requestRender();
      return;
    }
    addHoverColorPoint(coordinate, selectionLabel(selection));
    requestRender();
    return;
  }
  const point = DATA.points[selection.index];
  if (!point || !pointVisible(point)) {
    requestRender();
    return;
  }
  const coordinate = pointCoordinate(point);
  if (thumbnailToggle.checked && addHoverAtlasThumbnail(point, coordinate)) {
    requestRender();
    return;
  }
  addHoverColorPoint(coordinate, point.label);
  requestRender();
}

function addHoverAtlasThumbnail(point, coordinate) {
  const texture = atlasTextures[point.atlas];
  if (!texture || !coordinate) return false;
  const positions = new Float32Array([coordinate[0] || 0, coordinate[1] || 0, coordinate[2] || 0]);
  const offsets = new Float32Array([
    point.column * DATA.tileSize / DATA.atlasSize,
    point.row * DATA.tileSize / DATA.atlasSize,
  ]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("offset", new THREE.BufferAttribute(offsets, 2));
  const material = new THREE.ShaderMaterial({
    uniforms: {
      textureAtlas: { value: texture },
      repeat: { value: new THREE.Vector2(DATA.tileSize / DATA.atlasSize, DATA.tileSize / DATA.atlasSize) },
      pointSize: { value: Math.max(DATA.options.hoverSize || 0, DATA.options.pointSize * 3.5) },
    },
    vertexShader: `
      attribute vec2 offset;
      varying vec2 vOffset;
      uniform float pointSize;
      void main() {
        vOffset = offset;
        gl_PointSize = pointSize;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D textureAtlas;
      uniform vec2 repeat;
      varying vec2 vOffset;
      void main() {
        vec2 uv = vec2(gl_PointCoord.x, 1.0 - gl_PointCoord.y);
        vec4 texel = texture2D(textureAtlas, vOffset + uv * repeat);
        if (texel.a < 0.04) discard;
        gl_FragColor = texel;
      }
    `,
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });
  hoverGroup.add(new THREE.Points(geometry, material));
  return true;
}

function addHoverColorPoint(coordinate, label) {
  if (!coordinate) return;
  const positions = new Float32Array([coordinate[0] || 0, coordinate[1] || 0, coordinate[2] || 0]);
  const colors = new Float32Array(parseColor(label));
  hoverGroup.add(coloredPointCloud(positions, colors, {
    pointSize: Math.max(18, (DATA.options.hoverSize || DATA.options.pointSize * 3.0) * 0.55),
    opacity: 0.98,
    kind: "hover",
    indices: [0],
  }));
}

function intersectSelection(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersections = raycaster.intersectObjects([particleGroup, endpointGroup], true);
  if (!intersections.length) return null;
  const hit = intersections[0];
  if (hit.object.userData.index !== undefined) {
    return { kind: hit.object.userData.kind, index: hit.object.userData.index };
  }
  const indices = hit.object.userData.indices || [];
  const sourceIndex = indices[hit.index];
  return { kind: hit.object.userData.kind, index: sourceIndex };
}

function updateStatus() {
  const selected = pinnedSelection || hoverSelection;
  timeReadout.textContent = `t = ${step} / ${Math.max(0, (DATA.trajectory || []).length - 1)}`;
  statusElement.textContent = `${visibleIndices.length.toLocaleString()} points` +
    (DATA.mode === "trajectory" ? ` - ${DATA.counts.trajectories.toLocaleString()} trajectories` : "") +
    (selected ? ` - ${selectionLabel(selected)}` : "");
}

function onClassFilterChange() {
  hoverSelection = null;
  pinnedSelection = null;
  showSelection(null);
  rebuildScene();
}

projectionSelect.addEventListener("change", () => {
  projection = projectionSelect.value;
  rebuildScene();
});
slider.addEventListener("input", () => {
  step = Number(slider.value);
  hoverSelection = null;
  showSelection(pinnedSelection);
  rebuildScene();
});
for (const control of [showTarget, showGenerated, showTrajectory, thumbnailToggle]) {
  control.addEventListener("change", rebuildScene);
}
sidebarSplitter.addEventListener("pointerdown", event => startLayoutDrag("sidebar", event));
dockSplitter.addEventListener("pointerdown", event => startLayoutDrag("dock", event));
renderer.domElement.addEventListener("pointerdown", event => {
  dragging = true;
  moved = false;
  pointerX = event.clientX;
  pointerY = event.clientY;
  renderer.domElement.classList.add("dragging");
});
window.addEventListener("pointermove", event => {
  if (layoutDragMode) {
    updateLayoutDrag(event);
    return;
  }
  if (dragging) {
    const dx = event.clientX - pointerX;
    const dy = event.clientY - pointerY;
    pointerX = event.clientX;
    pointerY = event.clientY;
    moved = moved || Math.abs(dx) + Math.abs(dy) > 2;
    theta -= dx * 0.008;
    phi -= dy * 0.008;
    requestRender();
    return;
  }
  const nextSelection = intersectSelection(event);
  if (!sameSelection(hoverSelection, nextSelection)) {
    hoverSelection = nextSelection;
    if (!pinnedSelection) showSelection(hoverSelection);
    updateStatus();
  }
});
window.addEventListener("pointerup", event => {
  if (layoutDragMode) {
    finishLayoutDrag();
    return;
  }
  if (!dragging) return;
  dragging = false;
  renderer.domElement.classList.remove("dragging");
  if (!moved) {
    pinnedSelection = intersectSelection(event);
    showSelection(pinnedSelection);
    updateStatus();
  }
});
renderer.domElement.addEventListener("wheel", event => {
  event.preventDefault();
  radius *= Math.exp(event.deltaY * 0.001);
  radius = Math.max(10, Math.min(180, radius));
  requestRender();
}, { passive: false });
resetButton.addEventListener("click", () => {
  radius = 55;
  theta = 0.72;
  phi = 1.05;
  requestRender();
});
zoomInButton.addEventListener("click", () => {
  radius = Math.max(10, radius / 1.25);
  requestRender();
});
zoomOutButton.addEventListener("click", () => {
  radius = Math.min(180, radius * 1.25);
  requestRender();
});
window.addEventListener("resize", resize);
if (window.ResizeObserver) {
  const resizeObserver = new ResizeObserver(() => resize());
  resizeObserver.observe(main);
  resizeObserver.observe(document.getElementById("preview-wrap"));
}

initializeClassFilter(DATA.points, onClassFilterChange);
loadAtlases(DATA.atlases).then(() => {
  initializeAtlasTextures();
  resize();
  rebuildScene();
  showSelection(null);
});
'''
    return script.replace("__SHARED_SCRIPT__", shared_explorer_script()).replace(
        "__PAYLOAD_JSON__",
        payload_json,
    )
