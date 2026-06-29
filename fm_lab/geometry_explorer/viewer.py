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
  #app {{ display: grid; grid-template-columns: 310px 1fr; height: {height}px; background: #111; }}
  #sidebar {{ background: #222; padding: 16px; display: flex; flex-direction: column; gap: 13px; min-width: 0; }}
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
  #preview-wrap {{ width: 100%; aspect-ratio: 1; background: #191919; display: grid; place-items: center; }}
  #preview {{ width: 100%; height: 100%; image-rendering: pixelated; }}
  #sample-info {{ min-height: 116px; display: grid; gap: 5px; align-content: start; }}
  #sample-label {{ font-size: 24px; font-weight: 650; }}
  #sample-index {{ color: #a9a9a9; font-variant-numeric: tabular-nums; }}
  #metrics {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 10px; margin-top: 8px; padding-top: 10px; border-top: 1px solid #3a3a3a; font-size: 12px; color: #bdbdbd; }}
  .metric-key {{ min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #bdbdbd; }}
  .metric-value {{ color: #eee; font-variant-numeric: tabular-nums; text-align: right; }}
  #main {{ position: relative; min-width: 0; overflow: hidden; background: #111; }}
  #main canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }}
  #main canvas.dragging {{ cursor: grabbing; }}
  #status {{ position: absolute; right: 14px; bottom: 12px; color: #777; font-size: 12px; pointer-events: none; }}
  #view-controls {{ position: absolute; right: 14px; top: 12px; display: grid; gap: 5px; }}
  #view-controls button {{ width: 34px; height: 34px; font-size: 20px; }}
  @media (max-width: 760px) {{
    #app {{ grid-template-columns: 1fr; grid-template-rows: 1fr 230px; }}
    #sidebar {{ grid-row: 2; display: grid; grid-template-columns: 105px 145px minmax(0, 1fr); grid-template-rows: 1fr 1fr; gap: 8px 10px; padding: 10px; overflow: hidden; }}
    #preview-wrap {{ grid-column: 1; grid-row: 1 / span 2; }}
    #sample-info {{ grid-column: 3; grid-row: 1 / span 2; overflow-y: auto; }}
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
      <div id="metrics"></div>
    </div>
    <div id="legend"></div>
  </aside>
  <main id="main">
    <div id="view-controls">
      <button id="zoom-in" title="Zoom in" aria-label="Zoom in">+</button>
      <button id="zoom-out" title="Zoom out" aria-label="Zoom out">&minus;</button>
      <button id="reset" title="Reset view" aria-label="Reset view">&#8634;</button>
    </div>
    <div id="status"></div>
  </main>
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
const atlasImages = [];

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
group.add(lineGroup, endpointGroup, particleGroup);
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

for (const name of DATA.projections) {
  const option = document.createElement("option");
  option.value = name;
  option.textContent = name;
  projectionSelect.appendChild(option);
}
slider.max = Math.max(0, (DATA.trajectory || []).length - 1);
timeControl.style.display = DATA.mode === "trajectory" ? "grid" : "none";
populateLegend(DATA.palette || {});

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
  requestRender();
}

function buildEndpoints() {
  visibleIndices = DATA.points
    .map((point, index) => pointVisible(point) ? index : -1)
    .filter(index => index >= 0);
  if (!visibleIndices.length) return;
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
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: thumbnailToggle.checked ? DATA.options.pointSize * 0.07 : DATA.options.pointSize * 0.045,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(geometry, material);
  points.userData = { kind: "endpoint", indices: visibleIndices };
  endpointGroup.add(points);
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
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({ size: 0.72, vertexColors: true, sizeAttenuation: true });
  const points = new THREE.Points(geometry, material);
  points.userData = { kind: "trajectory", indices };
  particleGroup.add(points);
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
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = Math.max(1e-3, rect.width / Math.max(1, rect.height));
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

function showSelection(selection) {
  clearPreview();
  metricsElement.replaceChildren();
  labelElement.textContent = "-";
  indexElement.textContent = "Index: -";
  if (!selection) return;
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
    return;
  }
  const point = DATA.points[selection.index];
  drawPreview(point);
  labelElement.textContent = point.label || "-";
  indexElement.textContent = `${point.kind || "sample"} - Index: ${point.sourceIndex} - Row: ${point.rowId}`;
  const coordinate = pointCoordinate(point);
  appendMetric(metricsElement, `${projection} X`, coordinate[0]);
  appendMetric(metricsElement, `${projection} Y`, coordinate[1]);
  appendMetric(metricsElement, `${projection} Z`, coordinate[2] || 0);
  for (const [key, value] of Object.entries(point.details || {})) appendMetric(metricsElement, key, value);
  const diagnostics = (DATA.projectionDiagnostics || {})[projection] || {};
  for (const [key, values] of Object.entries(diagnostics)) appendMetric(metricsElement, key, values[selection.index]);
}

function drawPreview(point) {
  const margin = 12;
  drawPreviewTile(point, margin, margin, preview.clientWidth - margin * 2, preview.clientHeight - margin * 2);
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
renderer.domElement.addEventListener("pointerdown", event => {
  dragging = true;
  moved = false;
  pointerX = event.clientX;
  pointerY = event.clientY;
  renderer.domElement.classList.add("dragging");
});
window.addEventListener("pointermove", event => {
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
  hoverSelection = intersectSelection(event);
  if (!pinnedSelection) showSelection(hoverSelection);
  updateStatus();
});
window.addEventListener("pointerup", event => {
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

initializeClassFilter(DATA.points, onClassFilterChange);
loadAtlases(DATA.atlases).then(() => {
  resize();
  rebuildScene();
  showSelection(null);
});
'''
    return script.replace("__SHARED_SCRIPT__", shared_explorer_script()).replace(
        "__PAYLOAD_JSON__",
        payload_json,
    )
