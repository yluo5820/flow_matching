"""Shared browser chrome for image projection explorers."""

# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass

from fm_lab.image_diagnostics.config import ExplorerConfig


@dataclass(frozen=True)
class ExplorerDocument:
    title: str
    controls_html: str
    sample_info_html: str
    footer_html: str
    script: str
    height: int
    config: ExplorerConfig
    extra_css: str = ""
    control_label_width: int = 88
    mobile_height: int = 150
    mobile_columns: str = "120px 140px minmax(0, 1fr)"
    mobile_rows: str = "1fr 1fr"
    preview_row_span: int = 2


def class_filter_html() -> str:
    return """<div class="control">
      <span class="muted">Class</span>
      <details id="class-filter" class="class-menu">
        <summary id="class-filter-summary">All classes</summary>
        <div id="class-options" class="class-options"></div>
      </details>
    </div>"""


def build_explorer_document(document: ExplorerDocument) -> str:
    """Build the common explorer page around renderer-specific JavaScript."""

    config = document.config
    metrics_display = "grid" if config.show_metrics else "none"
    legend_display = "flex" if config.show_legend else "none"
    controls_display = "grid" if config.show_view_controls else "none"
    instructions_display = "block" if config.show_instructions else "none"
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; overflow: hidden; background: #111; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #f2f2f2; }
  #app { display: grid; grid-template-columns: __SIDEBAR_WIDTH__px 1fr; height: __HEIGHT__px; background: #111; }
  #sidebar { background: #222; padding: 16px; display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .control { display: grid; grid-template-columns: __CONTROL_LABEL_WIDTH__px 1fr; align-items: center; gap: 10px; }
  label, .muted { color: #c8c8c8; font-size: 14px; }
  select { width: 100%; height: 32px; background: #f3f3f3; color: #111; border: 0; border-radius: 2px; padding: 0 8px; }
  button { border: 1px solid #444; background: #1b1b1b; color: #ddd; cursor: pointer; }
  button:hover { background: #292929; }
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
  #sample-info { min-height: 88px; display: grid; gap: 5px; align-content: start; }
  #sample-label { font-size: 24px; font-weight: 650; }
  #sample-index { color: #a9a9a9; font-variant-numeric: tabular-nums; }
  #metrics { display: __METRICS_DISPLAY__; grid-template-columns: minmax(0, 1fr) auto; gap: 6px 10px; margin-top: 8px; padding-top: 10px; border-top: 1px solid #3a3a3a; font-size: 12px; color: #bdbdbd; }
  .metrics-heading { grid-column: 1 / -1; color: #f0f0f0; font-size: 13px; font-weight: 600; margin-bottom: 2px; }
  .metric-key { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #bdbdbd; }
  .metric-value { color: #eee; font-variant-numeric: tabular-nums; text-align: right; }
  #legend { display: __LEGEND_DISPLAY__; flex-wrap: wrap; gap: 6px 10px; }
  .legend-item { display: inline-flex; align-items: center; gap: 5px; font-size: 12px; color: #cfcfcf; }
  .swatch { width: 9px; height: 9px; }
  #sidebar-footer { display: __INSTRUCTIONS_DISPLAY__; margin-top: auto; color: #aaa; font-size: 12px; line-height: 1.45; }
  #main { position: relative; min-width: 0; overflow: hidden; background: #111; }
  #plot { position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; }
  #plot.dragging { cursor: grabbing; }
  #status { position: absolute; right: 14px; bottom: 12px; color: #777; font-size: 12px; pointer-events: none; }
  #view-controls { position: absolute; right: 14px; top: 12px; display: __CONTROLS_DISPLAY__; gap: 5px; }
  #view-controls button { width: 34px; height: 34px; font-size: 20px; }
  @media (max-width: 760px) {
    #app { grid-template-columns: 1fr; grid-template-rows: 1fr __MOBILE_HEIGHT__px; }
    #sidebar { grid-row: 2; display: grid; grid-template-columns: __MOBILE_COLUMNS__; grid-template-rows: __MOBILE_ROWS__; gap: 8px 10px; padding: 10px; overflow: hidden; }
    #sidebar > .control:nth-of-type(1) { grid-column: 2; grid-row: 1; }
    #sidebar > .control:nth-of-type(2) { grid-column: 2; grid-row: 2; }
    #preview-wrap { grid-column: 1; grid-row: 1 / span __PREVIEW_ROW_SPAN__; }
    #sample-info { grid-column: 3; grid-row: 1 / span __PREVIEW_ROW_SPAN__; overflow-y: auto; }
    #legend, #sidebar-footer { display: none; }
    #main { grid-row: 1; }
    .control { grid-template-columns: 1fr; gap: 4px; align-content: start; }
    #sample-label { font-size: 20px; }
  }
__EXTRA_CSS__
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
__CONTROLS_HTML__
    <div id="preview-wrap"><canvas id="preview"></canvas></div>
    <div id="sample-info">
__SAMPLE_INFO_HTML__
    </div>
    <div id="legend"></div>
    <div id="sidebar-footer">
__FOOTER_HTML__
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
__SCRIPT__
</script>
</body>
</html>"""
    replacements = {
        "__TITLE__": document.title,
        "__SIDEBAR_WIDTH__": str(int(config.sidebar_width)),
        "__HEIGHT__": str(int(document.height)),
        "__CONTROL_LABEL_WIDTH__": str(int(document.control_label_width)),
        "__METRICS_DISPLAY__": metrics_display,
        "__LEGEND_DISPLAY__": legend_display,
        "__INSTRUCTIONS_DISPLAY__": instructions_display,
        "__CONTROLS_DISPLAY__": controls_display,
        "__MOBILE_HEIGHT__": str(int(document.mobile_height)),
        "__MOBILE_COLUMNS__": document.mobile_columns,
        "__MOBILE_ROWS__": document.mobile_rows,
        "__PREVIEW_ROW_SPAN__": str(int(document.preview_row_span)),
        "__EXTRA_CSS__": document.extra_css,
        "__CONTROLS_HTML__": _indent(document.controls_html, 4),
        "__SAMPLE_INFO_HTML__": _indent(document.sample_info_html, 6),
        "__FOOTER_HTML__": _indent(document.footer_html, 6),
        "__SCRIPT__": document.script,
    }
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


def shared_explorer_script() -> str:
    """Return JavaScript helpers shared by 2D and trajectory explorers."""

    return r"""
function populateLegend(palette) {
  const legend = document.getElementById("legend");
  for (const [label, color] of Object.entries(palette)) {
    const item = document.createElement("span");
    item.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = color;
    const text = document.createElement("span");
    text.textContent = label;
    item.append(swatch, text);
    legend.appendChild(item);
  }
}

function initializeClassFilter(points, onChange) {
  const counts = new Map();
  for (const point of points) {
    counts.set(point.label, (counts.get(point.label) || 0) + 1);
  }
  const labels = Array.from(counts.keys()).sort((left, right) =>
    left.localeCompare(right, undefined, { numeric: true, sensitivity: "base" })
  );
  selectedLabels = new Set(labels);
  visibleIndices = points.map((_, index) => index);
  const allInput = document.createElement("input");
  allInput.type = "checkbox";
  allInput.checked = true;
  classOptions.appendChild(createClassOption(allInput, "All classes", points.length));
  const classInputs = labels.map(label => {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = label;
    input.checked = true;
    classOptions.appendChild(createClassOption(input, label || "(empty)", counts.get(label)));
    input.addEventListener("change", () => {
      allInput.checked = classInputs.every(value => value.checked);
      syncClassFilter(points, classInputs, onChange);
    });
    return input;
  });
  allInput.addEventListener("change", () => {
    for (const input of classInputs) input.checked = allInput.checked;
    syncClassFilter(points, classInputs, onChange);
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

function syncClassFilter(points, classInputs, onChange) {
  selectedLabels = new Set(
    classInputs.filter(input => input.checked).map(input => input.value)
  );
  visibleIndices = points
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
  onChange();
}

function loadAtlases(sources) {
  return Promise.all(sources.map(source => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => { atlasImages.push(image); resolve(); };
    image.onerror = reject;
    image.src = source;
  })));
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

function clearPreview() {
  previewContext.clearRect(0, 0, preview.clientWidth, preview.clientHeight);
  previewContext.fillStyle = "#191919";
  previewContext.fillRect(0, 0, preview.clientWidth, preview.clientHeight);
}

function resizeExplorerCanvases(afterResize) {
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
  afterResize();
}

function requestDraw() {
  if (!frameRequested) {
    frameRequested = true;
    requestAnimationFrame(draw);
  }
}

function installCanvasPointerHandlers(onDragMove) {
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
      onDragMove(event, dx, dy);
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
}

function nearestVisiblePoint(mouseX, mouseY, threshold, screenForIndex) {
  let best = null;
  let bestDistance = threshold * threshold;
  for (const index of visibleIndices) {
    const screen = screenForIndex(index);
    const screenX = Array.isArray(screen) ? screen[0] : screen.x;
    const screenY = Array.isArray(screen) ? screen[1] : screen.y;
    const dx = screenX - mouseX;
    const dy = screenY - mouseY;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best;
}

function appendMetric(container, name, value) {
  if (value === null) return;
  const key = document.createElement("span");
  key.className = "metric-key";
  key.title = name;
  key.textContent = name;
  const metric = document.createElement("span");
  metric.className = "metric-value";
  if (typeof value === "number") {
    metric.textContent = name.includes("agreement")
      ? `${(value * 100).toFixed(1)}%`
      : Number.isInteger(value)
        ? value.toLocaleString()
        : value.toFixed(3);
  } else {
    metric.textContent = String(value);
  }
  container.append(key, metric);
}
"""


def _indent(value: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in value.splitlines())
