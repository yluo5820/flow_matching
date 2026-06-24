"""Simple PIL image grids for batch generation outputs."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image


def make_image_grid(
    image_paths: list[Path],
    output_path: Path,
    *,
    tile_size: int = 256,
    padding: int = 8,
    background: tuple[int, int, int] = (255, 255, 255),
    columns: int | None = None,
) -> Path | None:
    """Create a simple thumbnail grid from existing image paths."""

    existing_paths = [path for path in image_paths if path.exists()]
    if not existing_paths:
        return None

    if columns is None:
        columns = max(1, math.ceil(math.sqrt(len(existing_paths))))
    rows = math.ceil(len(existing_paths) / columns)
    width = columns * tile_size + (columns + 1) * padding
    height = rows * tile_size + (rows + 1) * padding
    canvas = Image.new("RGB", (width, height), background)

    for index, path in enumerate(existing_paths):
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            x = padding + (index % columns) * (tile_size + padding)
            y = padding + (index // columns) * (tile_size + padding)
            x += (tile_size - thumbnail.width) // 2
            y += (tile_size - thumbnail.height) // 2
            canvas.paste(thumbnail, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
