"""Image loading helpers that isolate corrupt files from embedding batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_rgb_image(path: str | Path) -> Any:
    """Load an image as a detached RGB PIL image."""

    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()
