"""Shared categorical colors used by the thumbnail explorer."""

from __future__ import annotations

import colorsys

LABEL_PALETTE = [
    (141, 211, 199),
    (255, 255, 179),
    (190, 186, 218),
    (251, 128, 114),
    (128, 177, 211),
    (253, 180, 98),
    (179, 222, 105),
    (252, 205, 229),
    (188, 128, 189),
    (204, 235, 197),
]


def categorical_palette(labels: list[str] | tuple[str, ...]) -> dict[str, tuple[int, int, int]]:
    """Return stable label colors, preserving the original first ten colors."""

    return {label: categorical_color(index) for index, label in enumerate(labels)}


def categorical_color(index: int) -> tuple[int, int, int]:
    """Return a deterministic categorical RGB color for any non-negative index."""

    if index < len(LABEL_PALETTE):
        return LABEL_PALETTE[index]
    offset = index - len(LABEL_PALETTE)
    hue = (0.08 + offset * 0.618033988749895) % 1.0
    lightness = (0.56, 0.68, 0.48)[offset % 3]
    saturation = (0.58, 0.72, 0.64)[(offset // 3) % 3]
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (
        int(round(red * 255)),
        int(round(green * 255)),
        int(round(blue * 255)),
    )
