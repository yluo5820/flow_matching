"""Filesystem helpers for image generation outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fm_lab.image_generation.config import BatchGenerationConfig
from fm_lab.utils.config import save_config


def prepare_output_dir(
    config: BatchGenerationConfig,
    *,
    config_path: str | Path | None = None,
) -> Path:
    """Create the experiment output tree and persist the exact config used."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "grids").mkdir(exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)

    del config_path
    save_config(config.raw, output_dir / "config_used.yaml")
    return output_dir


def output_image_path(
    output_dir: Path,
    prompt_id: str,
    seed: int,
    image_index: int,
) -> Path:
    """Return the canonical per-image output path."""

    safe_prompt_id = safe_path_component(prompt_id)
    filename = f"{safe_prompt_id}_seed{int(seed):04d}_idx{int(image_index):02d}.png"
    return output_dir / "images" / safe_prompt_id / filename


def prompt_grid_path(output_dir: Path, prompt_id: str) -> Path:
    safe_prompt_id = safe_path_component(prompt_id)
    return output_dir / "images" / safe_prompt_id / f"{safe_prompt_id}_grid.png"


def family_grid_path(output_dir: Path, family: str) -> Path:
    safe_family = safe_path_component(family or "unknown_family")
    return output_dir / "grids" / f"family_{safe_family}_grid.png"


def safe_path_component(value: str) -> str:
    """Make a stable filesystem component from a prompt or family id."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


def write_manifest(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "manifest.jsonl"
    write_jsonl_rows(path, rows, mode="w")
    return path


def write_jsonl_rows(path: Path, rows: list[dict[str, Any]], *, mode: str = "a") -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
