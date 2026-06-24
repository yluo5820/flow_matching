"""Metadata rows and summary files for generated images."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any

from fm_lab.image_generation.config import BatchGenerationConfig
from fm_lab.image_generation.prompt_loader import PromptRecord
from fm_lab.image_generation.save_utils import write_jsonl_rows

SUMMARY_COLUMNS = [
    "experiment_name",
    "prompt_id",
    "family",
    "tags",
    "seed",
    "image_index",
    "status",
    "output_path",
    "runtime_seconds",
    "error",
]


def build_metadata_row(
    *,
    config: BatchGenerationConfig,
    prompt: PromptRecord,
    seed: int,
    image_index: int,
    output_path: Path,
    status: str,
    runtime_seconds: float | None,
    error: str | None,
) -> dict[str, Any]:
    """Build a complete per-image metadata record."""

    generation = config.generation
    model = config.model
    versions = dependency_versions()
    return {
        "experiment_name": config.experiment_name,
        "model_repo_id": model.repo_id,
        "prompt_id": prompt.prompt_id,
        "family": prompt.family,
        "tags": prompt.tags,
        "prompt": prompt.prompt,
        "negative_prompt": prompt.negative_prompt,
        "seed": int(seed),
        "image_index": int(image_index),
        "width": generation.width,
        "height": generation.height,
        "num_inference_steps": generation.num_inference_steps,
        "guidance_scale": generation.guidance_scale,
        "dtype": model.dtype,
        "device": model.device,
        "batch_size": generation.batch_size,
        "num_images_per_prompt": generation.num_images_per_prompt,
        "output_path": str(output_path),
        "status": status,
        "runtime_seconds": runtime_seconds,
        "timestamp": datetime.now(UTC).isoformat(),
        "diffusers_version": versions["diffusers"],
        "transformers_version": versions["transformers"],
        "torch_version": versions["torch"],
        "error": error,
    }


def append_metadata(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl_rows(output_dir / "metadata" / "per_image_metadata.jsonl", rows, mode="a")


def append_failures(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl_rows(output_dir / "metadata" / "failures.jsonl", rows, mode="a")


def reset_metadata_files(output_dir: Path) -> None:
    for path in (
        output_dir / "metadata" / "per_image_metadata.jsonl",
        output_dir / "metadata" / "failures.jsonl",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def write_summary_csv(output_dir: Path) -> Path:
    """Write summary.csv from the accumulated metadata JSONL."""

    metadata_path = output_dir / "metadata" / "per_image_metadata.jsonl"
    summary_path = output_dir / "summary.csv"
    rows = _read_jsonl(metadata_path)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            summary_row = {column: row.get(column) for column in SUMMARY_COLUMNS}
            if isinstance(summary_row.get("tags"), list):
                summary_row["tags"] = ",".join(summary_row["tags"])
            writer.writerow(summary_row)
    return summary_path


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("diffusers", "transformers", "torch"):
        try:
            versions[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
