"""Load and validate image metadata from batch-generation experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.utils.config import ConfigError

LOGGER = logging.getLogger("fm_lab.image_diagnostics")

INDEX_COLUMNS = [
    "row_id",
    "image_path",
    "prompt_id",
    "family",
    "tags",
    "prompt",
    "seed",
    "image_index",
    "model_repo_id",
    "status",
]


@dataclass(frozen=True)
class MetadataLoadResult:
    frame: pd.DataFrame
    metadata_path: Path
    total_rows: int
    status_included_rows: int
    duplicate_rows: int
    missing_images: int
    malformed_rows: int


def load_image_metadata(
    config: InputConfig,
    *,
    project_root: str | Path | None = None,
) -> MetadataLoadResult:
    """Read metadata, filter statuses, resolve image paths, and skip missing files."""

    root = Path(project_root or Path.cwd()).expanduser().resolve()
    experiment_dir = _resolve_from_root(config.experiment_dir, root)
    metadata_path = Path(config.metadata_path).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = experiment_dir / metadata_path
    metadata_path = metadata_path.resolve()
    if not metadata_path.exists():
        raise ConfigError(f"Image metadata file does not exist: {metadata_path}")

    raw_rows, malformed_rows = _read_rows(metadata_path)
    total_rows = len(raw_rows) + malformed_rows
    include_status = {str(value) for value in config.include_status}
    included = [
        row
        for row in raw_rows
        if str(row.get("status") or "success") in include_status
    ]
    status_included_rows = len(included)

    seen_paths: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    duplicate_rows = 0
    missing_images = 0
    for row in included:
        raw_path = row.get("output_path") or row.get("image_path")
        if not raw_path:
            missing_images += 1
            LOGGER.warning("Skipping metadata row without output_path/image_path.")
            continue
        image_path = resolve_image_path(
            str(raw_path),
            experiment_dir=experiment_dir,
            image_root=config.image_root,
            metadata_path=metadata_path,
            project_root=root,
            prompt_id=str(row.get("prompt_id", "")),
        )
        if image_path is None:
            missing_images += 1
            LOGGER.warning("Skipping missing image path: %s", raw_path)
            continue
        key = str(image_path)
        if key in seen_paths:
            duplicate_rows += 1
            continue
        seen_paths.add(key)
        normalized_rows.append(_normalize_row(row, image_path))

    frame = pd.DataFrame(normalized_rows)
    if frame.empty:
        frame = pd.DataFrame(columns=INDEX_COLUMNS)
    else:
        frame.insert(0, "row_id", range(len(frame)))
        extra_columns = [
            column for column in frame.columns if column not in INDEX_COLUMNS
        ]
        ordered = INDEX_COLUMNS + extra_columns
        frame = frame.loc[:, ordered]

    return MetadataLoadResult(
        frame=frame,
        metadata_path=metadata_path,
        total_rows=total_rows,
        status_included_rows=status_included_rows,
        duplicate_rows=duplicate_rows,
        missing_images=missing_images,
        malformed_rows=malformed_rows,
    )


def resolve_image_path(
    raw_path: str,
    *,
    experiment_dir: Path,
    image_root: str,
    metadata_path: Path,
    project_root: Path,
    prompt_id: str = "",
) -> Path | None:
    """Resolve generator paths written as either absolute or project-relative values."""

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve() if path.is_file() else None

    root_path = Path(image_root).expanduser()
    if not root_path.is_absolute():
        root_path = experiment_dir / root_path
    candidates = [
        project_root / path,
        experiment_dir / path,
        root_path / path,
        metadata_path.parent / path,
    ]
    if prompt_id:
        candidates.append(root_path / prompt_id / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        return frame.where(pd.notna(frame), None).to_dict(orient="records"), 0
    if suffix not in {".jsonl", ".json"}:
        raise ConfigError(f"Metadata must be JSONL or CSV: {path}")

    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                malformed += 1
                LOGGER.warning("Skipping malformed JSONL row %d in %s", line_number, path)
                continue
            if not isinstance(value, dict):
                malformed += 1
                LOGGER.warning("Skipping non-object JSONL row %d in %s", line_number, path)
                continue
            rows.append(value)
    return rows, malformed


def _normalize_row(row: dict[str, Any], image_path: Path) -> dict[str, Any]:
    normalized = dict(row)
    normalized["image_path"] = str(image_path)
    normalized["prompt_id"] = str(row.get("prompt_id") or "")
    normalized["family"] = str(row.get("family") or "")
    normalized["tags"] = _normalize_tags(row.get("tags"))
    normalized["prompt"] = str(row.get("prompt") or "")
    normalized["model_repo_id"] = str(row.get("model_repo_id") or "")
    normalized["status"] = str(row.get("status") or "success")
    normalized["seed"] = _optional_int(row.get("seed"))
    normalized["image_index"] = _optional_int(row.get("image_index"))
    return normalized


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [str(value)]


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_from_root(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()
