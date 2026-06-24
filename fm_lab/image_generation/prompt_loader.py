"""Prompt file loading and normalization."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fm_lab.image_generation.config import PromptConfig
from fm_lab.utils.config import ConfigError


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    prompt: str
    negative_prompt: str | None = None
    tags: list[str] = field(default_factory=list)
    family: str | None = None
    notes: str | None = None


def load_prompts(config: PromptConfig) -> list[PromptRecord]:
    """Load JSONL or CSV prompts and apply configured filters."""

    path = Path(config.source)
    if config.format.lower() == "jsonl":
        records = _load_jsonl(path, config)
    elif config.format.lower() == "csv":
        records = _load_csv(path, config)
    else:
        raise ConfigError(f"Unsupported prompt format: {config.format!r}")

    records = _filter_prompts(records, config)
    if config.max_prompts is not None:
        records = records[: int(config.max_prompts)]
    if not records:
        raise ConfigError("No prompts remain after loading and filtering.")
    return records


def _load_jsonl(path: Path, config: PromptConfig) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ConfigError(f"JSONL row must be an object at {path}:{line_number}")
            records.append(_normalize_row(row, config, index=len(records)))
    return records


def _load_csv(path: Path, config: PromptConfig) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(_normalize_row(row, config, index=len(records)))
    return records


def _normalize_row(row: dict[str, Any], config: PromptConfig, *, index: int) -> PromptRecord:
    prompt = _optional_text(row.get(config.text_field))
    if prompt is None:
        raise ConfigError(
            f"Prompt row {index + 1} is missing required field {config.text_field!r}."
        )

    prompt_id = _optional_text(row.get(config.id_field)) or f"prompt_{index:04d}"
    return PromptRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        negative_prompt=_optional_text(row.get(config.negative_prompt_field)),
        tags=_parse_tags(row.get(config.tag_field)),
        family=_optional_text(row.get(config.family_field)),
        notes=_optional_text(row.get(config.notes_field)),
    )


def _filter_prompts(records: list[PromptRecord], config: PromptConfig) -> list[PromptRecord]:
    include_tags = set(config.include_tags)
    exclude_tags = set(config.exclude_tags)
    include_families = set(config.include_families)
    exclude_families = set(config.exclude_families)

    filtered: list[PromptRecord] = []
    for record in records:
        record_tags = set(record.tags)
        family = record.family or ""
        if include_tags and not (record_tags & include_tags):
            continue
        if exclude_tags and record_tags & exclude_tags:
            continue
        if include_families and family not in include_families:
            continue
        if exclude_families and family in exclude_families:
            continue
        filtered.append(record)
    return filtered


def _parse_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_tags = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                raw_tags = parsed
            else:
                raw_tags = _split_tag_string(stripped)
        else:
            raw_tags = _split_tag_string(stripped)
    else:
        raw_tags = [value]
    return [str(tag).strip() for tag in raw_tags if str(tag).strip()]


def _split_tag_string(value: str) -> list[str]:
    delimiter = ";"
    if "," in value:
        delimiter = ","
    elif "|" in value:
        delimiter = "|"
    return [item.strip() for item in value.split(delimiter)]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
