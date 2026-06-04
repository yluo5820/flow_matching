"""YAML configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment config is missing or malformed."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file as a dictionary."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level config must be a mapping: {config_path}")
    return data


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Save a config dictionary as stable, human-readable YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `updates` onto `base` without mutating either input."""

    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        old_value = merged.get(key)
        if isinstance(old_value, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_update(old_value, value)
        else:
            merged[key] = value
    return merged
