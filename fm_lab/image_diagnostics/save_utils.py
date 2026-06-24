"""Filesystem and table helpers for diagnostics outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fm_lab.image_diagnostics.config import DiagnosticsRunConfig
from fm_lab.utils.config import save_config

if TYPE_CHECKING:
    import pandas as pd


class OptionalDependencyError(RuntimeError):
    """Raised when an optional diagnostics dependency is unavailable."""


def prepare_output_dir(config: DiagnosticsRunConfig) -> Path:
    """Create the diagnostics output tree and persist the effective config."""

    output_dir = config.output_dir
    for relative in ("features", "projections", "diagnostics", "explorer", "assets"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    save_config(config.raw, output_dir / "config_used.yaml")
    return output_dir


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("fm_lab.image_diagnostics")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(output_dir / "run_log.txt", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a Parquet table with an actionable missing-engine error."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise OptionalDependencyError(
            "Writing diagnostics tables requires pyarrow. "
            'Install with: python -m pip install -e ".[image-diagnostics]"'
        ) from exc
    return output_path


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet table with an actionable missing-engine error."""

    import pandas as pd

    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise OptionalDependencyError(
            "Reading diagnostics tables requires pyarrow. "
            'Install with: python -m pip install -e ".[image-diagnostics]"'
        ) from exc
