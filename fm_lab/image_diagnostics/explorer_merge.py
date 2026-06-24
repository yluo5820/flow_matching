"""Combine aligned precomputed projections into one explorer table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fm_lab.image_diagnostics.config import diagnostics_config_from_dict
from fm_lab.image_diagnostics.projection_diagnostics import (
    compute_projection_diagnostics,
)
from fm_lab.image_diagnostics.save_utils import read_parquet, write_parquet
from fm_lab.utils.config import ConfigError, load_config, save_config


def build_combined_explorer(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Build one explorer artifact from aligned precomputed explorer tables."""

    root = Path(project_root or Path.cwd()).resolve()
    raw = load_config(config_path)
    config = diagnostics_config_from_dict(raw)
    combine = raw.get("combine")
    if not isinstance(combine, Mapping):
        raise ConfigError("Combined explorer config requires a combine mapping.")

    base_path = _resolve_path(combine.get("base_data"), root)
    sources = combine.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ConfigError("combine.sources must be a non-empty list.")
    align_on = combine.get("align_on", ["row_id"])
    if not isinstance(align_on, list) or not align_on:
        raise ConfigError("combine.align_on must be a non-empty list.")

    frame = read_parquet(base_path)
    frame = combine_explorer_tables(
        frame,
        sources,
        align_on=[str(column) for column in align_on],
        project_root=root,
    )
    projection_columns = [
        column
        for column in frame.columns
        if str(column).endswith(("_x", "_y", "_z"))
    ]
    diagnostics = compute_projection_diagnostics(
        frame[["row_id", *projection_columns]],
        frame,
        k_neighbors=config.explorer.projection_diagnostics_k,
    )
    diagnostic_columns = [
        column for column in diagnostics.columns if column != "row_id"
    ]
    frame = frame.drop(
        columns=[column for column in diagnostic_columns if column in frame],
        errors="ignore",
    ).merge(diagnostics, on="row_id", how="left", validate="one_to_one")

    output_dir = (root / config.output_dir).resolve()
    explorer_path = output_dir / "explorer" / "explorer_data.parquet"
    write_parquet(frame, explorer_path)
    save_config(raw, output_dir / "config_used.yaml")
    return explorer_path


def combine_explorer_tables(
    base: pd.DataFrame,
    sources: Sequence[Mapping[str, Any]],
    *,
    align_on: Sequence[str],
    project_root: str | Path,
) -> pd.DataFrame:
    """Merge selected projections after validating exact sample alignment."""

    result = base.reset_index(drop=True).copy()
    _validate_alignment_columns(result, align_on, label="base explorer")
    base_index = pd.MultiIndex.from_frame(result[list(align_on)])
    if base_index.has_duplicates:
        raise ValueError(f"Base explorer has duplicate alignment keys: {list(align_on)}")

    for source_config in sources:
        data_path = _resolve_path(source_config.get("data_path"), Path(project_root))
        source = read_parquet(data_path)
        _validate_alignment_columns(source, align_on, label=str(data_path))
        source_index = pd.MultiIndex.from_frame(source[list(align_on)])
        if source_index.has_duplicates:
            raise ValueError(f"Explorer source has duplicate alignment keys: {data_path}")
        if set(source_index) != set(base_index):
            missing = len(set(base_index) - set(source_index))
            extra = len(set(source_index) - set(base_index))
            raise ValueError(
                f"Explorer source does not match base samples: {data_path} "
                f"(missing={missing}, extra={extra})"
            )
        aligned = source.set_index(list(align_on)).loc[base_index].reset_index()
        if "label" in result and "label" in aligned:
            if result["label"].astype(str).tolist() != aligned["label"].astype(str).tolist():
                raise ValueError(f"Explorer source labels do not match: {data_path}")

        projections = source_config.get("projections")
        if not isinstance(projections, Mapping) or not projections:
            raise ConfigError(f"Explorer source requires projections: {data_path}")
        for source_key, destination_key in projections.items():
            _copy_projection(
                result,
                aligned,
                source_key=str(source_key),
                destination_key=str(destination_key),
                data_path=data_path,
            )
    return result


def _copy_projection(
    result: pd.DataFrame,
    source: pd.DataFrame,
    *,
    source_key: str,
    destination_key: str,
    data_path: Path,
) -> None:
    columns = [f"{source_key}_x", f"{source_key}_y"]
    if f"{source_key}_z" in source:
        columns.append(f"{source_key}_z")
    missing = [column for column in columns[:2] if column not in source]
    if missing:
        raise ValueError(
            f"Projection {source_key!r} is missing from {data_path}: {missing}"
        )
    for axis, column in zip(("x", "y", "z"), columns, strict=False):
        destination = f"{destination_key}_{axis}"
        if destination in result:
            raise ValueError(f"Duplicate combined projection column: {destination}")
        result[destination] = source[column].to_numpy()


def _validate_alignment_columns(
    frame: pd.DataFrame,
    align_on: Sequence[str],
    *,
    label: str,
) -> None:
    missing = [column for column in align_on if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing alignment columns: {missing}")


def _resolve_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ConfigError("Combined explorer paths must be non-empty strings.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Combined explorer source does not exist: {path}")
    return path
