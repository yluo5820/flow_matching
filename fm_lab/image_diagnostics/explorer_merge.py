"""Combine aligned precomputed projections into one explorer table."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from fm_lab.image_diagnostics.config import (
    ExplorerConfig,
    diagnostics_config_from_dict,
)
from fm_lab.image_diagnostics.projection_diagnostics import (
    compute_projection_diagnostics,
)
from fm_lab.image_diagnostics.projections import projection_variants
from fm_lab.image_diagnostics.save_utils import read_parquet, write_parquet
from fm_lab.utils.config import ConfigError, load_config, save_config


@dataclass(frozen=True)
class ExplorerSource:
    data_path: Path
    frame: pd.DataFrame
    explorer_config: ExplorerConfig
    feature_name: str
    projection_names: dict[str, str]


@dataclass(frozen=True)
class DiscoveredExplorerGroup:
    key: str
    label: str
    sources: tuple[ExplorerSource, ...]
    align_on: tuple[str, ...]
    sample_count: int
    projection_count: int


@dataclass(frozen=True)
class LoadedExplorerGroup:
    frame: pd.DataFrame
    data_path: Path
    explorer_config: ExplorerConfig
    projection_names: dict[str, str]
    source_paths: tuple[Path, ...]


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


def discover_explorer_groups(
    data_dir: str | Path,
) -> list[DiscoveredExplorerGroup]:
    """Discover explorer outputs and group tables with identical sample rows."""

    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Explorer data directory does not exist: {root}")
    sources = [
        _load_explorer_source(path)
        for path in _discover_data_paths(root)
    ]
    grouped: dict[tuple[tuple[str, ...], str], list[ExplorerSource]] = {}
    for source in sources:
        align_on = _preferred_alignment(source.frame)
        signature = _alignment_signature(source.frame, align_on)
        grouped.setdefault((align_on, signature), []).append(source)

    groups = []
    for index, ((align_on, signature), compatible) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: (
                -sum(_projection_count(source.frame) for source in item[1]),
                -len(item[1][0].frame),
                item[0][1],
            ),
        )
    ):
        projection_count = sum(_projection_count(source.frame) for source in compatible)
        sample_count = len(compatible[0].frame)
        labels = sorted(
            {
                _dataset_label(source.frame)
                for source in compatible
                if _dataset_label(source.frame)
            }
        )
        dataset_label = ", ".join(labels) if labels else "dataset"
        groups.append(
            DiscoveredExplorerGroup(
                key=f"group_{index}_{signature[:10]}",
                label=(
                    f"{dataset_label} · {sample_count:,} samples · "
                    f"{projection_count} views"
                ),
                sources=tuple(sorted(compatible, key=lambda value: str(value.data_path))),
                align_on=align_on,
                sample_count=sample_count,
                projection_count=projection_count,
            )
        )
    return groups


def load_discovered_explorer_group(
    group: DiscoveredExplorerGroup,
) -> LoadedExplorerGroup:
    """Merge one compatible discovered group without writing an artifact."""

    base_source = max(
        group.sources,
        key=lambda source: (
            _has_prepacked_atlas(source.frame),
            _metadata_score(source.frame),
            source.data_path.name != "explorer_data.parquet",
        ),
    )
    projection_columns = {
        column
        for source in group.sources
        for column in _projection_columns(source.frame)
    }
    result = base_source.frame.drop(
        columns=[column for column in projection_columns if column in base_source.frame],
        errors="ignore",
    ).reset_index(drop=True)
    projection_names: dict[str, str] = {}
    used_display_names: set[str] = set()
    for source in group.sources:
        aligned = _align_frame(source.frame, result, group.align_on)
        _merge_sample_columns(
            result,
            aligned,
            projection_columns=set(_projection_columns(source.frame)),
            align_on=group.align_on,
        )
        for source_key in _projection_keys(source.frame):
            dimensions = 3 if f"{source_key}_z" in source.frame else 2
            destination_key = _unique_projection_key(
                result,
                feature_name=source.feature_name,
                source_key=source_key,
                dimensions=dimensions,
                source_dir=source.data_path.parent.parent.name,
            )
            _copy_projection(
                result,
                aligned,
                source_key=source_key,
                destination_key=destination_key,
                data_path=source.data_path,
            )
            display_name = _unique_display_name(
                _automatic_display_name(
                    source,
                    source_key=source_key,
                    dimensions=dimensions,
                ),
                used_display_names,
            )
            projection_names[destination_key] = display_name
            used_display_names.add(display_name)

    projection_columns = [
        column
        for column in result.columns
        if str(column).endswith(("_x", "_y", "_z"))
    ]
    diagnostics = compute_projection_diagnostics(
        result[["row_id", *projection_columns]],
        result,
        k_neighbors=base_source.explorer_config.projection_diagnostics_k,
    )
    diagnostic_columns = [
        column for column in diagnostics.columns if column != "row_id"
    ]
    result = result.drop(
        columns=[column for column in diagnostic_columns if column in result],
        errors="ignore",
    ).merge(diagnostics, on="row_id", how="left", validate="one_to_one")
    has_3d = any(str(column).endswith("_z") for column in projection_columns)
    config_source = next(
        (
            source
            for source in group.sources
            if source.explorer_config.renderer == "three3d"
        ),
        base_source,
    )
    explorer_config = replace(
        config_source.explorer_config,
        renderer="three3d" if has_3d else "canvas2d",
        selector_label="Projection",
        show_workspace=False,
    )
    return LoadedExplorerGroup(
        frame=result,
        data_path=base_source.data_path,
        explorer_config=explorer_config,
        projection_names=projection_names,
        source_paths=tuple(source.data_path for source in group.sources),
    )


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
        aligned = _align_frame(source, result, align_on)
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


def _discover_data_paths(root: Path) -> list[Path]:
    paths = []
    for explorer_dir in sorted(root.glob("*/explorer")):
        config_path = explorer_dir.parent / "config_used.yaml"
        if config_path.exists() and "combine" in load_config(config_path):
            continue
        enhanced = sorted(explorer_dir.glob("explorer_data_with*.parquet"))
        canonical = explorer_dir / "explorer_data.parquet"
        if enhanced:
            paths.append(max(enhanced, key=lambda path: path.stat().st_mtime_ns))
        elif canonical.is_file():
            paths.append(canonical)
    return paths


def _load_explorer_source(path: Path) -> ExplorerSource:
    frame = read_parquet(path).reset_index(drop=True)
    config_path = path.parent.parent / "config_used.yaml"
    if config_path.exists():
        config = diagnostics_config_from_dict(load_config(config_path))
        names = {
            variant.key: variant.name
            for variant in projection_variants(config.projection)
        }
        explorer_config = config.explorer
        feature_name = config.features.name
    else:
        names = {}
        explorer_config = ExplorerConfig()
        feature_name = path.parent.parent.name
    return ExplorerSource(
        data_path=path,
        frame=frame,
        explorer_config=explorer_config,
        feature_name=feature_name,
        projection_names=names,
    )


def _preferred_alignment(frame: pd.DataFrame) -> tuple[str, ...]:
    for columns in (
        ("dataset", "split", "source_index"),
        ("split", "source_index"),
        ("dataset", "original_index"),
        ("original_index",),
        ("row_id",),
    ):
        if all(column in frame for column in columns):
            return columns
    raise ValueError("Explorer data has no stable sample alignment columns.")


def _alignment_signature(
    frame: pd.DataFrame,
    align_on: tuple[str, ...],
) -> str:
    values = (
        frame[list(align_on)]
        .astype(str)
        .sort_values(list(align_on), kind="stable")
        .reset_index(drop=True)
    )
    hashed = pd.util.hash_pandas_object(values, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _projection_keys(frame: pd.DataFrame) -> list[str]:
    columns = set(frame.columns)
    return [
        str(column)[:-2]
        for column in frame.columns
        if str(column).endswith("_x")
        and f"{str(column)[:-2]}_y" in columns
    ]


def _projection_columns(frame: pd.DataFrame) -> list[str]:
    return [
        f"{key}_{axis}"
        for key in _projection_keys(frame)
        for axis in ("x", "y", "z")
        if f"{key}_{axis}" in frame
    ]


def _projection_count(frame: pd.DataFrame) -> int:
    return len(_projection_keys(frame))


def _metadata_score(frame: pd.DataFrame) -> int:
    projection_columns = set(_projection_columns(frame))
    return sum(
        1
        for column in frame.columns
        if column not in projection_columns
        and not any(
            token in str(column)
            for token in (
                "_knn_radius_",
                "_knn_mean_distance_",
                "_label_agreement_",
                "_distance_to_label_centroid",
                "_nearest_",
            )
        )
    )


def _has_prepacked_atlas(frame: pd.DataFrame) -> bool:
    return {
        "sprite_atlas_path",
        "sprite_atlas_index",
        "sprite_atlas_column",
        "sprite_atlas_row",
        "sprite_tile_size",
        "sprite_atlas_columns",
    } <= set(frame.columns)


def _merge_sample_columns(
    result: pd.DataFrame,
    source: pd.DataFrame,
    *,
    projection_columns: set[str],
    align_on: Sequence[str],
) -> None:
    excluded = {
        *align_on,
        "row_id",
        "feature_name",
        "feature_mode",
        "feature_source_id",
        "features_normalized",
        "feature_fingerprint",
    }
    for column in source.columns:
        if column in excluded or column in projection_columns:
            continue
        if column not in result:
            result[column] = source[column].to_numpy()
            continue
        existing = result[column]
        missing = existing.isna()
        if existing.dtype == object:
            missing = missing | existing.astype(str).eq("")
        if missing.any():
            result.loc[missing, column] = source.loc[missing, column].to_numpy()


def _dataset_label(frame: pd.DataFrame) -> str:
    if "dataset" not in frame or frame.empty:
        return ""
    values = sorted(set(frame["dataset"].dropna().astype(str)))
    return "/".join(values[:3])


def _unique_projection_key(
    frame: pd.DataFrame,
    *,
    feature_name: str,
    source_key: str,
    dimensions: int,
    source_dir: str,
) -> str:
    source_base = re.sub(r"_[23]d$", "", source_key, flags=re.IGNORECASE)
    base = _slug(f"{feature_name}_{source_base}_{dimensions}d")
    candidate = base
    if f"{candidate}_x" in frame:
        candidate = _slug(f"{source_dir}_{source_base}_{dimensions}d")
    suffix = 2
    while f"{candidate}_x" in frame:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _automatic_display_name(
    source: ExplorerSource,
    *,
    source_key: str,
    dimensions: int,
) -> str:
    feature = _feature_display_name(source.feature_name)
    projection = source.projection_names.get(
        source_key,
        source_key.replace("_", " ").upper(),
    )
    projection = re.sub(r"\s+[23]D$", "", projection, flags=re.IGNORECASE)
    return f"{feature} - {projection} {dimensions}D"


def _unique_display_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    suffix = 2
    while f"{name} ({suffix})" in used:
        suffix += 1
    return f"{name} ({suffix})"


def _feature_display_name(value: str) -> str:
    lower = value.lower()
    if "dinov2" in lower:
        return "DINOv2"
    if "raw" in lower or "pixel" in lower:
        return "Raw Pixels"
    return value.replace("_", " ").title()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


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


def _align_frame(
    source: pd.DataFrame,
    base: pd.DataFrame,
    align_on: Sequence[str],
) -> pd.DataFrame:
    order = base[list(align_on)].copy()
    order["__auto_explorer_order"] = range(len(order))
    aligned = order.merge(
        source,
        on=list(align_on),
        how="left",
        validate="one_to_one",
    )
    return (
        aligned.sort_values("__auto_explorer_order")
        .drop(columns="__auto_explorer_order")
        .reset_index(drop=True)
    )


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
