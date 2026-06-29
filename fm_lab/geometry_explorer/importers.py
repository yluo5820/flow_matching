"""Compatibility importers for existing explorer and run outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.trajectories import register_completed_run
from fm_lab.image_diagnostics.config import diagnostics_config_from_dict
from fm_lab.image_diagnostics.projections import projection_variants
from fm_lab.image_diagnostics.save_utils import read_parquet
from fm_lab.utils.config import load_config


def import_existing_outputs(
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
    dataset_root: str | Path = "outputs/dataset_explorer",
    runs_root: str | Path = "runs",
) -> dict[str, int]:
    """Index existing dataset explorer outputs and training runs."""

    registry = GeometryRegistry(workspace)
    dataset_count = import_dataset_explorers(registry, dataset_root)
    run_count = import_training_runs(registry, runs_root)
    return {"dataset_views": dataset_count, "runs": run_count}


def import_dataset_explorers(
    registry: GeometryRegistry,
    root: str | Path,
) -> int:
    base = Path(root).expanduser()
    if not base.exists():
        return 0
    count = 0
    for explorer_dir in sorted(base.glob("*/explorer")):
        try:
            data_path = _select_data_path(explorer_dir)
            frame = read_parquet(data_path)
        except Exception:
            continue
        output_dir = explorer_dir.parent
        config_path = output_dir / "config_used.yaml"
        raw = load_config(config_path) if config_path.exists() else {}
        family = _family_from_frame(frame, raw)
        variant = str(raw.get("variant", "original"))
        variant_id = f"{family}/{variant}"
        label_counts = _label_counts(frame)
        dataset_path = output_dir / "dataset_index.parquet"
        if not dataset_path.exists():
            dataset_path = data_path
        registry.register_dataset_variant(
            variant_id=variant_id,
            family=family,
            variant=variant,
            base="legacy",
            split=str(raw.get("input", {}).get("split", "")),
            dataset_path=dataset_path,
            data_path=None,
            labels_path=None,
            config_path=config_path if config_path.exists() else None,
            row_count=len(frame),
            label_counts=label_counts,
            image_shape=None,
            value_range=None,
        )
        projection_names, feature_name, feature_mode, renderer = _view_metadata(raw)
        view_id = _legacy_view_id(variant_id, output_dir.name, feature_name)
        registry.register_projection_view(
            view_id=view_id,
            variant_id=variant_id,
            feature_name=feature_name,
            feature_mode=feature_mode,
            explorer_data_path=data_path,
            output_dir=output_dir,
            projection_names=projection_names,
            renderer=renderer,
            row_count=len(frame),
        )
        count += 1
    return count


def import_training_runs(
    registry: GeometryRegistry,
    root: str | Path,
) -> int:
    base = Path(root).expanduser()
    if not base.exists():
        return 0
    count = 0
    for config_path in sorted(base.glob("**/config.yaml")):
        run_dir = config_path.parent
        try:
            register_completed_run(run_dir, workspace=registry.workspace)
        except Exception:
            continue
        count += 1
    return count


def _family_from_frame(frame: pd.DataFrame, raw: dict[str, Any]) -> str:
    input_type = str(raw.get("input", {}).get("type", "")).lower()
    if input_type:
        if input_type == "fashion_mnist":
            return "fashion_mnist"
        if input_type == "cifar10":
            color_mode = str(raw.get("input", {}).get("color_mode", "rgb"))
            return "cifar10_grayscale" if color_mode == "grayscale" else "cifar10"
        return input_type
    if "dataset" in frame and len(frame):
        return str(frame["dataset"].iloc[0])
    return "dataset"


def _select_data_path(explorer_dir: Path) -> Path:
    enhanced = sorted(explorer_dir.glob("explorer_data_with*.parquet"))
    if enhanced:
        return enhanced[-1]
    return explorer_dir / "explorer_data.parquet"


def _view_metadata(raw: dict[str, Any]) -> tuple[dict[str, str], str, str, str]:
    try:
        config = diagnostics_config_from_dict(raw)
    except Exception:
        return {}, "features", "unknown", "three3d"
    projection_names = {
        variant.key: variant.name
        for variant in projection_variants(config.projection)
    }
    return (
        projection_names,
        config.features.name,
        config.features.mode,
        config.explorer.renderer,
    )


def _label_counts(frame: pd.DataFrame) -> dict[str, int]:
    if "label" not in frame:
        return {}
    counts = frame["label"].astype(str).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _legacy_view_id(variant_id: str, name: str, feature_name: str) -> str:
    return f"legacy__{variant_id}__{feature_name}__{name}".replace("/", "__")
