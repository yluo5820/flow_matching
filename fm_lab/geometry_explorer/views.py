"""Build and register projection views for geometry explorer variants."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.variants import load_variant_bundle
from fm_lab.image_diagnostics.config import (
    diagnostics_config_from_dict,
)
from fm_lab.image_diagnostics.explorer_data import build_explorer_data
from fm_lab.image_diagnostics.feature_runner import compute_or_load_features
from fm_lab.image_diagnostics.label_store import ensure_label_store
from fm_lab.image_diagnostics.local_diagnostics import compute_or_load_local_diagnostics
from fm_lab.image_diagnostics.projection_diagnostics import compute_projection_diagnostics
from fm_lab.image_diagnostics.projections import compute_or_load_projections, projection_variants
from fm_lab.image_diagnostics.save_utils import (
    configure_logging,
    prepare_output_dir,
    write_parquet,
)
from fm_lab.utils.config import deep_update, load_config


def build_projection_view(
    *,
    variant_id: str,
    config_path: str | Path,
    workspace: str | Path = DEFAULT_WORKSPACE,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a projection/diagnostics explorer table for a registered variant."""

    registry = GeometryRegistry(workspace)
    family, variant = _split_variant_id(variant_id)
    variant_row = registry.get_dataset_variant(variant_id)
    raw = load_config(config_path)
    explorer_name = str(raw.get("explorer_name", f"{family}_{variant}_view"))
    output_root = registry.workspace / "datasets" / family / variant / "views"
    raw = deep_update(
        raw,
        {
            "explorer_name": explorer_name,
            "output": {"root_dir": str(output_root)},
            "input": _variant_input_override(registry, variant_row),
            "id_estimation": {"enabled": False},
        },
    )
    config = diagnostics_config_from_dict(raw)
    output_dir = prepare_output_dir(config)
    logger = configure_logging(output_dir)
    dataset = load_variant_bundle(variant_id, workspace=workspace)
    if dataset.vectors is None:
        raise RuntimeError(f"Dataset variant {variant_id} has no feature vectors.")
    started = time.perf_counter()

    write_parquet(dataset.metadata, output_dir / "dataset_index.parquet")
    feature_result = compute_or_load_features(
        config=config.features,
        dataset=dataset,
        output_dir=output_dir,
        save=config.output.save_features,
    )
    projections = compute_or_load_projections(
        feature_result.features,
        feature_result.metadata["row_id"],
        config.projection,
        output_dir,
        feature_name=config.features.name,
        save=config.output.save_projection,
        project_root=project_root,
    )
    diagnostics = (
        compute_or_load_local_diagnostics(
            feature_result.features,
            feature_result.metadata,
            replace(
                config.diagnostics,
                skip_existing=config.diagnostics.skip_existing
                and feature_result.loaded_from_cache,
            ),
            output_dir,
            feature_name=config.features.name,
            save=config.output.save_diagnostics,
        )
        if config.diagnostics.enabled
        else feature_result.metadata[["row_id"]].copy()
    )
    explorer_path = output_dir / "explorer" / "explorer_data.parquet"
    labels_path = ensure_label_store(output_dir / "explorer" / "manual_labels.csv")
    explorer_data = build_explorer_data(
        feature_result.metadata,
        projections,
        diagnostics,
        labels_path=labels_path,
    )
    if config.explorer.compute_projection_diagnostics:
        projection_diagnostics = compute_projection_diagnostics(
            projections,
            feature_result.metadata,
            k_neighbors=config.explorer.projection_diagnostics_k,
        )
        explorer_data = explorer_data.merge(
            projection_diagnostics,
            on="row_id",
            how="left",
            validate="one_to_one",
        )
    write_parquet(explorer_data, explorer_path)

    projection_names = {
        variant_config.key: variant_config.name
        for variant_config in projection_variants(config.projection)
    }
    view_id = _view_id(variant_id, explorer_name, config.features.name)
    registry.register_projection_view(
        view_id=view_id,
        variant_id=variant_id,
        feature_name=config.features.name,
        feature_mode=config.features.mode,
        explorer_data_path=explorer_path,
        output_dir=output_dir,
        projection_names=projection_names,
        renderer=config.explorer.renderer,
        row_count=len(explorer_data),
    )
    elapsed = time.perf_counter() - started
    logger.info("Registered geometry projection view %s in %.2f seconds.", view_id, elapsed)
    return {
        "view_id": view_id,
        "variant_id": variant_id,
        "output_dir": output_dir,
        "explorer_data": explorer_path,
        "rows": len(explorer_data),
        "projection_names": projection_names,
        "runtime_seconds": elapsed,
    }


def _split_variant_id(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise ValueError("Dataset variant must be formatted as family/variant.")
    family, variant = value.split("/", 1)
    return family, variant


def _view_id(variant_id: str, explorer_name: str, feature_name: str) -> str:
    slug = f"{variant_id}/{feature_name}/{explorer_name}"
    return slug.replace("/", "__").replace(" ", "_")


def _variant_input_override(registry: GeometryRegistry, row: Any) -> dict[str, Any]:
    values: dict[str, Any] = {"type": "numpy"}
    if row["data_path"]:
        values["data_path"] = str(registry.resolve(row["data_path"]))
    if row["labels_path"]:
        values["labels_path"] = str(registry.resolve(row["labels_path"]))
    if row["image_shape_json"]:
        values["image_shape"] = json.loads(row["image_shape_json"])
    if row["value_range_json"]:
        values["value_range"] = json.loads(row["value_range_json"])
    return values
