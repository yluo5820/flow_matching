"""End-to-end generic dataset explorer build runner."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from fm_lab.image_diagnostics.config import DiagnosticsRunConfig, FeatureConfig
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle, load_dataset
from fm_lab.image_diagnostics.explorer_data import build_explorer_data
from fm_lab.image_diagnostics.feature_models import ImageFeatureExtractor, load_feature_model
from fm_lab.image_diagnostics.feature_runner import FeatureResult, compute_or_load_features
from fm_lab.image_diagnostics.id_config import load_id_config
from fm_lab.image_diagnostics.id_runner import run_id_estimation
from fm_lab.image_diagnostics.label_store import ensure_label_store
from fm_lab.image_diagnostics.local_diagnostics import compute_or_load_local_diagnostics
from fm_lab.image_diagnostics.projection_diagnostics import (
    compute_projection_diagnostics,
)
from fm_lab.image_diagnostics.projections import compute_or_load_projections
from fm_lab.image_diagnostics.save_utils import (
    configure_logging,
    prepare_output_dir,
    write_parquet,
)


def run_diagnostics_build(
    config: DiagnosticsRunConfig,
    *,
    project_root: str | Path | None = None,
    dry_run: bool = False,
    model_loader: Callable[[FeatureConfig], ImageFeatureExtractor] = load_feature_model,
) -> dict[str, Any]:
    """Build features, projections, diagnostics, and explorer data."""

    configured_id = None
    id_config_path = None
    if config.id_estimation.enabled:
        id_config_path = _resolve_project_path(
            config.id_estimation.config_path,
            project_root=project_root,
        )
        configured_id = load_id_config(id_config_path)
    if dry_run:
        dataset = load_dataset(config.input, project_root=project_root)
        return _dry_run_result(config, dataset)

    output_dir = prepare_output_dir(config)
    logger = configure_logging(output_dir)
    dataset = load_dataset(
        config.input,
        project_root=project_root,
        thumbnail_dir=output_dir / "assets" / "thumbnails",
    )
    if dataset.metadata.empty:
        raise RuntimeError("No samples remain after loading and filtering the dataset.")
    _log_dataset_summary(logger, dataset)
    started = time.perf_counter()

    write_parquet(dataset.metadata, output_dir / "dataset_index.parquet")
    feature_output_dir = _feature_output_dir(
        config,
        project_root=project_root,
        default=output_dir,
    )
    feature_result: FeatureResult = compute_or_load_features(
        config=config.features,
        dataset=dataset,
        output_dir=feature_output_dir,
        save=config.output.save_features,
        model_loader=model_loader,
    )

    projection_config = replace(
        config.projection,
        skip_existing=(
            config.projection.skip_existing and feature_result.loaded_from_cache
        ),
    )
    projections = compute_or_load_projections(
        feature_result.features,
        feature_result.metadata["row_id"],
        projection_config,
        output_dir,
        feature_name=config.features.name,
        save=config.output.save_projection,
        project_root=project_root,
    )

    diagnostics_config = replace(
        config.diagnostics,
        skip_existing=(
            config.diagnostics.skip_existing and feature_result.loaded_from_cache
        ),
    )
    if diagnostics_config.enabled:
        diagnostics = compute_or_load_local_diagnostics(
            feature_result.features,
            feature_result.metadata,
            diagnostics_config,
            output_dir,
            feature_name=config.features.name,
            save=config.output.save_diagnostics,
        )
    else:
        diagnostics = feature_result.metadata[["row_id"]].copy()
        logger.info("Local diagnostics disabled.")

    explorer_path = output_dir / "explorer" / "explorer_data.parquet"
    labels_path = ensure_label_store(output_dir / "explorer" / "manual_labels.csv")
    if config.explorer.enabled and config.output.save_explorer_data:
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
        logger.info("Saved explorer data: %s", explorer_path)

    id_estimation = None
    if config.id_estimation.enabled:
        if not explorer_path.exists():
            raise RuntimeError(
                "Intrinsic-dimension estimation requires saved explorer data."
            )
        assert configured_id is not None
        configured_id = replace(
            configured_id,
            input=replace(
                configured_id.input,
                diagnostics_dir=str(output_dir.resolve()),
            ),
        )
        logger.info("Running intrinsic-dimension estimation: %s", id_config_path)
        id_estimation = run_id_estimation(
            configured_id,
            project_root=project_root,
        )
        logger = configure_logging(output_dir)
        logger.info(
            "Saved explorer with intrinsic-dimension estimates: %s",
            id_estimation.get("merged_explorer_path"),
        )

    elapsed = time.perf_counter() - started
    logger.info("Finished dataset explorer build in %.2f seconds.", elapsed)
    return {
        "dry_run": False,
        "explorer_name": config.explorer_name,
        "output_dir": output_dir,
        "samples": len(dataset.metadata),
        "feature_name": feature_result.feature_name,
        "feature_rows": len(feature_result.features),
        "feature_dimension": feature_result.features.shape[1],
        "projection_rows": len(projections),
        "diagnostic_rows": len(diagnostics),
        "explorer_data": explorer_path if explorer_path.exists() else None,
        "manual_labels": labels_path,
        "id_estimation": id_estimation,
        "runtime_seconds": elapsed,
    }


def _feature_output_dir(
    config: DiagnosticsRunConfig,
    *,
    project_root: str | Path | None,
    default: Path,
) -> Path:
    if not config.features.cache_dir:
        return default
    path = Path(config.features.cache_dir).expanduser()
    if not path.is_absolute():
        path = Path(project_root or Path.cwd()) / path
    return path.resolve()


def _resolve_project_path(
    value: str | None,
    *,
    project_root: str | Path | None,
) -> Path:
    if not value:
        raise RuntimeError("Configured path is empty.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(project_root or Path.cwd()) / path
    return path.resolve()


def _dry_run_result(
    config: DiagnosticsRunConfig,
    dataset: DatasetBundle,
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "explorer_name": config.explorer_name,
        "input_type": config.input.type,
        "source": dataset.source_description,
        "total_rows": dataset.total_rows,
        "selected_samples": len(dataset.metadata),
        "skipped_rows": dataset.skipped_rows,
        "feature_mode": config.features.mode,
        "feature_name": config.features.name,
        "requires_model_download": config.features.mode == "dinov2",
        "projection_method": config.projection.method,
        "k_neighbors": config.diagnostics.k_neighbors,
        "id_estimation_enabled": config.id_estimation.enabled,
        "id_estimation_config": config.id_estimation.config_path,
        "output_dir": config.output_dir,
    }


def _log_dataset_summary(logger: logging.Logger, dataset: DatasetBundle) -> None:
    logger.info("Dataset source: %s", dataset.source_description)
    logger.info("Source rows: %d", dataset.total_rows)
    logger.info("Selected samples: %d", len(dataset.metadata))
    logger.info("Skipped rows: %d", dataset.skipped_rows)
