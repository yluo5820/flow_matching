"""Runner for regular representation-space intrinsic dimension diagnostics."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fm_lab.image_diagnostics.id_config import IDEstimationConfig
from fm_lab.image_diagnostics.id_estimators import (
    ScalingEstimate,
    compute_global_id,
    compute_local_id,
)
from fm_lab.image_diagnostics.id_feature_loader import (
    IDFeatureBundle,
    inspect_id_input,
    load_id_features,
)
from fm_lab.image_diagnostics.save_utils import (
    configure_logging,
    read_parquet,
    write_parquet,
)
from fm_lab.utils.config import save_config

LOGGER = logging.getLogger("fm_lab.image_diagnostics")


def run_id_estimation(
    config: IDEstimationConfig,
    *,
    project_root: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or inspect a configured intrinsic-dimension analysis."""

    root = Path(project_root or Path.cwd()).resolve()
    output_dir = _resolve_output_dir(config, root)
    if dry_run:
        summary = inspect_id_input(config, project_root=root)
        return {
            "id_estimation_name": config.id_estimation_name,
            "feature_space": config.input.feature_space_name,
            "source_type": config.input.source_type,
            "feature_shape": summary.feature_shape,
            "metadata_rows": summary.metadata_rows,
            "feature_path": summary.feature_path,
            "explorer_path": summary.explorer_path,
            "group_sizes": summary.group_sizes,
            "local_estimators": _enabled_local_estimators(config),
            "global_estimators": _enabled_global_estimators(config),
            "k_values": list(config.local_id.k_values),
            "output_dir": output_dir,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config.raw, output_dir / "config_used.yaml")
    logger = configure_logging(output_dir)
    started = time.perf_counter()
    bundle = load_id_features(config, project_root=root)
    logger.info(
        "ID feature space: %s; rows: %d; dimension: %d",
        bundle.feature_space,
        len(bundle.features),
        bundle.features.shape[1],
    )
    _save_processed_features(bundle, config, output_dir)

    id_dir = output_dir / "intrinsic_dimension"
    id_dir.mkdir(parents=True, exist_ok=True)
    local_path = id_dir / f"local_id_{bundle.feature_space}.parquet"
    group_path = id_dir / f"group_id_{bundle.feature_space}.csv"

    local = _compute_or_load_local(
        bundle,
        config,
        local_path=local_path,
    )
    group, curves = _compute_or_load_groups(
        bundle,
        local,
        config,
        group_path=group_path,
    )
    if config.output.save_group_id and curves:
        _save_scaling_curves(curves, id_dir / "id_curves")

    merged_path = None
    if config.output.merge_into_explorer_data and not local.empty:
        merged_path = merge_id_into_explorer(
            bundle.explorer_path,
            local,
            feature_space=bundle.feature_space,
            config=config,
        )

    elapsed = time.perf_counter() - started
    manifest = {
        "id_estimation_name": config.id_estimation_name,
        "feature_space": bundle.feature_space,
        "feature_shape": list(bundle.features.shape),
        "source_path": str(bundle.source_path) if bundle.source_path else None,
        "explorer_path": str(bundle.explorer_path),
        "local_id_path": str(local_path) if config.output.save_local_id else None,
        "group_id_path": str(group_path) if config.output.save_group_id else None,
        "merged_explorer_path": str(merged_path) if merged_path else None,
        "local_rows": len(local),
        "group_rows": len(group),
        "elapsed_seconds": elapsed,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.info("Finished intrinsic-dimension estimation in %.2f seconds.", elapsed)
    return manifest


def merge_id_into_explorer(
    explorer_path: str | Path,
    local_id: pd.DataFrame,
    *,
    feature_space: str,
    config: IDEstimationConfig,
) -> Path:
    """Merge selected local ID estimates into a new explorer table."""

    path = Path(explorer_path).resolve()
    explorer = read_parquet(path)
    if "row_id" not in explorer or "row_id" not in local_id:
        raise ValueError("Explorer and local ID tables require row_id for merging.")
    representative_k = min(config.local_id.k_values, key=lambda value: abs(value - 15))
    selected = [
        "row_id",
        "two_nn_lid_local",
        f"mle_lid_k{representative_k}",
        f"participation_ratio_k{representative_k}",
        f"pca_dim_95_k{representative_k}",
        f"knn_radius_k{representative_k}",
        f"knn_mean_distance_k{representative_k}",
    ]
    selected = [column for column in selected if column in local_id]
    merge_frame = local_id[selected].copy()
    merge_frame["id_feature_space"] = feature_space
    replace_columns = [column for column in merge_frame if column != "row_id"]
    explorer = explorer.drop(columns=replace_columns, errors="ignore")
    merged = explorer.merge(
        merge_frame,
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    output_path = (
        path
        if config.output.overwrite_explorer_data
        else path.with_name(config.output.merged_explorer_name)
    )
    write_parquet(merged, output_path)
    return output_path


def _compute_or_load_local(
    bundle: IDFeatureBundle,
    config: IDEstimationConfig,
    *,
    local_path: Path,
) -> pd.DataFrame:
    if not config.local_id.enabled:
        return pd.DataFrame({"row_id": bundle.metadata["row_id"]})
    if config.output.skip_existing and local_path.exists():
        frame = read_parquet(local_path)
        if _local_cache_matches(frame, bundle, config):
            LOGGER.info("Loaded cached local ID estimates: %s", local_path)
            return frame
        LOGGER.warning("Cached local ID output is incompatible; recomputing.")
    frame = compute_local_id(
        bundle.features,
        bundle.metadata,
        config,
        feature_space=bundle.feature_space,
    )
    frame["id_feature_fingerprint"] = _feature_fingerprint(bundle, config)
    if config.output.save_local_id:
        write_parquet(frame, local_path)
        if config.output.save_csv:
            frame.to_csv(local_path.with_suffix(".csv"), index=False)
        LOGGER.info("Saved local ID estimates: %s", local_path)
    return frame


def _compute_or_load_groups(
    bundle: IDFeatureBundle,
    local: pd.DataFrame,
    config: IDEstimationConfig,
    *,
    group_path: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], ScalingEstimate]]:
    if not config.global_id.enabled:
        return pd.DataFrame(), {}
    parquet_path = group_path.with_suffix(".parquet")
    if config.output.skip_existing and group_path.exists():
        cached = pd.read_csv(group_path)
        if _group_cache_matches(cached, bundle, config):
            LOGGER.info("Loaded cached group ID estimates: %s", group_path)
            return cached, {}
        LOGGER.warning("Cached group ID output is incompatible; recomputing.")

    rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str], ScalingEstimate] = {}
    groups: list[tuple[str, str, np.ndarray]] = [
        ("__all__", "__all__", np.arange(len(bundle.features), dtype=int))
    ]
    if config.groups.enabled:
        for column in config.groups.groupby_columns:
            if column not in bundle.metadata:
                LOGGER.warning("Skipping missing ID group column: %s", column)
                continue
            values = bundle.metadata[column].fillna("missing").astype(str)
            for value, indices in values.groupby(values).groups.items():
                groups.append(
                    (
                        column,
                        str(value),
                        np.asarray(list(indices), dtype=int),
                    )
                )

    for column, value, indices in groups:
        row: dict[str, Any] = {
            "groupby_column": column,
            "group_value": value,
            "n_samples": len(indices),
            "feature_space": bundle.feature_space,
            "id_feature_fingerprint": _feature_fingerprint(bundle, config),
        }
        if len(indices) < config.global_id.min_group_size:
            row["skipped_reason"] = (
                f"fewer than {config.global_id.min_group_size} samples"
            )
            rows.append(row)
            LOGGER.warning(
                "Skipping ID group %s=%s with %d samples.",
                column,
                value,
                len(indices),
            )
            continue
        estimates, scaling = compute_global_id(
            bundle.features[indices],
            config,
            feature_space=bundle.feature_space,
        )
        row.update(estimates)
        row.update(_local_group_summaries(local.iloc[indices], config))
        rows.append(row)
        if scaling is not None and not scaling.curve.empty:
            curves[(column, value)] = scaling

    frame = pd.DataFrame(rows)
    if config.output.save_group_id:
        group_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(group_path, index=False)
        write_parquet(frame, parquet_path)
        LOGGER.info("Saved group ID estimates: %s", group_path)
    return frame, curves


def _local_group_summaries(
    local: pd.DataFrame,
    config: IDEstimationConfig,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for k_value in config.local_id.k_values:
        for column, prefix in (
            (f"mle_lid_k{k_value}", "local_mle_lid"),
            (f"participation_ratio_k{k_value}", "participation_ratio"),
        ):
            if column not in local:
                continue
            values = pd.to_numeric(local[column], errors="coerce")
            result[f"mean_{prefix}_k{k_value}"] = float(values.mean())
            result[f"median_{prefix}_k{k_value}"] = float(values.median())
    column = "two_nn_lid_local"
    if column in local:
        values = pd.to_numeric(local[column], errors="coerce")
        result["mean_two_nn_lid_local"] = float(values.mean())
        result["median_two_nn_lid_local"] = float(values.median())
    return result


def _save_scaling_curves(
    curves: dict[tuple[str, str], ScalingEstimate],
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for (column, value), estimate in curves.items():
        frame = estimate.curve.copy()
        frame.insert(0, "group_value", value)
        frame.insert(0, "groupby_column", column)
        frame["dimension"] = estimate.dimension
        frame["r2"] = estimate.r2
        filename = f"{_safe_name(column)}_{_safe_name(value)}_ball_scaling.csv"
        frame.to_csv(directory / filename, index=False)


def _save_processed_features(
    bundle: IDFeatureBundle,
    config: IDEstimationConfig,
    output_dir: Path,
) -> None:
    pca = config.features.pca_preprocess
    if not pca.enabled or not pca.save_features:
        return
    directory = output_dir / "features"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{bundle.feature_space}_features.npy", bundle.features)
    write_parquet(
        bundle.metadata,
        directory / f"{bundle.feature_space}_metadata.parquet",
    )


def _enabled_local_estimators(config: IDEstimationConfig) -> list[str]:
    estimators = config.local_id.estimators
    return [
        name
        for name in (
            "covariance_spectrum",
            "participation_ratio",
            "pca_threshold",
            "mle_lid",
            "two_nn",
            "ball_scaling",
        )
        if getattr(estimators, name)
    ]


def _enabled_global_estimators(config: IDEstimationConfig) -> list[str]:
    estimators = config.global_id.estimators
    enabled = [
        name
        for name in (
            "two_nn",
            "mle_lid",
            "participation_ratio",
            "pca_threshold",
            "correlation_dimension",
            "ball_scaling",
        )
        if getattr(estimators, name)
    ]
    return [*enabled, *(f"skdim_{name}" for name in config.global_id.skdim_estimators)]


def _local_cache_matches(
    frame: pd.DataFrame,
    bundle: IDFeatureBundle,
    config: IDEstimationConfig,
) -> bool:
    if "row_id" not in frame or "feature_space" not in frame:
        return False
    if frame["row_id"].tolist() != bundle.metadata["row_id"].tolist():
        return False
    if not frame["feature_space"].astype(str).eq(bundle.feature_space).all():
        return False
    if "id_feature_fingerprint" not in frame:
        return False
    if not frame["id_feature_fingerprint"].astype(str).eq(
        _feature_fingerprint(bundle, config)
    ).all():
        return False
    expected = {"two_nn_lid_local"}
    for k_value in config.local_id.k_values:
        expected.update(
            {
                f"knn_radius_k{k_value}",
                f"knn_mean_distance_k{k_value}",
            }
        )
        estimators = config.local_id.estimators
        if estimators.participation_ratio:
            expected.add(f"participation_ratio_k{k_value}")
        if estimators.mle_lid:
            expected.add(f"mle_lid_k{k_value}")
        if estimators.pca_threshold:
            for threshold in config.pca_thresholds.explained_variance:
                expected.add(
                    f"pca_dim_{int(round(threshold * 100))}_k{k_value}"
                )
        if estimators.ball_scaling:
            expected.add(f"ball_scaling_dim_k{k_value}")
            expected.add(f"ball_scaling_r2_k{k_value}")
    return expected <= set(frame.columns)


def _group_cache_matches(
    frame: pd.DataFrame,
    bundle: IDFeatureBundle,
    config: IDEstimationConfig,
) -> bool:
    required = {
        "groupby_column",
        "group_value",
        "n_samples",
        "feature_space",
    }
    if not required <= set(frame.columns):
        return False
    if not frame["feature_space"].astype(str).eq(bundle.feature_space).all():
        return False
    if "id_feature_fingerprint" not in frame:
        return False
    if not frame["id_feature_fingerprint"].astype(str).eq(
        _feature_fingerprint(bundle, config)
    ).all():
        return False
    available_groups = {
        column
        for column in config.groups.groupby_columns
        if column in bundle.metadata
    }
    cached_groups = set(frame["groupby_column"].astype(str))
    if not {"__all__", *available_groups} <= cached_groups:
        return False
    estimators = config.global_id.estimators
    expected: set[str] = set()
    if estimators.participation_ratio:
        expected.add("global_participation_ratio")
    if estimators.pca_threshold:
        for threshold in config.pca_thresholds.explained_variance:
            expected.add(f"global_pca_dim_{int(round(threshold * 100))}")
    if estimators.two_nn:
        expected.add("global_two_nn_lid")
    if estimators.mle_lid:
        expected.update(
            f"global_mle_lid_k{k_value}"
            for k_value in config.global_id.mle_k_values
        )
    if estimators.correlation_dimension:
        expected.add("correlation_dimension")
    if estimators.ball_scaling:
        expected.update(
            {"ball_scaling_dim", "ball_scaling_r2", "ball_scaling_num_radii"}
        )
    return expected <= set(frame.columns)


def _resolve_output_dir(config: IDEstimationConfig, root: Path) -> Path:
    path = config.output_dir.expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _feature_fingerprint(
    bundle: IDFeatureBundle,
    config: IDEstimationConfig,
) -> str:
    source_stat = bundle.source_path.stat() if bundle.source_path else None
    explorer_stat = bundle.explorer_path.stat()
    payload = {
        "feature_space": bundle.feature_space,
        "shape": list(bundle.features.shape),
        "source_path": str(bundle.source_path) if bundle.source_path else "raw_pixels",
        "source_size": source_stat.st_size if source_stat else None,
        "source_mtime_ns": source_stat.st_mtime_ns if source_stat else None,
        "explorer_size": explorer_stat.st_size,
        "explorer_mtime_ns": explorer_stat.st_mtime_ns,
        "normalize": config.features.normalize,
        "pca": {
            "enabled": config.features.pca_preprocess.enabled,
            "n_components": config.features.pca_preprocess.n_components,
            "whiten": config.features.pca_preprocess.whiten,
            "random_state": config.features.pca_preprocess.random_state,
        },
        "metric": config.distance.metric,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned[:120] or "group"
