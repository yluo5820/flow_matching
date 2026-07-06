"""Build and register projection views for geometry explorer variants."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from fm_lab.geometry_explorer.bundles import build_and_register_projection_payload_index
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.variants import load_variant_bundle
from fm_lab.image_diagnostics.config import (
    DiagnosticsRunConfig,
    diagnostics_config_from_dict,
)
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle
from fm_lab.image_diagnostics.explorer_data import build_explorer_data
from fm_lab.image_diagnostics.feature_runner import compute_or_load_features
from fm_lab.image_diagnostics.id_config import (
    IDEstimationConfig,
    id_config_from_dict,
    load_id_config,
)
from fm_lab.image_diagnostics.id_runner import run_id_estimation
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
        },
    )
    if raw.get("id_estimation", {}).get("enabled") and not raw["id_estimation"].get(
        "config_path"
    ):
        raw = deep_update(raw, {"id_estimation": {"config_path": "__auto__"}})
    config = diagnostics_config_from_dict(raw)
    output_dir = prepare_output_dir(config)
    logger = configure_logging(output_dir)
    dataset = load_variant_bundle(variant_id, workspace=workspace)
    if dataset.vectors is None:
        raise RuntimeError(f"Dataset variant {variant_id} has no feature vectors.")
    dataset = _sample_view_dataset(
        dataset,
        max_samples=config.input.max_samples,
        seed=config.input.sample_seed,
        strategy=config.input.sample_strategy,
    )
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

    id_estimation = None
    registered_explorer_path = explorer_path
    if config.id_estimation.enabled:
        if not config.output.save_features:
            raise RuntimeError("Geometry ID estimation requires output.save_features: true.")
        id_config = _id_config_for_view(
            config,
            output_dir=output_dir,
            project_root=project_root,
        )
        logger.info("Running intrinsic-dimension estimation for geometry view.")
        id_estimation = run_id_estimation(id_config, project_root=project_root)
        logger = configure_logging(output_dir)
        merged_path = id_estimation.get("merged_explorer_path")
        if merged_path:
            registered_explorer_path = Path(str(merged_path))

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
        explorer_data_path=registered_explorer_path,
        output_dir=output_dir,
        projection_names=projection_names,
        renderer=config.explorer.renderer,
        row_count=len(explorer_data),
    )
    build_and_register_projection_payload_index(view_id, workspace=workspace)
    elapsed = time.perf_counter() - started
    logger.info("Registered geometry projection view %s in %.2f seconds.", view_id, elapsed)
    return {
        "view_id": view_id,
        "variant_id": variant_id,
        "output_dir": output_dir,
        "explorer_data": registered_explorer_path,
        "rows": len(explorer_data),
        "projection_names": projection_names,
        "id_estimation": id_estimation,
        "runtime_seconds": elapsed,
    }


def _sample_view_dataset(
    dataset: DatasetBundle,
    *,
    max_samples: int | None,
    seed: int,
    strategy: str,
) -> DatasetBundle:
    if max_samples is None or max_samples >= len(dataset.metadata):
        return dataset
    metadata = dataset.metadata.reset_index(drop=True)
    if strategy == "stratified" and "label" in metadata:
        positions = _stratified_sample_positions(
            metadata["label"].astype(str).to_numpy(),
            max_samples=max_samples,
            seed=seed,
        )
    else:
        rng = np.random.default_rng(seed)
        positions = np.sort(
            rng.choice(len(metadata), size=max_samples, replace=False)
        )
    sampled_metadata = metadata.iloc[positions].reset_index(drop=True).copy()
    sampled_vectors = (
        np.asarray(dataset.vectors[positions], dtype=np.float32)
        if dataset.vectors is not None
        else None
    )
    return DatasetBundle(
        metadata=sampled_metadata,
        vectors=sampled_vectors,
        source_id=f"{dataset.source_id}:view-sample:{strategy}:{max_samples}:{seed}",
        source_description=(
            f"{dataset.source_description} "
            f"(view sample: {len(sampled_metadata):,}/{len(metadata):,}, {strategy})"
        ),
        total_rows=len(sampled_metadata),
        skipped_rows=dataset.skipped_rows,
        image_shape=dataset.image_shape,
        value_range=dataset.value_range,
    )


def _stratified_sample_positions(
    labels: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = {
        label: np.flatnonzero(labels == label)
        for label in sorted(set(labels), key=_natural_sort_key)
    }
    quotas = {label: 0 for label in groups}
    base = max_samples // max(1, len(groups))
    remainder = max_samples % max(1, len(groups))
    for offset, label in enumerate(groups):
        quotas[label] = min(len(groups[label]), base + int(offset < remainder))

    shortfall = max_samples - sum(quotas.values())
    while shortfall > 0:
        progressed = False
        for label, positions in groups.items():
            if shortfall <= 0:
                break
            available = len(positions) - quotas[label]
            if available <= 0:
                continue
            quotas[label] += 1
            shortfall -= 1
            progressed = True
        if not progressed:
            break

    selected: list[int] = []
    for label, positions in groups.items():
        quota = quotas[label]
        if quota <= 0:
            continue
        shuffled = np.array(positions, copy=True)
        rng.shuffle(shuffled)
        selected.extend(int(value) for value in shuffled[:quota])
    return np.asarray(sorted(selected), dtype=int)


def _natural_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


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


def _id_config_for_view(
    config: DiagnosticsRunConfig,
    *,
    output_dir: Path,
    project_root: str | Path | None,
) -> IDEstimationConfig:
    if config.id_estimation.config_path and config.id_estimation.config_path != "__auto__":
        path = Path(config.id_estimation.config_path).expanduser()
        if not path.is_absolute():
            path = Path(project_root or Path.cwd()) / path
        loaded = load_id_config(path)
        return replace(
            loaded,
            input=replace(
                loaded.input,
                diagnostics_dir=str(output_dir.resolve()),
            ),
        )
    name = f"{config.explorer_name}_{config.features.name}_id"
    raw = {
        "id_estimation_name": name,
        "input": {
            "diagnostics_dir": str(output_dir.resolve()),
            "explorer_data_path": "explorer/explorer_data.parquet",
            "embedding_source": f"features/{config.features.name}_features.npy",
            "embedding_metadata": f"features/{config.features.name}_metadata.parquet",
            "feature_space_name": config.features.name,
            "source_type": "npy",
        },
        "features": {
            "normalize": config.features.normalize,
            "pca_preprocess": {
                "enabled": True,
                "n_components": 50,
                "whiten": False,
                "random_state": 42,
                "save_features": True,
            },
        },
        "groups": {
            "enabled": True,
            "groupby_columns": ["label", "manual_label"],
        },
        "local_id": {
            "enabled": True,
            "k_values": [5, 10, 15, 30, 50],
            "covariance_eigenvalues": 10,
        },
        "global_id": {
            "enabled": True,
            "min_group_size": 20,
            "scaling_max_points": 2000,
        },
        "distance": {"metric": config.diagnostics.metric},
        "output": {
            "root_dir": str(output_dir / "id_estimation"),
            "merge_into_explorer_data": True,
            "merged_explorer_name": f"explorer_data_with_{config.features.name}_id.parquet",
            "overwrite_explorer_data": False,
            "skip_existing": True,
        },
    }
    return id_config_from_dict(raw)
