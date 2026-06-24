"""Feature extraction and cache management for generic datasets."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fm_lab.image_diagnostics.config import FeatureConfig
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle
from fm_lab.image_diagnostics.feature_models import (
    ImageFeatureExtractor,
    load_feature_model,
)
from fm_lab.image_diagnostics.image_dataset import load_rgb_image
from fm_lab.image_diagnostics.save_utils import read_parquet, write_parquet

LOGGER = logging.getLogger("fm_lab.image_diagnostics")


@dataclass(frozen=True)
class FeatureResult:
    feature_name: str
    features: np.ndarray
    metadata: pd.DataFrame
    loaded_from_cache: bool
    skipped_images: int = 0


def compute_or_load_features(
    *,
    config: FeatureConfig,
    dataset: DatasetBundle,
    output_dir: Path,
    save: bool = True,
    model_loader: Callable[[FeatureConfig], ImageFeatureExtractor] = load_feature_model,
) -> FeatureResult:
    """Load a compatible feature cache or compute features for the dataset."""

    feature_path, metadata_path = feature_cache_paths(output_dir, config.name)
    if save and config.skip_existing and feature_path.exists() and metadata_path.exists():
        features = np.load(feature_path)
        metadata = read_parquet(metadata_path)
        if _cache_matches(features, metadata, dataset, config):
            LOGGER.info("Loaded cached %s features: %s", config.name, feature_path)
            return FeatureResult(config.name, features, metadata, True)
        LOGGER.warning("Feature cache does not match the current dataset; recomputing.")

    if dataset.vectors is not None:
        if config.mode != "raw":
            raise ValueError("Vector datasets currently support features.mode=raw.")
        features = np.asarray(dataset.vectors, dtype=np.float32)
        metadata = dataset.metadata.reset_index(drop=True).copy()
        skipped = 0
    else:
        features, metadata, skipped = _extract_image_features(
            config,
            dataset.metadata,
            model_loader=model_loader,
        )

    if config.normalize:
        features = l2_normalize(features)
    metadata["feature_name"] = config.name
    metadata["feature_mode"] = config.mode
    metadata["feature_source_id"] = dataset.source_id
    metadata["features_normalized"] = config.normalize

    if save:
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, features)
        write_parquet(metadata, metadata_path)
        LOGGER.info(
            "Saved %d %s feature rows with dimension %d",
            len(features),
            config.name,
            features.shape[1],
        )
    return FeatureResult(config.name, features, metadata, False, skipped)


def feature_cache_paths(output_dir: Path, feature_name: str) -> tuple[Path, Path]:
    base = output_dir / "features"
    return (
        base / f"{feature_name}_features.npy",
        base / f"{feature_name}_metadata.parquet",
    )


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.finfo(np.float32).eps)


def _extract_image_features(
    config: FeatureConfig,
    metadata: pd.DataFrame,
    *,
    model_loader: Callable[[FeatureConfig], ImageFeatureExtractor],
) -> tuple[np.ndarray, pd.DataFrame, int]:
    extractor = model_loader(config) if config.mode == "dinov2" else None
    vectors: list[np.ndarray] = []
    positions: list[int] = []
    skipped = 0
    for start in range(0, len(metadata), config.batch_size):
        batch = metadata.iloc[start : start + config.batch_size]
        images = []
        batch_positions = []
        for position, row in zip(batch.index, batch.to_dict(orient="records"), strict=False):
            try:
                images.append(load_rgb_image(row["image_path"]))
                batch_positions.append(position)
            except Exception as exc:
                skipped += 1
                LOGGER.warning("Skipping unreadable image %s: %s", row["image_path"], exc)
        if not images:
            continue
        if extractor is None:
            batch_vectors = np.asarray(
                [
                    np.asarray(
                        image.resize(config.image_size),
                        dtype=np.float32,
                    ).reshape(-1)
                    / 255.0
                    for image in images
                ],
                dtype=np.float32,
            )
        else:
            batch_vectors = np.asarray(extractor.extract(images), dtype=np.float32)
        vectors.append(batch_vectors)
        positions.extend(batch_positions)
        for image in images:
            image.close()
    if not vectors:
        raise RuntimeError("No readable images were available for feature extraction.")
    return (
        np.concatenate(vectors, axis=0),
        metadata.loc[positions].reset_index(drop=True),
        skipped,
    )


def _cache_matches(
    features: np.ndarray,
    metadata: pd.DataFrame,
    dataset: DatasetBundle,
    config: FeatureConfig,
) -> bool:
    required = {
        "row_id",
        "feature_name",
        "feature_mode",
        "feature_source_id",
        "features_normalized",
    }
    if features.ndim != 2 or len(features) != len(metadata):
        return False
    if not required <= set(metadata.columns) or len(metadata) != len(dataset.metadata):
        return False
    if metadata["row_id"].tolist() != dataset.metadata["row_id"].tolist():
        return False
    if not (metadata["feature_name"].astype(str) == config.name).all():
        return False
    if not (metadata["feature_mode"].astype(str) == config.mode).all():
        return False
    if not (metadata["feature_source_id"].astype(str) == dataset.source_id).all():
        return False
    normalized = metadata["features_normalized"].map(_as_bool)
    return bool((normalized == config.normalize).all())


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
