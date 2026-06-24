"""Feature and metadata loading for intrinsic-dimension estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from fm_lab.image_diagnostics.feature_runner import l2_normalize
from fm_lab.image_diagnostics.id_config import IDEstimationConfig
from fm_lab.image_diagnostics.save_utils import read_parquet

LOGGER = logging.getLogger("fm_lab.image_diagnostics")


@dataclass(frozen=True)
class IDFeatureBundle:
    features: np.ndarray
    metadata: pd.DataFrame
    feature_space: str
    source_path: Path | None
    explorer_path: Path
    pca_model: PCA | None = None


@dataclass(frozen=True)
class IDInputSummary:
    feature_shape: tuple[int, int] | None
    metadata_rows: int
    explorer_path: Path
    feature_path: Path | None
    group_sizes: dict[str, dict[str, int]]


def load_id_features(
    config: IDEstimationConfig,
    *,
    project_root: str | Path | None = None,
) -> IDFeatureBundle:
    """Load aligned features and metadata, then apply configured preprocessing."""

    root = Path(project_root or Path.cwd()).resolve()
    diagnostics_dir = _resolve_required(config.input.diagnostics_dir, root)
    explorer_path = _resolve_input_path(
        config.input.explorer_data_path,
        diagnostics_dir=diagnostics_dir,
        project_root=root,
    )
    explorer = read_parquet(explorer_path).reset_index(drop=True)

    if config.input.source_type == "npy":
        assert config.input.embedding_source is not None
        feature_path = _resolve_input_path(
            config.input.embedding_source,
            diagnostics_dir=diagnostics_dir,
            project_root=root,
        )
        features = np.asarray(np.load(feature_path), dtype=np.float32)
        feature_metadata = _load_feature_metadata(
            config,
            feature_path=feature_path,
            diagnostics_dir=diagnostics_dir,
            project_root=root,
        )
        metadata = _align_metadata(explorer, feature_metadata, len(features))
    else:
        feature_path = None
        metadata = explorer.copy()
        features, metadata = _load_raw_pixels(config, metadata)

    if features.ndim != 2:
        raise ValueError(f"ID features must be a matrix; received shape {features.shape}.")
    if len(features) != len(metadata):
        raise ValueError(
            f"ID feature/metadata row mismatch: {len(features)} != {len(metadata)}."
        )
    if not np.isfinite(features).all():
        raise ValueError("ID features contain NaN or infinite values.")
    if "row_id" not in metadata:
        metadata["row_id"] = np.arange(len(metadata), dtype=int)

    if config.features.normalize:
        features = np.asarray(l2_normalize(features), dtype=np.float32)

    pca_model = None
    feature_space = config.input.feature_space_name
    pca_config = config.features.pca_preprocess
    if pca_config.enabled:
        components = min(pca_config.n_components, len(features), features.shape[1])
        if components < 1:
            raise ValueError("PCA preprocessing has no available components.")
        pca_model = PCA(
            n_components=components,
            whiten=pca_config.whiten,
            random_state=pca_config.random_state,
        )
        features = np.asarray(pca_model.fit_transform(features), dtype=np.float32)
        feature_space = f"{feature_space}_pca{components}"
        LOGGER.info("PCA-preprocessed ID features to %d dimensions.", components)

    return IDFeatureBundle(
        features=features,
        metadata=metadata.reset_index(drop=True),
        feature_space=feature_space,
        source_path=feature_path,
        explorer_path=explorer_path,
        pca_model=pca_model,
    )


def inspect_id_input(
    config: IDEstimationConfig,
    *,
    project_root: str | Path | None = None,
) -> IDInputSummary:
    """Inspect paths, feature shape, and planned group sizes without estimation."""

    root = Path(project_root or Path.cwd()).resolve()
    diagnostics_dir = _resolve_required(config.input.diagnostics_dir, root)
    explorer_path = _resolve_input_path(
        config.input.explorer_data_path,
        diagnostics_dir=diagnostics_dir,
        project_root=root,
    )
    metadata = read_parquet(explorer_path)
    feature_path = None
    feature_shape = None
    if config.input.source_type == "npy":
        assert config.input.embedding_source is not None
        feature_path = _resolve_input_path(
            config.input.embedding_source,
            diagnostics_dir=diagnostics_dir,
            project_root=root,
        )
        array = np.load(feature_path, mmap_mode="r")
        feature_shape = tuple(int(value) for value in array.shape)
    group_sizes = {
        column: {
            str(key): int(value)
            for key, value in metadata[column].fillna("missing").value_counts().items()
        }
        for column in config.groups.groupby_columns
        if column in metadata
    }
    return IDInputSummary(
        feature_shape=feature_shape,
        metadata_rows=len(metadata),
        explorer_path=explorer_path,
        feature_path=feature_path,
        group_sizes=group_sizes,
    )


def _load_feature_metadata(
    config: IDEstimationConfig,
    *,
    feature_path: Path,
    diagnostics_dir: Path,
    project_root: Path,
) -> pd.DataFrame | None:
    value = config.input.embedding_metadata
    if value:
        path = _resolve_input_path(
            value,
            diagnostics_dir=diagnostics_dir,
            project_root=project_root,
        )
        return read_parquet(path)
    inferred = feature_path.with_name(
        feature_path.name.replace("_features.npy", "_metadata.parquet")
    )
    return read_parquet(inferred) if inferred.exists() else None


def _align_metadata(
    explorer: pd.DataFrame,
    feature_metadata: pd.DataFrame | None,
    n_features: int,
) -> pd.DataFrame:
    if feature_metadata is None:
        if len(explorer) != n_features:
            raise ValueError(
                "Explorer rows do not match features and no embedding metadata was provided."
            )
        return explorer.copy()
    feature_metadata = feature_metadata.reset_index(drop=True)
    if len(feature_metadata) != n_features:
        raise ValueError("Embedding metadata row count does not match features.")
    if "row_id" in explorer and "row_id" in feature_metadata:
        if feature_metadata["row_id"].duplicated().any():
            raise ValueError("Embedding metadata contains duplicate row_id values.")
        if explorer["row_id"].duplicated().any():
            raise ValueError("Explorer data contains duplicate row_id values.")
        feature_ids = feature_metadata["row_id"].tolist()
        if set(feature_ids) != set(explorer["row_id"].tolist()):
            raise ValueError("Embedding metadata row IDs do not match explorer data.")
        aligned = explorer.set_index("row_id").loc[feature_ids].reset_index()
        for column in feature_metadata.columns:
            if column not in aligned:
                aligned[column] = feature_metadata[column].to_numpy()
        return aligned
    if len(explorer) != n_features:
        raise ValueError("Explorer rows do not match embedding metadata.")
    return explorer.copy()


def _load_raw_pixels(
    config: IDEstimationConfig,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    column = config.input.image_path_column
    if column not in metadata:
        raise ValueError(f"Raw pixel input requires metadata column {column!r}.")
    vectors: list[np.ndarray] = []
    positions: list[int] = []
    for position, value in enumerate(metadata[column]):
        try:
            with Image.open(Path(str(value)).expanduser()) as image:
                image = image.convert("L" if config.input.raw_grayscale else "RGB")
                if config.input.raw_image_size is not None:
                    image = image.resize(config.input.raw_image_size)
                vector = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
            vectors.append(vector)
            positions.append(position)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable raw image %s: %s", value, exc)
    if not vectors:
        raise RuntimeError("No readable images were available for raw-pixel ID features.")
    dimensions = {len(value) for value in vectors}
    if len(dimensions) != 1:
        raise ValueError(
            "Raw images have different sizes; configure input.raw_image_size."
        )
    return (
        np.asarray(vectors, dtype=np.float32),
        metadata.iloc[positions].reset_index(drop=True),
    )


def _resolve_required(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"ID diagnostics directory does not exist: {path}")
    return path


def _resolve_input_path(
    value: str,
    *,
    diagnostics_dir: Path,
    project_root: Path,
) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
    else:
        diagnostics_candidate = (diagnostics_dir / path).resolve()
        project_candidate = (project_root / path).resolve()
        resolved = (
            diagnostics_candidate
            if diagnostics_candidate.exists()
            else project_candidate
        )
    if not resolved.exists():
        raise FileNotFoundError(f"ID input path does not exist: {resolved}")
    return resolved
