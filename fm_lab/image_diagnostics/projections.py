"""Low-dimensional projections for image embeddings."""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from fm_lab.image_diagnostics.config import ProjectionConfig, ProjectionVariantConfig
from fm_lab.image_diagnostics.save_utils import OptionalDependencyError

LOGGER = logging.getLogger("fm_lab.image_diagnostics")
AXIS_NAMES = ("x", "y", "z")


def compute_or_load_projections(
    embeddings: np.ndarray,
    row_ids: pd.Series,
    config: ProjectionConfig,
    output_dir: Path,
    *,
    feature_name: str,
    save: bool = True,
    project_root: str | Path | None = None,
) -> pd.DataFrame:
    """Compute configured projections and return one merged row per image."""

    result = pd.DataFrame({"row_id": row_ids.to_numpy()})
    for variant in projection_variants(config):
        path = output_dir / "projections" / f"{feature_name}_{variant.key}.csv"
        if save and config.skip_existing and path.exists():
            projected = pd.read_csv(path)
            _validate_projection(
                projected,
                variant.key,
                row_ids,
                n_components=variant.n_components,
            )
            LOGGER.info("Loaded cached %s projection: %s", variant.name, path)
        else:
            coordinates = _compute_projection_variant(
                embeddings,
                variant,
                project_root=project_root,
            )
            projected = pd.DataFrame(
                {"row_id": row_ids.to_numpy()}
                | {
                    f"{variant.key}_{axis}": coordinates[:, index]
                    for index, axis in enumerate(AXIS_NAMES[: variant.n_components])
                }
            )
            if save:
                path.parent.mkdir(parents=True, exist_ok=True)
                projected.to_csv(path, index=False)
                LOGGER.info("Saved %s projection: %s", variant.name, path)
        result = result.merge(projected, on="row_id", how="left", validate="one_to_one")
    return result


def compute_projection(
    embeddings: np.ndarray,
    config: ProjectionConfig,
    *,
    method: str | None = None,
) -> np.ndarray:
    """Compute a low-dimensional UMAP, PCA, or t-SNE representation."""

    selected = method or config.method
    variant = ProjectionVariantConfig(
        name=selected.upper(),
        key=selected,
        method=selected,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        metric=config.metric,
        random_state=config.random_state,
    )
    return _compute_projection_variant(embeddings, variant)


def projection_variants(config: ProjectionConfig) -> tuple[ProjectionVariantConfig, ...]:
    """Return explicit variants or derive the legacy method/PCA/t-SNE set."""

    if config.variants:
        return config.variants
    methods = [config.method]
    if config.also_compute_pca and "pca" not in methods:
        methods.append("pca")
    if config.also_compute_tsne and "tsne" not in methods:
        methods.append("tsne")
    return tuple(
        ProjectionVariantConfig(
            name=_default_display_name(method),
            key=method,
            method=method,
            n_neighbors=config.n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=config.random_state,
        )
        for method in methods
    )


def _compute_projection_variant(
    embeddings: np.ndarray,
    variant: ProjectionVariantConfig,
    *,
    project_root: str | Path | None = None,
) -> np.ndarray:
    if variant.source_path:
        return _load_precomputed_projection(
            variant,
            expected_rows=len(embeddings),
            project_root=project_root,
        )
    selected = variant.method
    if len(embeddings) == 0:
        return np.empty((0, variant.n_components), dtype=np.float32)
    if selected == "pca":
        return _pca(embeddings, variant.random_state, variant.n_components)
    if selected == "umap":
        if len(embeddings) < 3:
            LOGGER.warning("Fewer than three samples; using PCA coordinates for UMAP output.")
            return _pca(embeddings, variant.random_state, variant.n_components)
        try:
            import umap
        except ImportError as exc:
            raise OptionalDependencyError(
                'UMAP requires umap-learn. Install ".[image-diagnostics]".'
            ) from exc
        reducer = umap.UMAP(
            n_components=variant.n_components,
            n_neighbors=min(variant.n_neighbors, len(embeddings) - 1),
            min_dist=variant.min_dist,
            metric=variant.metric,
            random_state=variant.random_state,
        )
        return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
    if selected == "tsne":
        if len(embeddings) < 3:
            LOGGER.warning("Fewer than three samples; using PCA coordinates for t-SNE output.")
            return _pca(embeddings, variant.random_state, variant.n_components)
        from sklearn.manifold import TSNE

        perplexity = min(
            variant.perplexity,
            max(1.0, (len(embeddings) - 1) / 3),
        )
        return TSNE(
            n_components=variant.n_components,
            metric=variant.metric,
            perplexity=perplexity,
            init=variant.init,
            learning_rate=variant.learning_rate,
            random_state=variant.random_state,
        ).fit_transform(embeddings)
    raise ValueError(f"Unsupported projection method: {selected}")


def _load_precomputed_projection(
    variant: ProjectionVariantConfig,
    *,
    expected_rows: int,
    project_root: str | Path | None,
) -> np.ndarray:
    path = Path(variant.source_path or "").expanduser()
    if not path.is_absolute():
        path = Path(project_root or Path.cwd()) / path
    path = path.resolve()
    if not path.exists():
        if not variant.download or not variant.source_url:
            raise FileNotFoundError(
                f"Precomputed projection for {variant.name!r} does not exist: {path}"
            )
        _download_projection(variant.source_url, path)
    if path.suffix.lower() == ".npy":
        coordinates = np.load(path)
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            coordinates = np.asarray(json.load(handle), dtype=np.float32)
    else:
        coordinates = (
            pd.read_csv(path)
            .iloc[:, -variant.n_components :]
            .to_numpy(dtype=np.float32)
        )
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.shape != (expected_rows, variant.n_components):
        raise ValueError(
            f"Projection {variant.name!r} has shape {coordinates.shape}; "
            f"expected ({expected_rows}, {variant.n_components})."
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Projection {variant.name!r} contains non-finite coordinates.")
    LOGGER.info("Loaded precomputed %s projection: %s", variant.name, path)
    return coordinates


def _download_projection(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading reference projection: %s", url)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(url, temporary_path)  # noqa: S310
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _pca(
    embeddings: np.ndarray,
    random_state: int,
    n_components: int = 2,
) -> np.ndarray:
    if len(embeddings) == 1:
        return np.zeros((1, n_components), dtype=np.float32)
    components = min(n_components, len(embeddings), embeddings.shape[1])
    coordinates = PCA(n_components=components, random_state=random_state).fit_transform(embeddings)
    if components < n_components:
        coordinates = np.column_stack(
            [coordinates, np.zeros((len(coordinates), n_components - components))]
        )
    return np.asarray(coordinates, dtype=np.float32)


def _validate_projection(
    frame: pd.DataFrame,
    method: str,
    expected_row_ids: pd.Series,
    n_components: int = 2,
) -> None:
    required = {
        "row_id",
        *(f"{method}_{axis}" for axis in AXIS_NAMES[:n_components]),
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Cached {method} projection is missing columns: {sorted(missing)}"
        )
    if frame["row_id"].tolist() != expected_row_ids.tolist():
        raise RuntimeError(
            f"Cached {method} projection row IDs do not match the embedding cache. "
            "Recompute the projection."
        )


def _default_display_name(method: str) -> str:
    if method == "tsne":
        return "T-SNE"
    return method.upper()
