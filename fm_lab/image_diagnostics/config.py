"""Typed configuration for generic dataset exploration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fm_lab.utils.config import ConfigError, load_config


@dataclass(frozen=True)
class InputConfig:
    type: str = "mnist"
    dataset_root: str = "data/mnist"
    split: str = "test"
    order: str = "source"
    thumbnail_mode: str = "files"
    max_samples: int | None = None
    sample_seed: int = 42
    download: bool = False
    data_path: str = ""
    labels_path: str | None = None
    image_shape: tuple[int, int] | None = None
    value_range: tuple[float, float] | None = None
    experiment_dir: str = ""
    metadata_path: str = "metadata/per_image_metadata.jsonl"
    image_root: str = "images"
    include_status: list[str] = field(default_factory=lambda: ["success"])


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = "raw"
    name: str = "raw_pixels"
    cache_dir: str | None = None
    normalize: bool = False
    skip_existing: bool = True
    image_size: tuple[int, int] = (64, 64)
    repo_id: str = "facebook/dinov2-base"
    batch_size: int = 16
    device: str = "auto"
    dtype: str = "float16"


@dataclass(frozen=True)
class ProjectionVariantConfig:
    name: str
    key: str
    method: str = "umap"
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    perplexity: float = 30.0
    init: str = "random"
    learning_rate: float | str = "auto"
    source_path: str | None = None
    source_url: str | None = None
    download: bool = False


@dataclass(frozen=True)
class ProjectionConfig:
    method: str = "umap"
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    also_compute_pca: bool = True
    also_compute_tsne: bool = False
    skip_existing: bool = True
    variants: tuple[ProjectionVariantConfig, ...] = ()


@dataclass(frozen=True)
class LocalDiagnosticsConfig:
    enabled: bool = True
    k_neighbors: int = 15
    metric: str = "euclidean"
    covariance_eigenvalues: int = 5
    compute_knn_radius: bool = True
    compute_covariance_spectrum: bool = True
    compute_participation_ratio: bool = True
    compute_two_nn_lid: bool = True
    compute_centroid_distances: bool = True
    compute_outlier_score: bool = True
    skip_existing: bool = True


@dataclass(frozen=True)
class OutputConfig:
    root_dir: str = "outputs/dataset_explorer"
    save_features: bool = True
    save_projection: bool = True
    save_diagnostics: bool = True
    save_explorer_data: bool = True


@dataclass(frozen=True)
class ExplorerConfig:
    enabled: bool = True
    app_type: str = "streamlit"
    renderer: str = "canvas2d"
    thumbnail_size: int = 160
    default_color_by: str = "label"
    height: int = 760
    sidebar_width: int = 280
    point_size: int = 11
    hover_size: int = 44
    transition_ms: int = 650
    transition_easing: str = "ease"
    scale_point_size_with_zoom: bool = True
    atlas_tile_size: int = 28
    atlas_size: int = 2048
    selector_label: str = "Projection"
    preview_mode: str = "original"
    show_metrics: bool = True
    show_legend: bool = True
    show_view_controls: bool = True
    show_instructions: bool = True
    show_workspace: bool = True
    compute_projection_diagnostics: bool = False
    projection_diagnostics_k: int = 15


@dataclass(frozen=True)
class IDEstimationPostprocessConfig:
    enabled: bool = False
    config_path: str | None = None


@dataclass(frozen=True)
class DiagnosticsRunConfig:
    explorer_name: str
    input: InputConfig = field(default_factory=InputConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    diagnostics: LocalDiagnosticsConfig = field(default_factory=LocalDiagnosticsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    explorer: ExplorerConfig = field(default_factory=ExplorerConfig)
    id_estimation: IDEstimationPostprocessConfig = field(
        default_factory=IDEstimationPostprocessConfig
    )
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return Path(self.output.root_dir) / self.explorer_name


def load_diagnostics_config(path: str | Path) -> DiagnosticsRunConfig:
    """Load and validate a dataset explorer YAML config."""

    return diagnostics_config_from_dict(load_config(path))


def diagnostics_config_from_dict(raw: dict[str, Any]) -> DiagnosticsRunConfig:
    """Convert a raw config mapping into typed sections."""

    explorer_name = raw.get("explorer_name") or raw.get("diagnostics_name")
    if not explorer_name:
        raise ConfigError("Dataset explorer config must define explorer_name.")

    config = DiagnosticsRunConfig(
        explorer_name=str(explorer_name),
        input=_input_config(_section(raw, "input")),
        features=_feature_config(raw),
        projection=_projection_config(_section(raw, "projection")),
        diagnostics=LocalDiagnosticsConfig(**_section(raw, "diagnostics")),
        output=_output_config(_section(raw, "output")),
        explorer=ExplorerConfig(**_section(raw, "explorer")),
        id_estimation=IDEstimationPostprocessConfig(
            **_section(raw, "id_estimation")
        ),
        raw=_normalized_raw(raw, str(explorer_name)),
    )
    validate_diagnostics_config(config)
    return config


def apply_diagnostics_overrides(
    raw: dict[str, Any],
    *,
    input_path: str | None = None,
    experiment_dir: str | None = None,
    feature_mode: str | None = None,
    recompute_features: bool = False,
    recompute_embeddings: bool = False,
    recompute_projection: bool = False,
    recompute_diagnostics: bool = False,
    no_explorer: bool = False,
    no_id_estimation: bool = False,
) -> dict[str, Any]:
    """Apply CLI overrides without mutating the loaded YAML mapping."""

    updated = deepcopy(raw)
    input_values = updated.setdefault("input", {})
    input_type = str(input_values.get("type", "image_metadata"))
    selected_path = input_path or experiment_dir
    if selected_path is not None:
        if input_type in {"mnist", "fashion_mnist", "cifar10"}:
            input_values["dataset_root"] = selected_path
        elif input_type == "numpy":
            input_values["data_path"] = selected_path
        else:
            input_values["experiment_dir"] = selected_path
    if feature_mode is not None:
        updated.setdefault("features", {})["mode"] = feature_mode
    if recompute_features or recompute_embeddings:
        updated.setdefault("features", {})["skip_existing"] = False
    if recompute_projection:
        updated.setdefault("projection", {})["skip_existing"] = False
    if recompute_diagnostics:
        updated.setdefault("diagnostics", {})["skip_existing"] = False
    if no_explorer:
        updated.setdefault("explorer", {})["enabled"] = False
        updated.setdefault("id_estimation", {})["enabled"] = False
    if no_id_estimation:
        updated.setdefault("id_estimation", {})["enabled"] = False
    return updated


def validate_diagnostics_config(config: DiagnosticsRunConfig) -> None:
    """Validate dataset, feature, and numerical settings."""

    input_config = config.input
    if input_config.type not in {
        "mnist",
        "fashion_mnist",
        "cifar10",
        "numpy",
        "image_metadata",
    }:
        raise ConfigError(f"Unsupported input.type: {input_config.type}")
    if input_config.max_samples is not None and input_config.max_samples < 1:
        raise ConfigError("input.max_samples must be positive or null.")
    if input_config.type in {"mnist", "fashion_mnist", "cifar10"}:
        if input_config.split not in {"train", "test", "all"}:
            raise ConfigError(
                f"{input_config.type} input.split must be train, test, or all."
            )
    if input_config.type in {"mnist", "fashion_mnist"}:
        dataset_name = (
            "MNIST" if input_config.type == "mnist" else "Fashion-MNIST"
        )
        if input_config.order not in {"source", "mldata"}:
            raise ConfigError(
                f"{dataset_name} input.order must be source or mldata."
            )
        if input_config.order == "mldata" and input_config.split != "all":
            raise ConfigError(
                f"{dataset_name} input.order=mldata requires input.split=all."
            )
        if input_config.thumbnail_mode not in {"files", "atlas"}:
            raise ConfigError(
                f"{dataset_name} input.thumbnail_mode must be files or atlas."
            )
    elif input_config.type == "cifar10":
        if input_config.order != "source":
            raise ConfigError("CIFAR-10 input.order must be source.")
        if input_config.thumbnail_mode not in {"files", "atlas"}:
            raise ConfigError("CIFAR-10 input.thumbnail_mode must be files or atlas.")
    elif input_config.type == "numpy":
        if not input_config.data_path:
            raise ConfigError("NumPy input requires input.data_path.")
    elif not input_config.experiment_dir:
        raise ConfigError("Image metadata input requires input.experiment_dir.")
    if input_config.image_shape is not None:
        if len(input_config.image_shape) != 2 or min(input_config.image_shape) < 1:
            raise ConfigError("input.image_shape must contain two positive dimensions.")

    features = config.features
    if features.mode not in {"raw", "dinov2"}:
        raise ConfigError(f"Unsupported features.mode: {features.mode}")
    if (
        features.mode == "dinov2"
        and input_config.type == "numpy"
        and input_config.image_shape is None
    ):
        raise ConfigError(
            "DINOv2 features on NumPy input require input.image_shape."
        )
    if len(features.image_size) != 2 or min(features.image_size) < 1:
        raise ConfigError("features.image_size must contain two positive dimensions.")
    if features.batch_size < 1:
        raise ConfigError("features.batch_size must be positive.")
    if features.dtype not in {"float32", "float16", "bfloat16"}:
        raise ConfigError(f"Unsupported features.dtype: {features.dtype}")

    if config.projection.method not in {"umap", "pca", "tsne"}:
        raise ConfigError(f"Unsupported projection method: {config.projection.method}")
    if config.projection.n_neighbors < 2:
        raise ConfigError("projection.n_neighbors must be at least 2.")
    if config.projection.min_dist < 0.0:
        raise ConfigError("projection.min_dist must be non-negative.")
    projection_keys: set[str] = set()
    for variant in config.projection.variants:
        if variant.method not in {"umap", "pca", "tsne"}:
            raise ConfigError(
                f"Unsupported projection method for {variant.name!r}: {variant.method}"
            )
        if not variant.name.strip() or not variant.key.strip():
            raise ConfigError("Projection variants require non-empty name and key values.")
        if variant.n_components not in {2, 3}:
            raise ConfigError(
                f"Projection {variant.name!r} n_components must be 2 or 3."
            )
        if variant.key in projection_keys:
            raise ConfigError(f"Duplicate projection variant key: {variant.key}")
        projection_keys.add(variant.key)
        if variant.n_neighbors < 2:
            raise ConfigError(f"Projection {variant.name!r} n_neighbors must be at least 2.")
        if variant.min_dist < 0.0:
            raise ConfigError(f"Projection {variant.name!r} min_dist must be non-negative.")
        if variant.perplexity <= 0.0:
            raise ConfigError(f"Projection {variant.name!r} perplexity must be positive.")
        if variant.source_url and not variant.source_path:
            raise ConfigError(
                f"Projection {variant.name!r} source_url requires source_path."
            )
    if config.diagnostics.k_neighbors < 2:
        raise ConfigError("diagnostics.k_neighbors must be at least 2.")
    if config.diagnostics.covariance_eigenvalues < 1:
        raise ConfigError("diagnostics.covariance_eigenvalues must be positive.")
    if config.explorer.app_type != "streamlit":
        raise ConfigError("Only explorer.app_type=streamlit is currently supported.")
    if config.explorer.renderer not in {"canvas2d", "three3d"}:
        raise ConfigError("explorer.renderer must be canvas2d or three3d.")
    if config.explorer.renderer == "three3d":
        if not config.projection.variants:
            raise ConfigError("explorer.renderer=three3d requires projection.variants.")
    if config.explorer.transition_easing not in {"ease", "linear"}:
        raise ConfigError("explorer.transition_easing must be ease or linear.")
    if config.explorer.preview_mode not in {"original", "map"}:
        raise ConfigError("explorer.preview_mode must be original or map.")
    if config.explorer.projection_diagnostics_k < 1:
        raise ConfigError("explorer.projection_diagnostics_k must be positive.")
    if config.id_estimation.enabled and not config.id_estimation.config_path:
        raise ConfigError(
            "id_estimation.config_path is required when ID estimation is enabled."
        )
    if config.id_estimation.enabled and (
        not config.explorer.enabled or not config.output.save_explorer_data
    ):
        raise ConfigError(
            "ID estimation requires explorer.enabled and output.save_explorer_data."
        )
    for name in (
        "height",
        "sidebar_width",
        "point_size",
        "hover_size",
        "transition_ms",
        "atlas_tile_size",
        "atlas_size",
    ):
        if getattr(config.explorer, name) < 1:
            raise ConfigError(f"explorer.{name} must be positive.")


def _input_config(values: dict[str, Any]) -> InputConfig:
    if "image_shape" in values and values["image_shape"] is not None:
        values["image_shape"] = tuple(int(value) for value in values["image_shape"])
    if "value_range" in values and values["value_range"] is not None:
        values["value_range"] = tuple(float(value) for value in values["value_range"])
    return InputConfig(**values)


def _feature_config(raw: dict[str, Any]) -> FeatureConfig:
    values = _section(raw, "features")
    if not values and "embedding" in raw:
        embedding = _section(raw, "embedding")
        models = embedding.pop("models", [])
        enabled = [model for model in models if model.get("enabled", True)]
        if enabled:
            model = enabled[0]
            values = {
                "mode": "dinov2",
                "name": model.get("name", "dinov2"),
                "repo_id": model.get("repo_id", "facebook/dinov2-base"),
                "batch_size": model.get("batch_size", 16),
                "device": model.get("device", "auto"),
                "dtype": model.get("dtype", "float16"),
                "normalize": embedding.get("normalize_embeddings", True),
                "skip_existing": embedding.get("skip_existing", True),
            }
    if "image_size" in values:
        values["image_size"] = tuple(int(value) for value in values["image_size"])
    return FeatureConfig(**values)


def _projection_config(values: dict[str, Any]) -> ProjectionConfig:
    raw_variants = values.pop("variants", [])
    if not isinstance(raw_variants, list):
        raise ConfigError("projection.variants must be a list.")
    variants = tuple(ProjectionVariantConfig(**variant) for variant in raw_variants)
    return ProjectionConfig(variants=variants, **values)


def _output_config(values: dict[str, Any]) -> OutputConfig:
    if "save_embeddings" in values and "save_features" not in values:
        values["save_features"] = values.pop("save_embeddings")
    return OutputConfig(**values)


def _normalized_raw(raw: dict[str, Any], explorer_name: str) -> dict[str, Any]:
    normalized = deepcopy(raw)
    normalized["explorer_name"] = explorer_name
    normalized.pop("diagnostics_name", None)
    return normalized


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Config section {name!r} must be a mapping.")
    return deepcopy(value)
