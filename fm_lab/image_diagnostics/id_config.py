"""Configuration for representation-space intrinsic dimension estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fm_lab.utils.config import ConfigError, load_config


@dataclass(frozen=True)
class IDInputConfig:
    diagnostics_dir: str
    explorer_data_path: str = "explorer/explorer_data.parquet"
    embedding_source: str | None = None
    embedding_metadata: str | None = None
    feature_space_name: str = "features"
    source_type: str = "npy"
    image_path_column: str = "image_path"
    raw_image_size: tuple[int, int] | None = None
    raw_grayscale: bool = False


@dataclass(frozen=True)
class PCAProcessConfig:
    enabled: bool = False
    n_components: int = 50
    whiten: bool = False
    random_state: int = 42
    save_features: bool = True


@dataclass(frozen=True)
class IDFeatureConfig:
    normalize: bool = True
    pca_preprocess: PCAProcessConfig = field(default_factory=PCAProcessConfig)


@dataclass(frozen=True)
class IDGroupConfig:
    enabled: bool = True
    groupby_columns: tuple[str, ...] = ("family", "prompt_id", "manual_label")


@dataclass(frozen=True)
class LocalEstimatorConfig:
    covariance_spectrum: bool = True
    participation_ratio: bool = True
    pca_threshold: bool = True
    mle_lid: bool = True
    two_nn: bool = True
    ball_scaling: bool = True


@dataclass(frozen=True)
class LocalIDConfig:
    enabled: bool = True
    k_values: tuple[int, ...] = (5, 10, 15, 30, 50)
    covariance_eigenvalues: int = 10
    estimators: LocalEstimatorConfig = field(default_factory=LocalEstimatorConfig)


@dataclass(frozen=True)
class GlobalEstimatorConfig:
    two_nn: bool = True
    mle_lid: bool = True
    participation_ratio: bool = True
    pca_threshold: bool = True
    correlation_dimension: bool = True
    ball_scaling: bool = True


@dataclass(frozen=True)
class GlobalIDConfig:
    enabled: bool = True
    estimators: GlobalEstimatorConfig = field(default_factory=GlobalEstimatorConfig)
    min_group_size: int = 20
    mle_k_values: tuple[int, ...] = (10, 20)
    scaling_quantiles: tuple[float, ...] = (0.05, 0.1, 0.2, 0.35, 0.5)
    scaling_max_points: int = 2000
    skdim_estimators: tuple[str, ...] = ()


@dataclass(frozen=True)
class PCAThresholdConfig:
    explained_variance: tuple[float, ...] = (0.8, 0.9, 0.95, 0.99)


@dataclass(frozen=True)
class IDDistanceConfig:
    metric: str = "cosine"


@dataclass(frozen=True)
class IDOutputConfig:
    root_dir: str = "outputs/image_diagnostics"
    save_local_id: bool = True
    save_group_id: bool = True
    save_csv: bool = True
    merge_into_explorer_data: bool = True
    merged_explorer_name: str = "explorer_data_with_id.parquet"
    overwrite_explorer_data: bool = False
    skip_existing: bool = True


@dataclass(frozen=True)
class IDEstimationConfig:
    id_estimation_name: str
    input: IDInputConfig
    features: IDFeatureConfig = field(default_factory=IDFeatureConfig)
    groups: IDGroupConfig = field(default_factory=IDGroupConfig)
    local_id: LocalIDConfig = field(default_factory=LocalIDConfig)
    global_id: GlobalIDConfig = field(default_factory=GlobalIDConfig)
    pca_thresholds: PCAThresholdConfig = field(default_factory=PCAThresholdConfig)
    distance: IDDistanceConfig = field(default_factory=IDDistanceConfig)
    output: IDOutputConfig = field(default_factory=IDOutputConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return Path(self.output.root_dir) / self.id_estimation_name


def load_id_config(path: str | Path) -> IDEstimationConfig:
    """Load and validate an intrinsic-dimension YAML config."""

    return id_config_from_dict(load_config(path))


def id_config_from_dict(raw: dict[str, Any]) -> IDEstimationConfig:
    """Convert an ID config mapping to typed sections."""

    name = raw.get("id_estimation_name")
    if not name:
        raise ConfigError("ID estimation config requires id_estimation_name.")
    input_values = _section(raw, "input")
    if "raw_image_size" in input_values and input_values["raw_image_size"] is not None:
        input_values["raw_image_size"] = tuple(
            int(value) for value in input_values["raw_image_size"]
        )
    feature_values = _section(raw, "features")
    pca_values = feature_values.pop("pca_preprocess", {})
    group_values = _section(raw, "groups")
    if "groupby_columns" in group_values:
        group_values["groupby_columns"] = tuple(
            str(value) for value in group_values["groupby_columns"]
        )
    local_values = _section(raw, "local_id")
    local_estimator_values = local_values.pop("estimators", {})
    if "local_pca_threshold" in local_estimator_values:
        local_estimator_values["pca_threshold"] = local_estimator_values.pop(
            "local_pca_threshold"
        )
    if "k_values" in local_values:
        local_values["k_values"] = tuple(int(value) for value in local_values["k_values"])
    global_values = _section(raw, "global_id")
    global_estimator_values = global_values.pop("estimators", {})
    if "mle_k_values" in global_values:
        global_values["mle_k_values"] = tuple(
            int(value) for value in global_values["mle_k_values"]
        )
    if "scaling_quantiles" in global_values:
        global_values["scaling_quantiles"] = tuple(
            float(value) for value in global_values["scaling_quantiles"]
        )
    if "skdim_estimators" in global_values:
        global_values["skdim_estimators"] = tuple(
            str(value) for value in global_values["skdim_estimators"]
        )
    threshold_values = _section(raw, "pca_thresholds")
    if "explained_variance" in threshold_values:
        threshold_values["explained_variance"] = tuple(
            float(value) for value in threshold_values["explained_variance"]
        )

    config = IDEstimationConfig(
        id_estimation_name=str(name),
        input=IDInputConfig(**input_values),
        features=IDFeatureConfig(
            pca_preprocess=PCAProcessConfig(**pca_values),
            **feature_values,
        ),
        groups=IDGroupConfig(**group_values),
        local_id=LocalIDConfig(
            estimators=LocalEstimatorConfig(**local_estimator_values),
            **local_values,
        ),
        global_id=GlobalIDConfig(
            estimators=GlobalEstimatorConfig(**global_estimator_values),
            **global_values,
        ),
        pca_thresholds=PCAThresholdConfig(**threshold_values),
        distance=IDDistanceConfig(**_section(raw, "distance")),
        output=IDOutputConfig(**_section(raw, "output")),
        raw=raw,
    )
    validate_id_config(config)
    return config


def apply_id_overrides(
    raw: dict[str, Any],
    *,
    diagnostics_dir: str | None = None,
    embedding_source: str | None = None,
    feature_space: str | None = None,
    recompute: bool = False,
) -> dict[str, Any]:
    """Apply CLI overrides without mutating the input config."""

    from copy import deepcopy

    updated = deepcopy(raw)
    input_values = updated.setdefault("input", {})
    if diagnostics_dir is not None:
        input_values["diagnostics_dir"] = diagnostics_dir
    if embedding_source is not None:
        input_values["embedding_source"] = embedding_source
        input_values["source_type"] = "npy"
    if feature_space is not None:
        input_values["feature_space_name"] = feature_space
    if recompute:
        updated.setdefault("output", {})["skip_existing"] = False
    return updated


def validate_id_config(config: IDEstimationConfig) -> None:
    """Validate estimator and feature settings."""

    if config.input.source_type not in {"npy", "raw_pixels"}:
        raise ConfigError("input.source_type must be npy or raw_pixels.")
    if config.input.source_type == "npy" and not config.input.embedding_source:
        raise ConfigError("NumPy ID input requires input.embedding_source.")
    if config.input.raw_image_size is not None:
        if len(config.input.raw_image_size) != 2 or min(config.input.raw_image_size) < 1:
            raise ConfigError("input.raw_image_size must contain two positive values.")
    if config.distance.metric not in {"cosine", "euclidean"}:
        raise ConfigError("distance.metric must be cosine or euclidean.")
    if not config.local_id.k_values or min(config.local_id.k_values) < 2:
        raise ConfigError("local_id.k_values must contain values of at least 2.")
    if config.local_id.covariance_eigenvalues < 1:
        raise ConfigError("local_id.covariance_eigenvalues must be positive.")
    if config.global_id.min_group_size < 2:
        raise ConfigError("global_id.min_group_size must be at least 2.")
    if not config.global_id.mle_k_values or min(config.global_id.mle_k_values) < 2:
        raise ConfigError("global_id.mle_k_values must contain values of at least 2.")
    quantiles = config.global_id.scaling_quantiles
    if len(quantiles) < 2 or any(value <= 0 or value >= 1 for value in quantiles):
        raise ConfigError("global_id.scaling_quantiles must be values between 0 and 1.")
    thresholds = config.pca_thresholds.explained_variance
    if not thresholds or any(value <= 0 or value > 1 for value in thresholds):
        raise ConfigError("PCA explained-variance thresholds must be in (0, 1].")
    pca = config.features.pca_preprocess
    if pca.enabled and pca.n_components < 1:
        raise ConfigError("features.pca_preprocess.n_components must be positive.")
    if config.output.overwrite_explorer_data and not config.output.merge_into_explorer_data:
        raise ConfigError(
            "output.overwrite_explorer_data requires merge_into_explorer_data."
        )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Config section {name!r} must be a mapping.")
    return dict(value)
