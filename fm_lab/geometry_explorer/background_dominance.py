"""Foreground/background dominance confirmation datasets and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors

from fm_lab.geometry_explorer.photometric import (
    _alpha_mask,
    _background_field,
    _foreground_detail,
    _foreground_texture,
    _illumination_field,
    _image_stats,
    _label_values,
    _relative_to_project,
    _resolve_level,
    _stable_seed,
    _write_preview_grid,
)
from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle, load_dataset
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.utils.config import ConfigError, save_config
from fm_lab.utils.logging import write_json

COMPONENTS = ("full", "foreground", "background", "mask")
DEFAULT_EXPERIMENTS = ("a", "b", "c", "d", "e")
DEFAULT_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
DEFAULT_BACKGROUND_POOL_SIZES = (10, 20, 50, 100)
DEFAULT_FOREGROUND_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)


@dataclass(frozen=True)
class BackgroundDominanceConfig:
    family: str
    dataset_root: str
    output_root: str = "data"
    split: str = "all"
    order: str = "source"
    base_samples: int = 10_000
    variants_per_base: int = 5
    seed: int = 42
    level: str = "level_04_background"
    experiments: tuple[str, ...] = DEFAULT_EXPERIMENTS
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS
    background_pool_sizes: tuple[int, ...] = DEFAULT_BACKGROUND_POOL_SIZES
    foreground_scales: tuple[float, ...] = DEFAULT_FOREGROUND_SCALES
    metrics_max_samples: int = 5_000
    metrics_pairs: int = 100_000
    overwrite: bool = False


@dataclass(frozen=True)
class RenderedComponents:
    full: np.ndarray
    foreground: np.ndarray
    background: np.ndarray
    mask: np.ndarray
    foreground_black: np.ndarray
    background_black: np.ndarray
    metadata: dict[str, float | int | str]


def build_background_dominance_experiments(
    config: BackgroundDominanceConfig,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate foreground/background dominance datasets and metric summaries."""

    root = Path(project_root or Path.cwd()).expanduser().resolve()
    _validate_config(config)
    level = _resolve_level(config.level)
    source = _load_source(config, project_root=root)
    vectors = np.asarray(source.vectors, dtype=np.float32)
    metadata = source.metadata.reset_index(drop=True)
    constant_gray = float(np.mean(vectors))
    output_base = root / config.output_root / _dominance_dataset_name(config.family)
    output_base.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    if "a" in config.experiments or "b" in config.experiments:
        component = _build_experiment_a(
            config,
            level_key=level.key,
            vectors=vectors,
            source_metadata=metadata,
            constant_gray=constant_gray,
            output_base=output_base,
            project_root=root,
        )
        results.extend(component["datasets"])
        metrics_rows.extend(component["metrics"])
    if "c" in config.experiments:
        sweep = _build_experiment_c(
            config,
            level_key=level.key,
            vectors=vectors,
            source_metadata=metadata,
            constant_gray=constant_gray,
            output_base=output_base,
            project_root=root,
        )
        results.extend(sweep["datasets"])
        metrics_rows.extend(sweep["metrics"])
    if "d" in config.experiments:
        shared = _build_experiment_d(
            config,
            level_key=level.key,
            vectors=vectors,
            source_metadata=metadata,
            constant_gray=constant_gray,
            output_base=output_base,
            project_root=root,
        )
        results.extend(shared["datasets"])
        metrics_rows.extend(shared["metrics"])
    if "e" in config.experiments:
        area = _build_experiment_e(
            config,
            level_key=level.key,
            vectors=vectors,
            source_metadata=metadata,
            constant_gray=constant_gray,
            output_base=output_base,
            project_root=root,
        )
        results.extend(area["datasets"])
        metrics_rows.extend(area["metrics"])

    metrics_path = output_base / "metrics" / "summary.csv"
    if metrics_rows:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_frame = pd.DataFrame.from_records(metrics_rows)
        metrics_frame.to_csv(metrics_path, index=False)
        _write_summary_plots(metrics_frame, metrics_path.parent)
    write_json(
        {
            "family": config.family,
            "level": level.key,
            "base_samples": int(len(vectors)),
            "variants_per_base": int(config.variants_per_base),
            "experiments": list(config.experiments),
            "datasets": [
                {
                    "variant_id": result["variant_id"],
                    "rows": result["rows"],
                    "dataset_config_path": str(result["dataset_config_path"]),
                }
                for result in results
            ],
            "metrics_path": str(metrics_path) if metrics_rows else None,
        },
        output_base / "manifest.json",
    )
    return {
        "family": config.family,
        "output_dir": output_base,
        "datasets": results,
        "metrics_path": metrics_path if metrics_rows else None,
    }


def render_foreground_background_components(
    image: np.ndarray,
    *,
    family: str,
    level: str,
    base_index: int,
    variant_index: int,
    seed: int,
    constant_gray: float,
    background_lambda: float = 1.0,
    background_template_id: int | None = None,
    foreground_scale: float = 1.0,
) -> RenderedComponents:
    """Render full, foreground, background, and mask component views."""

    resolved = _resolve_level(level)
    rng = np.random.default_rng(
        _stable_seed(seed, family, resolved.key, base_index, variant_index)
    )
    original = np.clip(np.asarray(image, dtype=np.float32).reshape(28, 28), 0.0, 1.0)
    alpha = _alpha_mask(original, family=family)
    foreground = _foreground_detail(original, family=family)
    brightness = 0.0
    contrast = 1.0
    gamma = 1.0
    illumination_strength = 0.0
    illumination_angle = 0.0
    foreground_texture_strength = 0.0

    if resolved.code == 6:
        contrast = float(rng.uniform(0.75, 1.25))
        brightness = float(rng.uniform(-0.08, 0.08))
        gamma = float(rng.uniform(0.8, 1.35))
        illumination, illumination_strength, illumination_angle = _illumination_field(rng)
        background, background_strength = _sample_background(
            rng,
            family=family,
            seed=seed,
            template_id=background_template_id,
        )
        texture, foreground_texture_strength = _foreground_texture(rng)
        foreground = np.clip(foreground * illumination + texture, 0.0, 1.0)
    elif resolved.code == 4:
        background, background_strength = _sample_background(
            rng,
            family=family,
            seed=seed,
            template_id=background_template_id,
        )
        foreground = np.full_like(original, float(rng.uniform(0.65, 1.0)))
    else:
        background, background_strength = _sample_background(
            rng,
            family=family,
            seed=seed,
            template_id=background_template_id,
        )

    if foreground_scale != 1.0:
        alpha = _scale_center(alpha, foreground_scale)
        foreground = _scale_center(foreground, foreground_scale)

    background = np.clip(
        (1.0 - background_lambda) * constant_gray + background_lambda * background,
        0.0,
        1.0,
    )
    fg_black = np.clip(alpha * foreground, 0.0, 1.0)
    bg_black = np.clip((1.0 - alpha) * background, 0.0, 1.0)
    foreground_mean = np.clip(fg_black + (1.0 - alpha) * constant_gray, 0.0, 1.0)
    background_mean = np.clip(bg_black + alpha * constant_gray, 0.0, 1.0)
    full = np.clip(fg_black + bg_black, 0.0, 1.0)
    if resolved.code == 6:
        full = np.clip(contrast * np.power(full, gamma) + brightness, 0.0, 1.0)

    stats = _component_metadata(
        full=full,
        background=background,
        alpha=alpha,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma,
        illumination_strength=illumination_strength,
        illumination_angle=illumination_angle,
        background_strength=background_strength,
        foreground_texture_strength=foreground_texture_strength,
        background_lambda=background_lambda,
        background_template_id=background_template_id,
        foreground_scale=foreground_scale,
    )
    return RenderedComponents(
        full=full.astype(np.float32),
        foreground=foreground_mean.astype(np.float32),
        background=background_mean.astype(np.float32),
        mask=alpha.astype(np.float32),
        foreground_black=fg_black.astype(np.float32),
        background_black=bg_black.astype(np.float32),
        metadata=stats,
    )


def _build_experiment_a(
    config: BackgroundDominanceConfig,
    *,
    level_key: str,
    vectors: np.ndarray,
    source_metadata: pd.DataFrame,
    constant_gray: float,
    output_base: Path,
    project_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_component, labels, metadata = _component_arrays(
        config,
        level_key=level_key,
        vectors=vectors,
        source_metadata=source_metadata,
        constant_gray=constant_gray,
    )
    output_dir = output_base / "experiment_a" / level_key
    datasets: list[dict[str, Any]] = []
    for component in COMPONENTS:
        component_metadata = metadata.copy()
        component_metadata["component"] = component
        datasets.append(
            _write_component_dataset(
                config,
                rows_by_component[component],
                labels,
                component_metadata,
                experiment="a",
                condition=f"{level_key}_{component}",
                output_dir=output_dir / component,
                project_root=project_root,
            )
        )
    combined_rows = np.concatenate([rows_by_component[name] for name in COMPONENTS], axis=0)
    combined_metadata_parts = []
    combined_labels: list[str] = []
    for component in COMPONENTS:
        part = metadata.copy()
        part["component"] = component
        part["class_label"] = part["label"]
        part["label"] = component
        combined_metadata_parts.append(part)
        combined_labels.extend([component] * len(part))
    datasets.append(
        _write_component_dataset(
            config,
            combined_rows,
            np.asarray(combined_labels, dtype="<U64"),
            pd.concat(combined_metadata_parts, ignore_index=True),
            experiment="a",
            condition=f"{level_key}_combined",
            output_dir=output_dir / "combined",
            project_root=project_root,
        )
    )
    metrics = []
    if "b" in config.experiments:
        metrics.append(
            _component_metric_row(
                rows_by_component,
                metadata,
                dataset=config.family,
                experiment="b",
                condition=level_key,
                max_samples=config.metrics_max_samples,
                num_pairs=config.metrics_pairs,
                seed=config.seed,
            )
        )
    metrics.append(
        _component_metric_row(
            rows_by_component,
            metadata,
            dataset=config.family,
            experiment="a",
            condition=level_key,
            max_samples=config.metrics_max_samples,
            num_pairs=config.metrics_pairs,
            seed=config.seed,
        )
    )
    return {"datasets": datasets, "metrics": metrics}


def _build_experiment_c(
    config: BackgroundDominanceConfig,
    *,
    level_key: str,
    vectors: np.ndarray,
    source_metadata: pd.DataFrame,
    constant_gray: float,
    output_base: Path,
    project_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    datasets: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for lambda_value in config.lambdas:
        rows_by_component, labels, metadata = _component_arrays(
            config,
            level_key=level_key,
            vectors=vectors,
            source_metadata=source_metadata,
            constant_gray=constant_gray,
            background_lambda=float(lambda_value),
        )
        condition = f"lambda_{_value_token(lambda_value)}"
        datasets.append(
            _write_component_dataset(
                config,
                rows_by_component["full"],
                labels,
                metadata,
                experiment="c",
                condition=condition,
                output_dir=output_base / "experiment_c" / condition,
                project_root=project_root,
            )
        )
        metrics.append(
            _component_metric_row(
                rows_by_component,
                metadata,
                dataset=config.family,
                experiment="c",
                condition=condition,
                max_samples=config.metrics_max_samples,
                num_pairs=config.metrics_pairs,
                seed=config.seed,
            )
        )
    return {"datasets": datasets, "metrics": metrics}


def _build_experiment_d(
    config: BackgroundDominanceConfig,
    *,
    level_key: str,
    vectors: np.ndarray,
    source_metadata: pd.DataFrame,
    constant_gray: float,
    output_base: Path,
    project_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    datasets: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for pool_size in config.background_pool_sizes:
        for mode in ("independent", "class_correlated", "shared_across_classes"):
            rows_by_component, labels, metadata = _component_arrays(
                config,
                level_key=level_key,
                vectors=vectors,
                source_metadata=source_metadata,
                constant_gray=constant_gray,
                background_id_mode=mode,
                background_pool_size=int(pool_size),
            )
            condition = f"k{int(pool_size):03d}_{mode}"
            datasets.append(
                _write_component_dataset(
                    config,
                    rows_by_component["full"],
                    labels,
                    metadata,
                    experiment="d",
                    condition=condition,
                    output_dir=output_base / "experiment_d" / f"k{int(pool_size):03d}" / mode,
                    project_root=project_root,
                )
            )
            metrics.append(
                _component_metric_row(
                    rows_by_component,
                    metadata,
                    dataset=config.family,
                    experiment="d",
                    condition=condition,
                    max_samples=config.metrics_max_samples,
                    num_pairs=config.metrics_pairs,
                    seed=config.seed,
                )
            )
    return {"datasets": datasets, "metrics": metrics}


def _build_experiment_e(
    config: BackgroundDominanceConfig,
    *,
    level_key: str,
    vectors: np.ndarray,
    source_metadata: pd.DataFrame,
    constant_gray: float,
    output_base: Path,
    project_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    datasets: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for scale in config.foreground_scales:
        rows_by_component, labels, metadata = _component_arrays(
            config,
            level_key=level_key,
            vectors=vectors,
            source_metadata=source_metadata,
            constant_gray=constant_gray,
            foreground_scale=float(scale),
        )
        condition = f"scale_{_value_token(scale)}"
        datasets.append(
            _write_component_dataset(
                config,
                rows_by_component["full"],
                labels,
                metadata,
                experiment="e",
                condition=condition,
                output_dir=output_base / "experiment_e" / condition,
                project_root=project_root,
            )
        )
        metrics.append(
            _component_metric_row(
                rows_by_component,
                metadata,
                dataset=config.family,
                experiment="e",
                condition=condition,
                max_samples=config.metrics_max_samples,
                num_pairs=config.metrics_pairs,
                seed=config.seed,
            )
        )
    return {"datasets": datasets, "metrics": metrics}


def _component_arrays(
    config: BackgroundDominanceConfig,
    *,
    level_key: str,
    vectors: np.ndarray,
    source_metadata: pd.DataFrame,
    constant_gray: float,
    background_lambda: float = 1.0,
    background_id_mode: str = "per_sample",
    background_pool_size: int = 0,
    foreground_scale: float = 1.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    total = len(vectors) * config.variants_per_base
    rows = {name: np.empty((total, 28 * 28), dtype=np.float32) for name in COMPONENTS}
    labels = np.empty(total, dtype="<U64")
    records: list[dict[str, Any]] = []
    for base_position, vector in enumerate(vectors):
        source_row = source_metadata.iloc[base_position]
        label, label_id = _label_values(source_row, family=config.family)
        base_index = int(source_row.get("original_index", base_position))
        base_source_index = int(source_row.get("source_index", base_position))
        split = str(source_row.get("split", ""))
        for variant_index in range(config.variants_per_base):
            position = base_position * config.variants_per_base + variant_index
            background_id = _background_template_id(
                mode=background_id_mode,
                pool_size=background_pool_size,
                seed=config.seed,
                base_position=base_position,
                base_index=base_index,
                variant_index=variant_index,
                label_id=label_id,
            )
            rendered = render_foreground_background_components(
                vector,
                family=config.family,
                level=level_key,
                base_index=base_index,
                variant_index=variant_index,
                seed=config.seed,
                constant_gray=constant_gray,
                background_lambda=background_lambda,
                background_template_id=background_id,
                foreground_scale=foreground_scale,
            )
            for component in COMPONENTS:
                rows[component][position] = getattr(rendered, component).reshape(-1)
            labels[position] = label
            records.append(
                {
                    "row_id": position,
                    "dataset": _dominance_dataset_name(config.family),
                    "split": split,
                    "label": label,
                    "label_id": label_id,
                    "family": label,
                    "prompt_id": f"background_dominance_{config.family}_{label_id}",
                    "prompt": f"background dominance {config.family} {label}",
                    "tags": [
                        _dominance_dataset_name(config.family),
                        str(label),
                        level_key,
                    ],
                    "sample_type": "background_dominance",
                    "status": "success",
                    "photometric_level": level_key,
                    "base_index": base_index,
                    "base_source_index": base_source_index,
                    "base_position": base_position,
                    "variant_index": variant_index,
                    "orbit_id": f"{config.family}:{base_index}",
                    **rendered.metadata,
                }
            )
    return rows, labels, pd.DataFrame.from_records(records)


def _write_component_dataset(
    config: BackgroundDominanceConfig,
    rows: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    *,
    experiment: str,
    condition: str,
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise ConfigError(
            f"Background dominance output already exists: {output_dir}. "
            "Pass --overwrite to replace generated files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = metadata.reset_index(drop=True).copy()
    metadata["row_id"] = np.arange(len(metadata), dtype=int)
    metadata["experiment"] = experiment
    metadata["condition"] = condition
    labels = np.asarray(labels, dtype="<U64")

    images_path = output_dir / "images.npy"
    labels_path = output_dir / "labels.npy"
    metadata_path = output_dir / "metadata.parquet"
    dataset_config_path = output_dir / "dataset.yaml"
    np.save(images_path, rows)
    np.save(labels_path, labels)
    np.save(output_dir / "base_indices.npy", metadata["base_index"].to_numpy(dtype=np.int64))
    np.save(
        output_dir / "variant_indices.npy",
        metadata["variant_index"].to_numpy(dtype=np.int64),
    )
    write_parquet(metadata, metadata_path)
    _write_preview_grid(rows, output_path=output_dir / "preview_grid.png", variants_per_base=1)
    variant = f"background_{experiment}_{condition}"
    save_config(
        _dataset_config(
            config,
            variant=variant,
            images_path=images_path,
            labels_path=labels_path,
            metadata_path=metadata_path,
            project_root=project_root,
        ),
        dataset_config_path,
    )
    write_json(
        {
            "family": config.family,
            "variant_id": f"{config.family}/{variant}",
            "experiment": experiment,
            "condition": condition,
            "rows": int(len(rows)),
            "images_path": str(images_path),
            "labels_path": str(labels_path),
            "metadata_path": str(metadata_path),
            "dataset_config_path": str(dataset_config_path),
        },
        output_dir / "manifest.json",
    )
    return {
        "variant_id": f"{config.family}/{variant}",
        "rows": int(len(rows)),
        "output_dir": output_dir,
        "dataset_config_path": dataset_config_path,
    }


def _component_metric_row(
    rows: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    *,
    dataset: str,
    experiment: str,
    condition: str,
    max_samples: int,
    num_pairs: int,
    seed: int,
) -> dict[str, Any]:
    positions = _metric_positions(len(metadata), max_samples=max_samples, seed=seed)
    sampled = metadata.iloc[positions].reset_index(drop=True)
    full = np.asarray(rows["full"][positions], dtype=np.float32)
    foreground = np.asarray(rows["foreground"][positions], dtype=np.float32)
    background = np.asarray(rows["background"][positions], dtype=np.float32)
    mask = np.asarray(rows["mask"][positions], dtype=np.float32)
    labels = sampled["label"].astype(str).to_numpy()
    background_ids = sampled.get("background_template_id", pd.Series([-1] * len(sampled)))
    background_labels = background_ids.astype(str).to_numpy()
    pair_metrics = _distance_pair_metrics(
        full,
        foreground,
        background,
        mask,
        num_pairs=num_pairs,
        seed=seed,
    )
    overlap = _knn_overlap_metrics(
        full,
        foreground,
        background,
        mask,
        labels=labels,
        background_labels=background_labels,
        seed=seed,
    )
    return {
        "dataset": dataset,
        "experiment": experiment,
        "condition": condition,
        "n_samples": int(len(sampled)),
        **pair_metrics,
        **overlap,
    }


def _write_summary_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if frame.empty:
        return
    plot_specs = (
        (
            "distance_correlations.png",
            ("spearman_full_bg", "spearman_full_fg", "spearman_full_mask"),
            "Spearman distance correlation with full image",
        ),
        (
            "knn_purity.png",
            ("knn_class_purity_full", "knn_background_purity_full"),
            "kNN purity in full-image space",
        ),
        (
            "background_fraction.png",
            ("bg_fraction_median", "fg_fraction_median", "mask_fraction_median"),
            "Median component distance fraction",
        ),
    )
    labels = [
        f"{row.experiment}:{row.condition}"
        for row in frame[["experiment", "condition"]].itertuples(index=False)
    ]
    x = np.arange(len(frame), dtype=float)
    for filename, columns, title in plot_specs:
        existing = [column for column in columns if column in frame]
        if not existing:
            continue
        width = min(0.8 / max(1, len(existing)), 0.35)
        figure, axis = plt.subplots(figsize=(max(8.0, len(frame) * 0.45), 4.8))
        for offset, column in enumerate(existing):
            values = frame[column].to_numpy(dtype=float)
            axis.bar(
                x + (offset - (len(existing) - 1) / 2) * width,
                values,
                width=width,
                label=column,
            )
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)


def _distance_pair_metrics(
    full: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    *,
    num_pairs: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(full)
    pair_count = min(num_pairs, max(1, n * max(1, n - 1)))
    left = rng.integers(0, n, size=pair_count)
    right = rng.integers(0, n, size=pair_count)
    same = left == right
    if np.any(same):
        right[same] = (right[same] + 1) % n
    d_full = _squared_distances(full, left, right)
    d_fg = _squared_distances(foreground, left, right)
    d_bg = _squared_distances(background, left, right)
    d_mask = _squared_distances(mask, left, right)
    eps = float(np.finfo(np.float32).eps)
    return {
        "bg_fraction_median": float(np.median(d_bg / (d_full + eps))),
        "fg_fraction_median": float(np.median(d_fg / (d_full + eps))),
        "mask_fraction_median": float(np.median(d_mask / (d_full + eps))),
        "pearson_full_bg": _pearson(d_full, d_bg),
        "pearson_full_fg": _pearson(d_full, d_fg),
        "pearson_full_mask": _pearson(d_full, d_mask),
        "spearman_full_bg": _spearman(d_full, d_bg),
        "spearman_full_fg": _spearman(d_full, d_fg),
        "spearman_full_mask": _spearman(d_full, d_mask),
    }


def _knn_overlap_metrics(
    full: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    *,
    labels: np.ndarray,
    background_labels: np.ndarray,
    seed: int,
    k: int = 15,
) -> dict[str, float]:
    del seed
    k = min(k, max(1, len(full) - 1))
    full_neighbors = _knn_indices(full, k)
    fg_neighbors = _knn_indices(foreground, k)
    bg_neighbors = _knn_indices(background, k)
    mask_neighbors = _knn_indices(mask, k)
    return {
        "knn_class_purity_full": _neighbor_purity(full_neighbors, labels),
        "knn_class_purity_fg": _neighbor_purity(fg_neighbors, labels),
        "knn_class_purity_bg": _neighbor_purity(bg_neighbors, labels),
        "knn_class_purity_mask": _neighbor_purity(mask_neighbors, labels),
        "knn_background_purity_full": _neighbor_purity(full_neighbors, background_labels),
        "knn_background_purity_bg": _neighbor_purity(bg_neighbors, background_labels),
        "knn_overlap_full_fg": _neighbor_overlap(full_neighbors, fg_neighbors),
        "knn_overlap_full_bg": _neighbor_overlap(full_neighbors, bg_neighbors),
        "knn_overlap_full_mask": _neighbor_overlap(full_neighbors, mask_neighbors),
    }


def _knn_indices(values: np.ndarray, k: int) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(values)
    return neighbors.kneighbors(return_distance=False)


def _neighbor_purity(neighbors: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(labels[neighbors] == labels[:, None]))


def _neighbor_overlap(left: np.ndarray, right: np.ndarray) -> float:
    overlaps = []
    for left_row, right_row in zip(left, right, strict=False):
        overlaps.append(len(set(left_row.tolist()) & set(right_row.tolist())) / len(left_row))
    return float(np.mean(overlaps))


def _squared_distances(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = values[left] - values[right]
    return np.einsum("ij,ij->i", diff, diff)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(pearsonr(left, right).statistic)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _metric_positions(total: int, *, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_samples, replace=False))


def _component_metadata(
    *,
    full: np.ndarray,
    background: np.ndarray,
    alpha: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float,
    illumination_strength: float,
    illumination_angle: float,
    background_strength: float,
    foreground_texture_strength: float,
    background_lambda: float,
    background_template_id: int | None,
    foreground_scale: float,
) -> dict[str, float | int | str]:
    stats = _image_stats(full)
    background_stats = _image_stats(background)
    foreground_area = float(np.mean(alpha > 0.05))
    soft_foreground_mass = float(np.mean(alpha))
    stats.update(
        {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "gamma": float(gamma),
            "illumination_strength": float(illumination_strength),
            "illumination_angle": float(illumination_angle),
            "background_strength": float(background_strength),
            "foreground_texture_strength": float(foreground_texture_strength),
            "background_lambda": float(background_lambda),
            "background_template_id": int(background_template_id or -1),
            "mean_background_luminance": float(np.mean(background)),
            "background_contrast_stat": float(background_stats["contrast_stat"]),
            "background_low_frequency_stat": float(background_stats["edge_density"]),
            "foreground_scale": float(foreground_scale),
            "foreground_area": foreground_area,
            "soft_foreground_mass": soft_foreground_mass,
        }
    )
    return stats


def _sample_background(
    rng: np.random.Generator,
    *,
    family: str,
    seed: int,
    template_id: int | None,
) -> tuple[np.ndarray, float]:
    if template_id is None:
        return _background_field(rng)
    template_rng = np.random.default_rng(
        _stable_seed(seed, family, "background_template", int(template_id), 0)
    )
    return _background_field(template_rng)


def _background_template_id(
    *,
    mode: str,
    pool_size: int,
    seed: int,
    base_position: int,
    base_index: int,
    variant_index: int,
    label_id: int,
) -> int | None:
    if mode == "per_sample" or pool_size <= 0:
        return _stable_seed(seed, "background", "per_sample", base_index, variant_index)
    if mode == "independent":
        template_seed = _stable_seed(
            seed,
            "background",
            "independent",
            base_index,
            variant_index,
        )
        return template_seed % pool_size
    if mode == "class_correlated":
        per_class = max(1, pool_size // 10)
        return (label_id * per_class + variant_index % per_class) % pool_size
    if mode == "shared_across_classes":
        del base_position
        return (base_index + variant_index) % pool_size
    raise ConfigError(f"Unknown background id mode: {mode}")


def _scale_center(image: np.ndarray, scale: float) -> np.ndarray:
    from PIL import Image

    if scale <= 0:
        raise ConfigError("Foreground scale must be positive.")
    if abs(scale - 1.0) < 1e-6:
        return image.astype(np.float32, copy=True)
    size = max(1, int(round(28 * scale)))
    pixels = np.asarray(np.round(np.clip(image, 0.0, 1.0) * 255.0), dtype=np.uint8)
    resized = Image.fromarray(pixels, mode="L").resize(
        (size, size),
        resample=Image.Resampling.BICUBIC,
    )
    values = np.asarray(resized, dtype=np.float32) / 255.0
    output = np.zeros((28, 28), dtype=np.float32)
    if size <= 28:
        y0 = (28 - size) // 2
        x0 = (28 - size) // 2
        output[y0 : y0 + size, x0 : x0 + size] = values
    else:
        y0 = (size - 28) // 2
        x0 = (size - 28) // 2
        output = values[y0 : y0 + 28, x0 : x0 + 28]
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _dataset_config(
    config: BackgroundDominanceConfig,
    *,
    variant: str,
    images_path: Path,
    labels_path: Path,
    metadata_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "family": config.family,
        "variant": variant,
        "base": "original",
        "split": config.split,
        "seed": config.seed,
        "input": {
            "type": "numpy",
            "data_path": _relative_to_project(images_path, project_root),
            "labels_path": _relative_to_project(labels_path, project_root),
            "metadata_path": _relative_to_project(metadata_path, project_root),
            "image_shape": [28, 28],
            "value_range": [0.0, 1.0],
            "thumbnail_mode": "atlas",
        },
        "selection": {},
    }


def _load_source(
    config: BackgroundDominanceConfig,
    *,
    project_root: Path,
) -> DatasetBundle:
    dataset = load_dataset(
        InputConfig(
            type=config.family,
            dataset_root=config.dataset_root,
            split=config.split,
            order=config.order,
            thumbnail_mode="none",
            max_samples=config.base_samples,
            sample_seed=config.seed,
            download=False,
        ),
        project_root=project_root,
        thumbnail_dir=None,
    )
    if dataset.vectors is None:
        raise ConfigError("Background dominance experiments require vector-backed data.")
    if dataset.vectors.shape[1] != 28 * 28:
        raise ConfigError(f"Expected flattened 28x28 data, got {dataset.vectors.shape}.")
    return dataset


def _validate_config(config: BackgroundDominanceConfig) -> None:
    if config.family not in {"mnist", "fashion_mnist"}:
        raise ConfigError("Background dominance experiments support mnist and fashion_mnist.")
    if config.base_samples < 1:
        raise ConfigError("--base-samples must be positive.")
    if config.variants_per_base < 1:
        raise ConfigError("--variants-per-base must be positive.")
    supported = set(DEFAULT_EXPERIMENTS)
    unknown = set(config.experiments) - supported
    if unknown:
        raise ConfigError(f"Unknown background dominance experiments: {sorted(unknown)}")


def _dominance_dataset_name(family: str) -> str:
    return (
        "background_dominance_fashion_mnist"
        if family == "fashion_mnist"
        else "background_dominance_mnist"
    )


def _value_token(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")
