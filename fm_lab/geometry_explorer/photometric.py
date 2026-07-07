"""Photometric MNIST/Fashion-MNIST ladder generation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import load_dataset
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.utils.config import ConfigError, save_config
from fm_lab.utils.logging import write_json


@dataclass(frozen=True)
class PhotometricLevel:
    key: str
    code: int
    name: str


@dataclass(frozen=True)
class PhotometricBuildConfig:
    family: str
    dataset_root: str
    output_root: str = "data"
    split: str = "all"
    order: str = "source"
    base_samples: int = 10_000
    variants_per_base: int = 5
    clean_variants_per_base: int = 1
    seed: int = 42
    levels: tuple[str, ...] = ()
    overwrite: bool = False


PHOTOMETRIC_LEVELS: tuple[PhotometricLevel, ...] = (
    PhotometricLevel("level_00_clean", 0, "Clean"),
    PhotometricLevel("level_01_global", 1, "Global brightness/contrast/gamma"),
    PhotometricLevel("level_02_illumination", 2, "Smooth illumination"),
    PhotometricLevel("level_04_background", 4, "Structured background"),
    PhotometricLevel("level_05_foreground_texture", 5, "Foreground texture"),
    PhotometricLevel("level_06_full", 6, "Full photometric rendering"),
)
DEFAULT_LEVEL_KEYS = tuple(level.key for level in PHOTOMETRIC_LEVELS)

FASHION_MNIST_LABELS = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def build_photometric_ladder(
    config: PhotometricBuildConfig,
    *,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Generate paired photometric variants and registration configs."""

    root = Path(project_root or Path.cwd()).expanduser().resolve()
    if config.family not in {"mnist", "fashion_mnist"}:
        raise ConfigError("Photometric ladder supports only mnist and fashion_mnist.")
    if config.base_samples < 1:
        raise ConfigError("--base-samples must be positive.")
    if config.variants_per_base < 1:
        raise ConfigError("--variants-per-base must be positive.")
    if config.clean_variants_per_base < 1:
        raise ConfigError("--clean-variants-per-base must be positive.")

    levels = _resolve_levels(config.levels or DEFAULT_LEVEL_KEYS)
    source = load_dataset(
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
        project_root=root,
        thumbnail_dir=None,
    )
    if source.vectors is None:
        raise ConfigError("Photometric ladder requires vector-backed image data.")

    base_vectors = np.asarray(source.vectors, dtype=np.float32)
    if base_vectors.ndim != 2 or base_vectors.shape[1] != 28 * 28:
        raise ConfigError(
            f"Expected flattened 28x28 images, got {base_vectors.shape}."
        )
    base_metadata = source.metadata.reset_index(drop=True)
    output_base = root / config.output_root / _photometric_dataset_name(config.family)
    output_base.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for level in levels:
        level_variants = _level_variants_per_base(config, level)
        level_dir = output_base / level.key
        if level_dir.exists() and any(level_dir.iterdir()) and not config.overwrite:
            raise ConfigError(
                f"Photometric output already exists: {level_dir}. "
                "Pass --overwrite to replace generated files."
            )
        level_dir.mkdir(parents=True, exist_ok=True)
        rows, labels, metadata = _render_level(
            base_vectors,
            base_metadata,
            family=config.family,
            level=level,
            variants_per_base=level_variants,
            seed=config.seed,
        )
        images_path = level_dir / "images.npy"
        labels_path = level_dir / "labels.npy"
        base_indices_path = level_dir / "base_indices.npy"
        variant_indices_path = level_dir / "variant_indices.npy"
        metadata_path = level_dir / "metadata.parquet"
        dataset_config_path = level_dir / "dataset.yaml"

        np.save(images_path, rows)
        np.save(labels_path, labels)
        np.save(
            base_indices_path,
            metadata["base_index"].to_numpy(dtype=np.int64),
        )
        np.save(
            variant_indices_path,
            metadata["variant_index"].to_numpy(dtype=np.int64),
        )
        write_parquet(metadata, metadata_path)
        save_config(
            _level_manifest(config, level, rows, source, level_variants),
            level_dir / "config.yaml",
        )
        _write_preview_grid(
            rows,
            output_path=level_dir / "preview_grid.png",
            variants_per_base=level_variants,
        )
        dataset_config = _dataset_variant_config(
            config,
            level,
            images_path=images_path,
            labels_path=labels_path,
            metadata_path=metadata_path,
            project_root=root,
        )
        save_config(dataset_config, dataset_config_path)
        write_json(
            {
                "family": config.family,
                "level": level.key,
                "rows": int(len(rows)),
                "base_samples": int(len(base_vectors)),
                "variants_per_base": int(level_variants),
                "images_path": str(images_path),
                "labels_path": str(labels_path),
                "metadata_path": str(metadata_path),
                "dataset_config_path": str(dataset_config_path),
            },
            level_dir / "manifest.json",
        )
        results.append(
            {
                "family": config.family,
                "level": level.key,
                "variant_id": f"{config.family}/photometric_{level.key}",
                "rows": int(len(rows)),
                "output_dir": level_dir,
                "dataset_config_path": dataset_config_path,
            }
        )
    return results


def render_photometric_variant(
    image: np.ndarray,
    *,
    family: str,
    level: str | PhotometricLevel,
    base_index: int,
    variant_index: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render one deterministic grayscale photometric variant."""

    resolved = _resolve_level(level)
    rng = np.random.default_rng(
        _stable_seed(seed, family, resolved.key, base_index, variant_index)
    )
    original = np.asarray(image, dtype=np.float32).reshape(28, 28)
    original = np.clip(original, 0.0, 1.0)
    alpha = _alpha_mask(original, family=family)
    brightness = 0.0
    contrast = 1.0
    gamma = 1.0
    illumination_strength = 0.0
    illumination_angle = 0.0
    background_strength = 0.0
    foreground_texture_strength = 0.0

    if resolved.code == 0:
        rendered = original
    elif resolved.code == 1:
        contrast = float(rng.uniform(0.5, 1.5))
        brightness = float(rng.uniform(-0.2, 0.2))
        gamma = float(rng.uniform(0.6, 1.8))
        rendered = np.clip(contrast * np.power(alpha, gamma) + brightness, 0.0, 1.0)
    elif resolved.code == 2:
        illumination, illumination_strength, illumination_angle = _illumination_field(rng)
        rendered = np.clip(alpha * illumination, 0.0, 1.0)
    elif resolved.code == 4:
        background, background_strength = _background_field(rng)
        foreground = float(rng.uniform(0.65, 1.0))
        rendered = np.clip(alpha * foreground + (1.0 - alpha) * background, 0.0, 1.0)
    elif resolved.code == 5:
        texture, foreground_texture_strength = _foreground_texture(rng)
        foreground_base = _foreground_detail(original, family=family)
        foreground = np.clip(foreground_base + texture, 0.0, 1.0)
        rendered = np.clip(alpha * foreground, 0.0, 1.0)
    elif resolved.code == 6:
        contrast = float(rng.uniform(0.75, 1.25))
        brightness = float(rng.uniform(-0.08, 0.08))
        gamma = float(rng.uniform(0.8, 1.35))
        illumination, illumination_strength, illumination_angle = _illumination_field(rng)
        background, background_strength = _background_field(rng)
        texture, foreground_texture_strength = _foreground_texture(rng)
        foreground_base = _foreground_detail(original, family=family)
        foreground = np.clip(foreground_base * illumination + texture, 0.0, 1.0)
        rendered = alpha * foreground + (1.0 - alpha) * background
        rendered = np.clip(contrast * np.power(rendered, gamma) + brightness, 0.0, 1.0)
    else:  # pragma: no cover - guarded by _resolve_levels
        raise ConfigError(f"Unsupported photometric level: {resolved.key}")

    stats = _image_stats(rendered)
    stats.update(
        {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "gamma": float(gamma),
            "illumination_strength": float(illumination_strength),
            "illumination_angle": float(illumination_angle),
            "background_strength": float(background_strength),
            "foreground_texture_strength": float(foreground_texture_strength),
        }
    )
    return rendered.astype(np.float32, copy=False), stats


def _render_level(
    base_vectors: np.ndarray,
    base_metadata: pd.DataFrame,
    *,
    family: str,
    level: PhotometricLevel,
    variants_per_base: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    total = len(base_vectors) * variants_per_base
    rows = np.empty((total, 28 * 28), dtype=np.float32)
    labels = np.empty(total, dtype="<U64")
    records: list[dict[str, Any]] = []
    dataset_name = _photometric_dataset_name(family)

    for base_position, vector in enumerate(base_vectors):
        base_row = base_metadata.iloc[base_position]
        label, label_id = _label_values(base_row, family=family)
        base_index = int(base_row.get("original_index", base_position))
        base_source_index = int(base_row.get("source_index", base_position))
        split = str(base_row.get("split", ""))
        for variant_index in range(variants_per_base):
            position = base_position * variants_per_base + variant_index
            rendered, stats = render_photometric_variant(
                vector,
                family=family,
                level=level,
                base_index=base_index,
                variant_index=variant_index,
                seed=seed,
            )
            rows[position] = rendered.reshape(-1)
            labels[position] = label
            records.append(
                {
                    "row_id": position,
                    "dataset": dataset_name,
                    "split": split,
                    "label": label,
                    "label_id": label_id,
                    "family": label,
                    "prompt_id": f"{dataset_name}_{label_id}",
                    "prompt": f"{dataset_name.replace('_', ' ')} {label}",
                    "tags": [dataset_name, str(label), level.key],
                    "sample_type": "photometric_dataset",
                    "status": "success",
                    "photometric_level": level.key,
                    "photometric_level_code": level.code,
                    "photometric_level_name": level.name,
                    "base_index": base_index,
                    "base_source_index": base_source_index,
                    "base_position": base_position,
                    "variant_index": variant_index,
                    "orbit_id": f"{family}:{base_index}",
                    **stats,
                }
            )
    return rows, labels, pd.DataFrame.from_records(records)


def _alpha_mask(image: np.ndarray, *, family: str) -> np.ndarray:
    if family == "fashion_mnist":
        high = float(np.max(image))
        if high <= np.finfo(np.float32).eps:
            return np.zeros_like(image, dtype=np.float32)
        return np.clip(image / high, 0.0, 1.0).astype(np.float32)
    low, high = np.quantile(image, (0.01, 0.99))
    scale = float(high - low)
    if scale <= np.finfo(np.float32).eps:
        return np.zeros_like(image, dtype=np.float32)
    alpha = np.clip((image - low) / scale, 0.0, 1.0)
    return np.power(alpha, 0.8).astype(np.float32)


def _foreground_detail(image: np.ndarray, *, family: str) -> np.ndarray:
    if family == "fashion_mnist":
        return np.clip(0.35 + 0.65 * image, 0.0, 1.0).astype(np.float32)
    return np.full_like(image, 0.85, dtype=np.float32)


def _illumination_field(rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, 28, dtype=np.float32),
        np.linspace(-1.0, 1.0, 28, dtype=np.float32),
        indexing="ij",
    )
    angle = float(rng.uniform(-math.pi, math.pi))
    strength = float(rng.uniform(0.15, 0.55))
    directional = np.cos(angle) * xx + np.sin(angle) * yy
    low_noise = _smooth_noise(rng, grid_size=5)
    field = 0.7 * directional + 0.3 * low_noise
    field = _normalize_zero_one(field)
    low = float(rng.uniform(0.45, 0.8))
    high = float(rng.uniform(1.05, 1.75))
    illumination = low + (high - low) * field
    illumination = 1.0 + strength * (illumination - float(np.mean(illumination)))
    return np.clip(illumination, 0.25, 2.0).astype(np.float32), strength, angle


def _background_field(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    base = float(rng.uniform(0.12, 0.5))
    low_strength = float(rng.uniform(0.12, 0.4))
    mid_strength = float(rng.uniform(0.03, 0.18))
    low = _smooth_noise(rng, grid_size=4)
    mid = _smooth_noise(rng, grid_size=10)
    background = base + low_strength * low + mid_strength * mid
    background = np.clip(background, 0.0, 1.0).astype(np.float32)
    return background, float(np.std(background))


def _foreground_texture(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    strength = float(rng.uniform(0.05, 0.22))
    low = _smooth_noise(rng, grid_size=7)
    high = rng.normal(0.0, 1.0, size=(28, 28)).astype(np.float32)
    texture = strength * (0.7 * low + 0.3 * high)
    return texture.astype(np.float32), strength


def _smooth_noise(rng: np.random.Generator, *, grid_size: int) -> np.ndarray:
    from PIL import Image

    values = rng.normal(0.0, 1.0, size=(grid_size, grid_size)).astype(np.float32)
    normalized = _normalize_zero_one(values)
    pixels = np.asarray(np.round(normalized * 255.0), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="L").resize(
        (28, 28),
        resample=Image.Resampling.BICUBIC,
    )
    field = np.asarray(image, dtype=np.float32) / 255.0
    field = field - float(np.mean(field))
    scale = float(np.std(field))
    if scale <= np.finfo(np.float32).eps:
        return np.zeros((28, 28), dtype=np.float32)
    return (field / scale).astype(np.float32)


def _normalize_zero_one(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    scale = max(high - low, float(np.finfo(np.float32).eps))
    return ((values - low) / scale).astype(np.float32)


def _image_stats(image: np.ndarray) -> dict[str, float]:
    dy, dx = np.gradient(image.astype(np.float32))
    edge = np.sqrt(dx * dx + dy * dy)
    lowpass = _lowpass_7x7(image)
    high_frequency = image - lowpass
    return {
        "mean_luminance": float(np.mean(image)),
        "contrast_stat": float(np.std(image)),
        "edge_density": float(np.mean(edge)),
        "high_frequency_energy": float(np.mean(np.abs(high_frequency))),
    }


def _lowpass_7x7(image: np.ndarray) -> np.ndarray:
    from PIL import Image

    pixels = np.asarray(np.round(np.clip(image, 0.0, 1.0) * 255.0), dtype=np.uint8)
    small = Image.fromarray(pixels, mode="L").resize(
        (7, 7),
        resample=Image.Resampling.BICUBIC,
    )
    large = small.resize((28, 28), resample=Image.Resampling.BICUBIC)
    return np.asarray(large, dtype=np.float32) / 255.0


def _write_preview_grid(
    rows: np.ndarray,
    *,
    output_path: Path,
    variants_per_base: int,
    max_base_rows: int = 10,
) -> None:
    from PIL import Image

    base_rows = min(max_base_rows, max(1, len(rows) // variants_per_base))
    grid = np.zeros((base_rows * 28, variants_per_base * 28), dtype=np.uint8)
    for base_position in range(base_rows):
        for variant_index in range(variants_per_base):
            row_index = base_position * variants_per_base + variant_index
            if row_index >= len(rows):
                continue
            tile = rows[row_index].reshape(28, 28)
            pixels = np.asarray(np.round(np.clip(tile, 0.0, 1.0) * 255.0), dtype=np.uint8)
            y0 = base_position * 28
            x0 = variant_index * 28
            grid[y0 : y0 + 28, x0 : x0 + 28] = pixels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="L").save(output_path)


def _dataset_variant_config(
    config: PhotometricBuildConfig,
    level: PhotometricLevel,
    *,
    images_path: Path,
    labels_path: Path,
    metadata_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "family": config.family,
        "variant": f"photometric_{level.key}",
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


def _level_manifest(
    config: PhotometricBuildConfig,
    level: PhotometricLevel,
    rows: np.ndarray,
    source: Any,
    variants_per_base: int,
) -> dict[str, Any]:
    return {
        "family": config.family,
        "source_dataset_root": config.dataset_root,
        "source_split": config.split,
        "source_order": config.order,
        "source_id": source.source_id,
        "level": {
            "key": level.key,
            "code": level.code,
            "name": level.name,
        },
        "base_samples": int(source.vectors.shape[0]) if source.vectors is not None else 0,
        "variants_per_base": int(variants_per_base),
        "rows": int(len(rows)),
        "seed": int(config.seed),
        "image_shape": [28, 28],
        "value_range": [0.0, 1.0],
    }


def _level_variants_per_base(
    config: PhotometricBuildConfig,
    level: PhotometricLevel,
) -> int:
    return config.clean_variants_per_base if level.code == 0 else config.variants_per_base


def _label_values(row: pd.Series, *, family: str) -> tuple[str, int]:
    if "label_id" in row and not pd.isna(row["label_id"]):
        label_id = int(row["label_id"])
    else:
        label_id = int(row.get("label", 0))
    if family == "fashion_mnist":
        label = str(row.get("label", FASHION_MNIST_LABELS[label_id]))
    else:
        label = str(label_id)
    return label, label_id


def _resolve_levels(values: tuple[str, ...]) -> tuple[PhotometricLevel, ...]:
    levels = tuple(_resolve_level(value) for value in values)
    if not levels:
        raise ConfigError("At least one photometric level is required.")
    return levels


def _resolve_level(value: str | PhotometricLevel) -> PhotometricLevel:
    if isinstance(value, PhotometricLevel):
        return value
    for level in PHOTOMETRIC_LEVELS:
        if value in {level.key, str(level.code), f"{level.code:02d}", level.name}:
            return level
    supported = ", ".join(level.key for level in PHOTOMETRIC_LEVELS)
    raise ConfigError(f"Unknown photometric level {value!r}. Supported: {supported}.")


def _stable_seed(
    seed: int,
    family: str,
    level: str,
    base_index: int,
    variant_index: int,
) -> int:
    payload = f"{seed}:{family}:{level}:{base_index}:{variant_index}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _photometric_dataset_name(family: str) -> str:
    return "photometric_fashion_mnist" if family == "fashion_mnist" else "photometric_mnist"


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())
