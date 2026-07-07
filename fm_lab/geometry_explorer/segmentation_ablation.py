"""Ground-truth segmentation-mask ablation datasets."""

from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components

from fm_lab.geometry_explorer.photometric import _relative_to_project
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.utils.config import ConfigError, save_config
from fm_lab.utils.logging import write_json

SEGMENTATION_DATASETS = ("oxford_iiit_pet", "cub200_segmentations")
SEGMENTATION_COMPONENTS = ("raw", "fg", "bg", "mask", "crop")


@dataclass(frozen=True)
class SegmentationAblationConfig:
    dataset: str
    dataset_root: str
    output_root: str = "data/segmentation_ablation"
    split: str = "all"
    image_size: int = 32
    max_samples: int | None = None
    sample_seed: int = 42
    sample_strategy: str = "random"
    fill: str = "mean"
    mask_root: str = ""
    image_root: str = ""
    allow_mask_only: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class SegmentedItem:
    sample_id: str
    split: str
    image_path: Path | None
    mask_path: Path
    label: str
    label_id: int
    source_index: int
    extra: dict[str, Any]


def build_segmentation_ablation(
    config: SegmentationAblationConfig,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare fixed-size image/mask ablation variants from segmentation labels."""

    root = Path(project_root or Path.cwd()).expanduser().resolve()
    _validate_config(config)
    items = _load_items(config, project_root=root)
    positions = _select_positions(
        items,
        maximum=config.max_samples,
        strategy=config.sample_strategy,
        seed=config.sample_seed,
    )
    selected = [items[int(position)] for position in positions]
    output_base = root / config.output_root / config.dataset / config.split
    if output_base.exists():
        if not config.overwrite:
            raise ConfigError(f"Output already exists: {output_base}. Use --overwrite.")
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    masks = np.stack([_load_mask(item, config=config) for item in selected], axis=0)
    mask_metadata = pd.DataFrame.from_records(
        [
            {
                "row_id": index,
                **_item_metadata(item, dataset=config.dataset),
                **_mask_stats(masks[index]),
            }
            for index, item in enumerate(selected)
        ]
    )
    write_parquet(mask_metadata, output_base / "mask_metadata.parquet")
    np.save(output_base / "masks.npy", masks.astype(np.float32))
    _write_preview_grid(
        np.repeat(masks[..., None], 3, axis=-1),
        output_base / "mask_preview_grid.png",
    )

    has_all_images = all(item.image_path is not None for item in selected)
    if not has_all_images and not config.allow_mask_only:
        missing = sum(item.image_path is None for item in selected)
        raise ConfigError(
            f"{config.dataset} has {missing} selected masks without matching images."
        )
    components = (
        _image_components(selected, masks, config=config)
        if has_all_images
        else {"mask": np.repeat(masks[..., None], 3, axis=-1).astype(np.float32)}
    )
    datasets = _write_component_datasets(
        config,
        output_base=output_base,
        components=components,
        metadata=mask_metadata,
        project_root=root,
    )
    metrics_path = output_base / "metrics" / "summary.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(_summary_rows(config, mask_metadata, components)).to_csv(
        metrics_path,
        index=False,
    )
    manifest = {
        "dataset": config.dataset,
        "split": config.split,
        "image_size": config.image_size,
        "rows": int(len(selected)),
        "mask_only": not has_all_images,
        "missing_images": int(sum(item.image_path is None for item in selected)),
        "components": list(components),
        "datasets": [
            {
                "variant_id": row["variant_id"],
                "rows": row["rows"],
                "dataset_config_path": str(row["dataset_config_path"]),
            }
            for row in datasets
        ],
        "metrics_path": str(metrics_path),
    }
    write_json(manifest, output_base / "manifest.json")
    return {
        "dataset": config.dataset,
        "output_dir": output_base,
        "datasets": datasets,
        "metrics_path": metrics_path,
        "mask_only": not has_all_images,
        "missing_images": int(sum(item.image_path is None for item in selected)),
    }


def _load_items(config: SegmentationAblationConfig, *, project_root: Path) -> list[SegmentedItem]:
    if config.dataset == "oxford_iiit_pet":
        return _load_oxford_pet_items(config, project_root=project_root)
    if config.dataset == "cub200_segmentations":
        return _load_cub_segmentation_items(config, project_root=project_root)
    raise ConfigError(f"Unsupported segmentation dataset: {config.dataset}")


def _load_oxford_pet_items(
    config: SegmentationAblationConfig,
    *,
    project_root: Path,
) -> list[SegmentedItem]:
    dataset_root = _resolve(config.dataset_root, project_root)
    _ensure_oxford_pet_extracted(dataset_root)
    annotations = dataset_root / "annotations"
    images = dataset_root / "images"
    split_lookup = _oxford_split_lookup(annotations)
    list_path = annotations / "list.txt"
    rows = _read_oxford_rows(list_path)
    items = []
    for source_index, row in enumerate(rows):
        sample_id = row["sample_id"]
        split = split_lookup.get(sample_id, "all")
        if not _split_matches(split, config.split):
            continue
        image_path = images / f"{sample_id}.jpg"
        mask_path = annotations / "trimaps" / f"{sample_id}.png"
        if not mask_path.is_file():
            continue
        breed = _oxford_breed_name(sample_id)
        items.append(
            SegmentedItem(
                sample_id=sample_id,
                split=split,
                image_path=image_path if image_path.is_file() else None,
                mask_path=mask_path,
                label=breed.replace("_", " "),
                label_id=int(row["class_id"]) - 1,
                source_index=source_index,
                extra={
                    "breed_name": breed,
                    "species": "cat" if int(row["species"]) == 1 else "dog",
                    "species_id": int(row["species"]) - 1,
                    "breed_id": int(row["breed_id"]),
                },
            )
        )
    return _require_items(items, config)


def _load_cub_segmentation_items(
    config: SegmentationAblationConfig,
    *,
    project_root: Path,
) -> list[SegmentedItem]:
    mask_root = _resolve(config.mask_root or config.dataset_root, project_root)
    image_root = _resolve_optional_image_root(config.image_root, project_root)
    items = []
    for source_index, mask_path in enumerate(sorted(mask_root.glob("*/*.png"))):
        class_dir = mask_path.parent.name
        label_id, label = _cub_label(class_dir)
        image_path = None
        if image_root is not None:
            candidate = image_root / class_dir / f"{mask_path.stem}.jpg"
            if candidate.is_file():
                image_path = candidate
        items.append(
            SegmentedItem(
                sample_id=mask_path.stem,
                split="all",
                image_path=image_path,
                mask_path=mask_path,
                label=label,
                label_id=label_id,
                source_index=source_index,
                extra={
                    "class_dir": class_dir,
                    "class_name": label,
                    "has_source_image": image_path is not None,
                },
            )
        )
    return _require_items(items, config)


def _image_components(
    items: list[SegmentedItem],
    masks: np.ndarray,
    *,
    config: SegmentationAblationConfig,
) -> dict[str, np.ndarray]:
    images = np.stack([_load_image(item, config=config) for item in items], axis=0)
    mask = masks[..., None]
    fill = _fill_color(images, mode=config.fill).reshape(1, 1, 1, 3)
    filled = np.broadcast_to(fill, images.shape)
    return {
        "raw": images.astype(np.float32),
        "fg": (mask * images + (1.0 - mask) * filled).astype(np.float32),
        "bg": ((1.0 - mask) * images + mask * filled).astype(np.float32),
        "mask": np.repeat(masks[..., None], 3, axis=-1).astype(np.float32),
        "crop": np.stack(
            [_crop_and_resize(image, masks[index]) for index, image in enumerate(images)],
            axis=0,
        ).astype(np.float32),
    }


def _write_component_datasets(
    config: SegmentationAblationConfig,
    *,
    output_base: Path,
    components: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    project_root: Path,
) -> list[dict[str, Any]]:
    results = []
    for component, images in components.items():
        component_metadata = metadata.copy()
        component_metadata["component"] = component
        component_metadata["sample_type"] = "dataset"
        component_metadata["status"] = "success"
        component_metadata["base_label"] = component_metadata["label"]
        component_metadata["base_label_id"] = component_metadata["label_id"]
        labels = component_metadata["label_id"].to_numpy(dtype=np.int64)
        results.append(
            _write_dataset(
                config,
                component=component,
                output_dir=output_base / component,
                images=images,
                labels=labels,
                metadata=component_metadata,
                project_root=project_root,
            )
        )
    if len(components) > 1:
        images, combined_metadata, labels = _combined_rows(components, metadata)
        results.append(
            _write_dataset(
                config,
                component="combined",
                output_dir=output_base / "combined",
                images=images,
                labels=labels,
                metadata=combined_metadata,
                project_root=project_root,
            )
        )
    return results


def _write_dataset(
    config: SegmentationAblationConfig,
    *,
    component: str,
    output_dir: Path,
    images: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    project_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = np.clip(np.asarray(images, dtype=np.float32), 0.0, 1.0).reshape(
        len(images),
        -1,
    )
    metadata = metadata.reset_index(drop=True).copy()
    metadata["row_id"] = np.arange(len(metadata), dtype=int)
    images_path = output_dir / "images.npy"
    labels_path = output_dir / "labels.npy"
    metadata_path = output_dir / "metadata.parquet"
    dataset_config_path = output_dir / "dataset.yaml"
    np.save(images_path, rows)
    np.save(labels_path, labels)
    write_parquet(metadata, metadata_path)
    _write_preview_grid(images, output_dir / "preview_grid.png")
    variant = f"segmentation_{config.split}_{component}"
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
            "family": config.dataset,
            "variant_id": f"{config.dataset}/{variant}",
            "component": component,
            "rows": int(len(rows)),
            "images_path": str(images_path),
            "labels_path": str(labels_path),
            "metadata_path": str(metadata_path),
            "dataset_config_path": str(dataset_config_path),
        },
        output_dir / "manifest.json",
    )
    return {
        "variant_id": f"{config.dataset}/{variant}",
        "rows": int(len(rows)),
        "output_dir": output_dir,
        "dataset_config_path": dataset_config_path,
    }


def _combined_rows(
    components: dict[str, np.ndarray],
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    image_rows = []
    metadata_rows = []
    labels = []
    for index in range(len(metadata)):
        for component, images in components.items():
            image_rows.append(images[index])
            row = metadata.iloc[[index]].copy()
            row["component"] = component
            row["base_label"] = row["label"].astype(str).to_numpy()
            row["base_label_id"] = row["label_id"].astype(int).to_numpy()
            row["label"] = component
            row["label_id"] = list(components).index(component)
            row["family"] = component
            row["prompt_id"] = f"segmentation_{component}"
            row["prompt"] = f"Segmentation ablation {component}"
            metadata_rows.append(row)
            labels.append(component)
    return (
        np.stack(image_rows, axis=0).astype(np.float32),
        pd.concat(metadata_rows, ignore_index=True),
        np.asarray(labels, dtype="<U16"),
    )


def _dataset_config(
    config: SegmentationAblationConfig,
    *,
    variant: str,
    images_path: Path,
    labels_path: Path,
    metadata_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    return {
        "family": config.dataset,
        "variant": variant,
        "base": "original",
        "split": config.split,
        "seed": config.sample_seed,
        "input": {
            "type": "numpy",
            "data_path": _relative_to_project(images_path, project_root),
            "labels_path": _relative_to_project(labels_path, project_root),
            "metadata_path": _relative_to_project(metadata_path, project_root),
            "image_shape": [config.image_size, config.image_size, 3],
            "value_range": [0.0, 1.0],
            "thumbnail_mode": "atlas",
        },
        "selection": {},
    }


def _item_metadata(item: SegmentedItem, *, dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "split": item.split,
        "sample_id": item.sample_id,
        "source_index": int(item.source_index),
        "label": item.label,
        "label_id": int(item.label_id),
        "class_label": item.label,
        "class_label_id": int(item.label_id),
        "family": item.label,
        "prompt_id": f"{dataset}_{item.sample_id}",
        "prompt": item.label,
        "source_image_path": str(item.image_path) if item.image_path is not None else "",
        "mask_path": str(item.mask_path),
        **item.extra,
    }


def _summary_rows(
    config: SegmentationAblationConfig,
    metadata: pd.DataFrame,
    components: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = [
        {
            "dataset": config.dataset,
            "split": config.split,
            "component": "mask",
            "metric_type": "mask_summary",
            "n_samples": int(len(metadata)),
            "mask_area_ratio_mean": float(metadata["mask_area_ratio"].mean()),
            "mask_area_ratio_median": float(metadata["mask_area_ratio"].median()),
            "valid_mask_rate": float(metadata["valid_mask"].mean()),
            "num_classes": int(metadata["label_id"].nunique()),
        }
    ]
    if {"raw", "fg", "bg", "mask", "crop"}.issubset(components):
        rows.extend(
            {
                "dataset": config.dataset,
                "split": config.split,
                "component": component,
                "metric_type": "component_variance",
                "n_samples": int(len(metadata)),
                "pixel_variance": float(np.var(values)),
                "mean_luminance": float(np.mean(values)),
            }
            for component, values in components.items()
        )
    return rows


def _load_image(item: SegmentedItem, *, config: SegmentationAblationConfig) -> np.ndarray:
    if item.image_path is None:
        raise ConfigError(f"Missing source image for {item.sample_id}")
    from PIL import Image

    image = Image.open(item.image_path).convert("RGB")
    image = image.resize(
        (config.image_size, config.image_size),
        resample=_resampling("bicubic"),
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_mask(item: SegmentedItem, *, config: SegmentationAblationConfig) -> np.ndarray:
    from PIL import Image

    mask_image = Image.open(item.mask_path).convert("L")
    values = np.asarray(mask_image)
    if config.dataset == "oxford_iiit_pet":
        foreground = (values != 2) & (values > 0)
    else:
        foreground = values > 0
    pixels = np.asarray(foreground, dtype=np.uint8) * 255
    resized = Image.fromarray(pixels, mode="L").resize(
        (config.image_size, config.image_size),
        resample=_resampling("box"),
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _mask_stats(mask: np.ndarray) -> dict[str, Any]:
    soft = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    hard = soft >= 0.5
    area = int(np.count_nonzero(hard))
    area_ratio = float(area / hard.size)
    labeled, component_count = connected_components(hard)
    component_sizes = np.bincount(labeled.reshape(-1), minlength=component_count + 1)[1:]
    largest = int(component_sizes.max()) if component_sizes.size else 0
    bbox = _mask_bbox(hard)
    if bbox is None:
        y0 = x0 = y1 = x1 = -1
        bbox_area_ratio = 0.0
    else:
        y0, x0, y1, x1 = bbox
        bbox_area_ratio = float((y1 - y0) * (x1 - x0) / hard.size)
    return {
        "mask_area": area,
        "mask_area_ratio": area_ratio,
        "soft_mask_mass": float(np.mean(soft)),
        "num_components": int(component_count),
        "largest_component_ratio": float(largest / area) if area else 0.0,
        "bbox_y0": int(y0),
        "bbox_x0": int(x0),
        "bbox_y1": int(y1),
        "bbox_x1": int(x1),
        "bbox_area_ratio": bbox_area_ratio,
        "valid_mask": bool(area > 0),
    }


def _crop_and_resize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from PIL import Image

    bbox = _mask_bbox(mask >= 0.5)
    if bbox is None:
        return image.astype(np.float32, copy=True)
    y0, x0, y1, x1 = bbox
    pad = max(1, int(round(image.shape[0] * 0.05)))
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(image.shape[0], y1 + pad)
    x1 = min(image.shape[1], x1 + pad)
    crop = np.asarray(np.clip(image[y0:y1, x0:x1] * 255.0, 0.0, 255.0), dtype=np.uint8)
    resized = Image.fromarray(crop, mode="RGB").resize(
        (image.shape[1], image.shape[0]),
        resample=_resampling("bicubic"),
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    y, x = np.where(mask)
    if len(y) == 0:
        return None
    return int(y.min()), int(x.min()), int(y.max()) + 1, int(x.max()) + 1


def _fill_color(images: np.ndarray, *, mode: str) -> np.ndarray:
    if mode == "gray":
        return np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
    return np.asarray(np.mean(images, axis=(0, 1, 2)), dtype=np.float32)


def _select_positions(
    items: list[SegmentedItem],
    *,
    maximum: int | None,
    strategy: str,
    seed: int,
) -> np.ndarray:
    total = len(items)
    if maximum is None or maximum >= total:
        return np.arange(total, dtype=int)
    if maximum < 1:
        raise ConfigError("--max-samples must be positive when provided.")
    rng = np.random.default_rng(seed)
    if strategy == "random":
        return np.sort(rng.choice(total, size=maximum, replace=False))
    if strategy != "stratified":
        raise ConfigError("--sample-strategy must be random or stratified.")
    labels = np.asarray([item.label_id for item in items], dtype=int)
    unique = np.sort(np.unique(labels))
    per_class = maximum // len(unique)
    remainder = maximum % len(unique)
    selected = []
    for offset, label in enumerate(unique):
        candidates = np.flatnonzero(labels == label)
        take = min(len(candidates), per_class + (1 if offset < remainder else 0))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < maximum:
        missing = maximum - len(selected)
        remaining = np.setdiff1d(np.arange(total, dtype=int), np.asarray(selected))
        selected.extend(rng.choice(remaining, size=missing, replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def _write_preview_grid(
    images: np.ndarray,
    output_path: Path,
    *,
    columns: int = 10,
    max_images: int = 100,
) -> None:
    from PIL import Image

    count = min(len(images), max_images)
    if count == 0:
        return
    tile_size = images.shape[1]
    columns = min(columns, count)
    rows = int(np.ceil(count / columns))
    grid = np.zeros((rows * tile_size, columns * tile_size, 3), dtype=np.uint8)
    pixels = np.asarray(np.round(np.clip(images[:count], 0.0, 1.0) * 255.0), dtype=np.uint8)
    for index, tile in enumerate(pixels):
        row, column = divmod(index, columns)
        y0 = row * tile_size
        x0 = column * tile_size
        grid[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid, mode="RGB").save(output_path)


def _ensure_oxford_pet_extracted(dataset_root: Path) -> None:
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    if images_dir.is_dir() and annotations_dir.is_dir():
        return
    for archive_name in ("images.tar.gz", "annotations.tar.gz"):
        archive_path = dataset_root / archive_name
        if not archive_path.is_file():
            raise ConfigError(f"Missing Oxford Pet archive or extracted folder: {archive_path}")
        _safe_extract_tar(archive_path, dataset_root)


def _safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            target = (output_root / member.name).resolve()
            if output_root not in target.parents and target != output_root:
                raise ConfigError(f"Unsafe tar member path in {archive_path}: {member.name}")
        archive.extractall(output_root)


def _read_oxford_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            sample_id, class_id, species, breed_id = stripped.split()
            rows.append(
                {
                    "sample_id": sample_id,
                    "class_id": class_id,
                    "species": species,
                    "breed_id": breed_id,
                }
            )
    return rows


def _oxford_split_lookup(annotations: Path) -> dict[str, str]:
    lookup = {}
    for name, split in (("trainval.txt", "trainval"), ("test.txt", "test")):
        path = annotations / name
        if not path.is_file():
            continue
        for row in _read_oxford_rows(path):
            lookup[row["sample_id"]] = split
    return lookup


def _split_matches(item_split: str, requested: str) -> bool:
    if requested == "all":
        return True
    if requested == "train":
        requested = "trainval"
    return item_split == requested


def _oxford_breed_name(sample_id: str) -> str:
    return sample_id.rsplit("_", 1)[0]


def _cub_label(class_dir: str) -> tuple[int, str]:
    prefix, _, name = class_dir.partition(".")
    label_id = int(prefix) - 1 if prefix.isdigit() else 0
    return label_id, name.replace("_", " ")


def _resolve_optional_image_root(value: str, project_root: Path) -> Path | None:
    candidates = []
    if value:
        candidates.append(_resolve(value, project_root))
    candidates.extend(
        [
            project_root / "data" / "CUB_200_2011" / "images",
            project_root / "data" / "CUB_200_2011" / "CUB_200_2011" / "images",
            project_root / "data" / "cub200" / "images",
            project_root / "data" / "cub_200_2011" / "images",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _require_items(
    items: list[SegmentedItem],
    config: SegmentationAblationConfig,
) -> list[SegmentedItem]:
    if not items:
        raise ConfigError(
            f"No segmentation rows found for {config.dataset} split {config.split!r}."
        )
    return items


def _validate_config(config: SegmentationAblationConfig) -> None:
    if config.dataset not in SEGMENTATION_DATASETS:
        raise ConfigError(f"--dataset must be one of {SEGMENTATION_DATASETS}.")
    if config.split not in {"all", "train", "trainval", "test"}:
        raise ConfigError("--split must be all, train/trainval, or test.")
    if config.image_size < 16:
        raise ConfigError("--image-size must be at least 16.")
    if config.fill not in {"mean", "gray"}:
        raise ConfigError("--fill must be mean or gray.")


def _resolve(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _resampling(name: str) -> Any:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    return {
        "bicubic": resampling.BICUBIC,
        "box": resampling.BOX,
    }[name]
