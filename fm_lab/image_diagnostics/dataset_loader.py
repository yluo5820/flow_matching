"""Dataset adapters that produce feature-ready rows and optional thumbnails."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fm_lab.data.mnist import MNISTImages
from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.metadata_loader import MetadataLoadResult, load_image_metadata
from fm_lab.image_diagnostics.palette import LABEL_PALETTE
from fm_lab.utils.config import ConfigError


@dataclass(frozen=True)
class DatasetBundle:
    metadata: pd.DataFrame
    vectors: np.ndarray | None
    source_id: str
    source_description: str
    total_rows: int
    skipped_rows: int = 0


def load_dataset(
    config: InputConfig,
    *,
    project_root: str | Path | None = None,
    thumbnail_dir: str | Path | None = None,
) -> DatasetBundle:
    """Load a configured dataset into aligned metadata and optional raw vectors."""

    root = Path(project_root or Path.cwd()).expanduser().resolve()
    if config.type == "mnist":
        return _load_mnist(config, root, thumbnail_dir)
    if config.type == "numpy":
        return _load_numpy(config, root, thumbnail_dir)
    return _load_image_metadata(config, root)


def _load_mnist(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    images, labels, split_values, original_indices = _load_mnist_arrays(
        config,
        dataset_root,
    )
    indices = _sample_indices(len(images), config.max_samples, config.sample_seed)
    vectors = np.asarray(images[indices], dtype=np.float32)
    selected_labels = labels[indices].astype(int)
    image_paths = [""] * len(indices)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_mnist_sprite_atlases(
            vectors,
            selected_labels,
            output_dir=Path(thumbnail_dir).parent / "atlases",
        )
    elif config.thumbnail_mode == "files":
        image_paths = _export_grayscale_thumbnails(
            vectors,
            image_shape=(28, 28),
            value_range=(0.0, 1.0),
            source_indices=indices,
            output_dir=thumbnail_dir,
            prefix="mnist",
        )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(indices)),
            "image_path": image_paths,
            "dataset": "mnist",
            "split": split_values[indices],
            "label": [str(value) for value in selected_labels],
            "family": [str(value) for value in selected_labels],
            "prompt_id": [f"digit_{value}" for value in selected_labels],
            "prompt": [f"MNIST digit {value}" for value in selected_labels],
            "tags": [["mnist", f"digit_{value}"] for value in selected_labels],
            "source_index": indices,
            "original_index": original_indices[indices],
            "sample_type": "dataset",
            "status": "success",
            **atlas_metadata,
        }
    )
    source_files = sorted(dataset_root.glob("*-idx*-ubyte.gz"))
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            source_files,
            extra=f"{config.split}:{config.order}:{indices.tolist()}",
        ),
        source_description=(
            f"MNIST {config.split} split at {dataset_root} ({config.order} order)"
        ),
        total_rows=len(images),
    )


def _load_mnist_arrays(
    config: InputConfig,
    dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splits = ("train", "test") if config.split == "all" else (config.split,)
    image_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    offset = 0
    for split in splits:
        dataset = MNISTImages(
            root=dataset_root,
            train=split == "train",
            download=config.download,
            normalize="zero_one",
        )
        images = dataset.images.numpy()
        labels = dataset.labels.numpy()
        order = (
            np.argsort(labels, kind="stable")
            if config.order == "mldata"
            else np.arange(len(images))
        )
        image_parts.append(images[order])
        label_parts.append(labels[order])
        split_parts.append(np.asarray([split] * len(order), dtype=object))
        index_parts.append(np.arange(offset, offset + len(order), dtype=int)[order])
        offset += len(order)
    return (
        np.concatenate(image_parts),
        np.concatenate(label_parts),
        np.concatenate(split_parts),
        np.concatenate(index_parts),
    )


def _export_mnist_sprite_atlases(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    output_dir: str | Path,
    tile_size: int = 28,
    atlas_size: int = 2048,
) -> dict[str, object]:
    from PIL import Image

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    columns = atlas_size // tile_size
    capacity = columns * columns
    positions = np.arange(len(vectors), dtype=int)
    atlas_indices = positions // capacity
    atlas_columns = positions % capacity % columns
    atlas_rows = positions % capacity // columns
    digest = hashlib.sha256()
    digest.update(b"grayscale-alpha-v2")
    digest.update(np.asarray(labels, dtype=np.uint8).tobytes())
    digest.update(str(len(vectors)).encode())
    prefix = f"mnist_reference_{digest.hexdigest()[:12]}"
    atlas_paths = [
        directory / f"{prefix}_{index:02d}.png"
        for index in range(max(1, int(atlas_indices.max()) + 1 if len(vectors) else 1))
    ]
    if not all(path.exists() for path in atlas_paths):
        for path in directory.glob("mnist_reference_*.png"):
            if path not in atlas_paths:
                path.unlink()
        pixels = np.asarray(np.round(vectors.reshape(-1, tile_size, tile_size) * 255.0))
        pixels = np.asarray(pixels, dtype=np.uint8)
        for atlas_index, path in enumerate(atlas_paths):
            start = atlas_index * capacity
            end = min(len(vectors), start + capacity)
            atlas = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)
            for local_index, sample_index in enumerate(range(start, end)):
                row, column = divmod(local_index, columns)
                color = LABEL_PALETTE[int(labels[sample_index]) % len(LABEL_PALETTE)]
                tile = atlas[
                    row * tile_size : (row + 1) * tile_size,
                    column * tile_size : (column + 1) * tile_size,
                ]
                tile[..., :3] = color
                tile[..., 3] = pixels[sample_index]
            Image.fromarray(atlas, mode="RGBA").save(path, optimize=True)
    resolved_paths = [str(path.resolve()) for path in atlas_paths]
    return {
        "sprite_atlas_path": [resolved_paths[index] for index in atlas_indices],
        "sprite_atlas_index": atlas_indices,
        "sprite_atlas_column": atlas_columns,
        "sprite_atlas_row": atlas_rows,
        "sprite_tile_size": tile_size,
        "sprite_atlas_columns": columns,
        "sprite_atlas_size": atlas_size,
    }


def _load_numpy(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    data_path = _resolve(config.data_path, project_root)
    if not data_path.exists():
        raise ConfigError(f"NumPy input does not exist: {data_path}")
    vectors = np.load(data_path, mmap_mode="r")
    if vectors.ndim != 2:
        raise ConfigError(f"NumPy input must have shape (samples, features): {vectors.shape}")
    indices = _sample_indices(len(vectors), config.max_samples, config.sample_seed)
    selected = np.asarray(vectors[indices], dtype=np.float32)
    labels = _load_labels(config.labels_path, project_root, len(vectors))
    selected_labels = labels[indices] if labels is not None else np.asarray([""] * len(indices))
    image_paths = [""] * len(indices)
    if config.image_shape is not None:
        if int(np.prod(config.image_shape)) != selected.shape[1]:
            raise ConfigError(
                f"input.image_shape={config.image_shape} does not match "
                f"feature dimension {selected.shape[1]}."
            )
        value_range = config.value_range or (
            float(np.nanmin(selected)),
            float(np.nanmax(selected)),
        )
        image_paths = _export_grayscale_thumbnails(
            selected,
            image_shape=config.image_shape,
            value_range=value_range,
            source_indices=indices,
            output_dir=thumbnail_dir,
            prefix=data_path.stem,
        )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(indices)),
            "image_path": image_paths,
            "dataset": data_path.stem,
            "split": "",
            "label": [str(value) for value in selected_labels],
            "family": [str(value) for value in selected_labels],
            "prompt_id": [str(value) for value in selected_labels],
            "prompt": "",
            "tags": [[] for _ in indices],
            "source_index": indices,
            "sample_type": "array",
            "status": "success",
        }
    )
    source_files = [data_path]
    if config.labels_path:
        source_files.append(_resolve(config.labels_path, project_root))
    return DatasetBundle(
        metadata=metadata,
        vectors=selected,
        source_id=_files_source_id(source_files, extra=str(indices.tolist())),
        source_description=f"NumPy array {data_path}",
        total_rows=len(vectors),
    )


def _load_image_metadata(config: InputConfig, project_root: Path) -> DatasetBundle:
    result: MetadataLoadResult = load_image_metadata(config, project_root=project_root)
    metadata = result.frame.copy()
    if "dataset" not in metadata:
        metadata["dataset"] = Path(config.experiment_dir).name
    if "sample_type" not in metadata:
        metadata["sample_type"] = "image"
    if "label" not in metadata:
        metadata["label"] = ""
    source_files = [result.metadata_path]
    source_files.extend(Path(path) for path in metadata.get("image_path", []))
    return DatasetBundle(
        metadata=metadata,
        vectors=None,
        source_id=_files_source_id(source_files),
        source_description=f"Image metadata {result.metadata_path}",
        total_rows=result.total_rows,
        skipped_rows=(
            result.missing_images + result.duplicate_rows + result.malformed_rows
        ),
    )


def _export_grayscale_thumbnails(
    vectors: np.ndarray,
    *,
    image_shape: tuple[int, int],
    value_range: tuple[float, float],
    source_indices: np.ndarray,
    output_dir: str | Path | None,
    prefix: str,
) -> list[str]:
    if output_dir is None:
        return [""] * len(vectors)
    from PIL import Image

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    low, high = value_range
    scale = max(high - low, np.finfo(np.float32).eps)
    paths: list[str] = []
    for vector, source_index in zip(vectors, source_indices, strict=False):
        normalized = np.clip((vector.reshape(image_shape) - low) / scale, 0.0, 1.0)
        pixels = np.asarray(np.round(normalized * 255.0), dtype=np.uint8)
        path = directory / f"{prefix}_{int(source_index):05d}.png"
        if not path.exists():
            Image.fromarray(pixels, mode="L").save(path)
        paths.append(str(path.resolve()))
    return paths


def _load_labels(
    path_value: str | None,
    project_root: Path,
    expected_rows: int,
) -> np.ndarray | None:
    if not path_value:
        return None
    path = _resolve(path_value, project_root)
    if not path.exists():
        raise ConfigError(f"Label input does not exist: {path}")
    if path.suffix.lower() == ".npy":
        labels = np.load(path)
    else:
        frame = pd.read_csv(path)
        if frame.shape[1] != 1:
            raise ConfigError("CSV labels must contain exactly one column.")
        labels = frame.iloc[:, 0].to_numpy()
    labels = np.asarray(labels).reshape(-1)
    if len(labels) != expected_rows:
        raise ConfigError(
            f"Label count {len(labels)} does not match sample count {expected_rows}."
        )
    return labels


def _sample_indices(total: int, maximum: int | None, seed: int) -> np.ndarray:
    if maximum is None or maximum >= total:
        return np.arange(total, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=maximum, replace=False))


def _files_source_id(paths: list[Path], *, extra: str = "") -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.expanduser().resolve()
        digest.update(str(resolved).encode())
        if resolved.exists():
            stat = resolved.stat()
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    digest.update(extra.encode())
    return digest.hexdigest()


def _resolve(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
