"""Dataset adapters that produce feature-ready rows and optional thumbnails."""

from __future__ import annotations

import gzip
import hashlib
import logging
import struct
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fm_lab.data.mnist import MNISTImages
from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.metadata_loader import MetadataLoadResult, load_image_metadata
from fm_lab.image_diagnostics.palette import LABEL_PALETTE
from fm_lab.utils.config import ConfigError

LOGGER = logging.getLogger("fm_lab.image_diagnostics")

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
CIFAR10_DIRECTORY = "cifar-10-batches-bin"
CIFAR10_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
CIFAR100_MD5 = "03b5dce01913d631647c71ecec9e9cb8"
CIFAR100_DIRECTORY = "cifar-100-binary"
CINIC10_ZIP_URL = "https://datashare.ed.ac.uk/download/DS_10283_3192.zip"
CINIC10_ZIP = "DS_10283_3192.zip"
CINIC10_ARCHIVE = "CINIC-10.tar.gz"
CINIC10_LABELS = CIFAR10_LABELS
CINIC10_SPLITS = ("train", "valid", "test")
FASHION_MNIST_URL_ROOT = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
    "master/data/fashion"
)
FASHION_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
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


@dataclass(frozen=True)
class DatasetBundle:
    metadata: pd.DataFrame
    vectors: np.ndarray | None
    source_id: str
    source_description: str
    total_rows: int
    skipped_rows: int = 0
    image_shape: tuple[int, ...] | None = None
    value_range: tuple[float, float] | None = None


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
    if config.type == "fashion_mnist":
        return _load_fashion_mnist(config, root, thumbnail_dir)
    if config.type == "cifar10":
        return _load_cifar10(config, root, thumbnail_dir)
    if config.type == "cifar100":
        return _load_cifar100(config, root, thumbnail_dir)
    if config.type == "cinic10":
        return _load_cinic10(config, root, thumbnail_dir)
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
        atlas_metadata = _export_grayscale_sprite_atlases(
            vectors,
            selected_labels,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix="mnist_reference",
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
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
    )


def _load_fashion_mnist(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    source_files = _ensure_fashion_mnist(
        dataset_root,
        split=config.split,
        download=config.download,
    )
    images, labels, split_values, original_indices = _load_fashion_mnist_arrays(
        config,
        dataset_root,
    )
    indices = _sample_indices(len(images), config.max_samples, config.sample_seed)
    vectors = (
        np.asarray(images[indices], dtype=np.float32).reshape(len(indices), -1)
        / 255.0
    )
    selected_labels = labels[indices].astype(int)
    label_names = np.asarray(
        [FASHION_MNIST_LABELS[value] for value in selected_labels],
        dtype=object,
    )
    image_paths = [""] * len(indices)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_grayscale_sprite_atlases(
            vectors,
            selected_labels,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix="fashion_mnist",
        )
    elif config.thumbnail_mode == "files":
        image_paths = _export_grayscale_thumbnails(
            vectors,
            image_shape=(28, 28),
            value_range=(0.0, 1.0),
            source_indices=indices,
            output_dir=thumbnail_dir,
            prefix="fashion_mnist",
        )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(indices)),
            "image_path": image_paths,
            "dataset": "fashion_mnist",
            "split": split_values[indices],
            "label": label_names,
            "label_id": selected_labels,
            "family": label_names,
            "prompt_id": [f"fashion_mnist_{value}" for value in selected_labels],
            "prompt": [f"Fashion-MNIST {value}" for value in label_names],
            "tags": [["fashion_mnist", str(value)] for value in label_names],
            "source_index": indices,
            "original_index": original_indices[indices],
            "sample_type": "dataset",
            "status": "success",
            **atlas_metadata,
        }
    )
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            source_files,
            extra=f"{config.split}:{config.order}:{indices.tolist()}",
        ),
        source_description=(
            f"Fashion-MNIST {config.split} split at {dataset_root} "
            f"({config.order} order)"
        ),
        total_rows=len(images),
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
    )


def _load_cifar10(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    data_dir = _ensure_cifar10(
        dataset_root,
        split=config.split,
        download=config.download,
    )
    images, labels, split_values, original_indices, source_files = (
        _load_cifar10_arrays(config, data_dir)
    )
    indices = _sample_indices(len(images), config.max_samples, config.sample_seed)
    selected_images = np.asarray(images[indices], dtype=np.uint8)
    selected_labels = labels[indices].astype(int)
    label_names = np.asarray(
        [CIFAR10_LABELS[value] for value in selected_labels],
        dtype=object,
    )
    grayscale = config.color_mode == "grayscale"
    dataset_name = "cifar10_grayscale" if grayscale else "cifar10"
    display_name = "CIFAR-10 grayscale" if grayscale else "CIFAR-10"
    if grayscale:
        feature_images = _rgb_to_grayscale(selected_images)
        atlas_images = np.repeat(feature_images[..., None], 3, axis=-1)
        vectors = feature_images.reshape(len(feature_images), -1)
        image_shape = (32, 32)
    else:
        feature_images = selected_images
        atlas_images = selected_images
        vectors = selected_images.reshape(len(selected_images), -1)
        image_shape = (32, 32, 3)
    image_paths = [""] * len(indices)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_rgb_sprite_atlases(
            atlas_images,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix=dataset_name,
        )
    elif config.thumbnail_mode == "files":
        if grayscale:
            image_paths = _export_grayscale_thumbnails(
                vectors,
                image_shape=(32, 32),
                value_range=(0.0, 255.0),
                source_indices=indices,
                output_dir=thumbnail_dir,
                prefix=dataset_name,
            )
        else:
            image_paths = _export_rgb_thumbnails(
                feature_images,
                source_indices=indices,
                output_dir=thumbnail_dir,
                prefix=dataset_name,
            )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(indices)),
            "image_path": image_paths,
            "dataset": dataset_name,
            "split": split_values[indices],
            "label": label_names,
            "label_id": selected_labels,
            "color_mode": config.color_mode,
            "family": label_names,
            "prompt_id": [f"{dataset_name}_{value}" for value in label_names],
            "prompt": [f"{display_name} {value}" for value in label_names],
            "tags": [
                [dataset_name, str(value), config.color_mode]
                for value in label_names
            ],
            "source_index": indices,
            "original_index": original_indices[indices],
            "sample_type": "dataset",
            "status": "success",
            **atlas_metadata,
        }
    )
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            source_files,
            extra=f"{config.split}:{config.color_mode}:{indices.tolist()}",
        ),
        source_description=(
            f"{display_name} {config.split} split at {dataset_root}"
        ),
        total_rows=len(images),
        image_shape=image_shape,
        value_range=(0.0, 255.0),
    )


def _load_cifar100(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    data_dir = _ensure_cifar100(
        dataset_root,
        split=config.split,
        download=config.download,
    )
    label_names_source = _load_cifar100_label_names(data_dir)
    images, labels, split_values, original_indices, source_files = (
        _load_cifar100_arrays(config, data_dir)
    )
    indices = _sample_indices(len(images), config.max_samples, config.sample_seed)
    selected_images = np.asarray(images[indices], dtype=np.uint8)
    selected_labels = labels[indices].astype(int)
    label_names = np.asarray(
        [label_names_source[value] for value in selected_labels],
        dtype=object,
    )
    dataset_name = "cifar100"
    display_name = "CIFAR-100"
    feature_images = selected_images
    atlas_images = selected_images
    vectors = selected_images.reshape(len(selected_images), -1)
    image_shape = (32, 32, 3)
    image_paths = [""] * len(indices)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_rgb_sprite_atlases(
            atlas_images,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix=dataset_name,
        )
    elif config.thumbnail_mode == "files":
        image_paths = _export_rgb_thumbnails(
            feature_images,
            source_indices=indices,
            output_dir=thumbnail_dir,
            prefix=dataset_name,
        )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(indices)),
            "image_path": image_paths,
            "dataset": dataset_name,
            "split": split_values[indices],
            "label": label_names,
            "label_id": selected_labels,
            "color_mode": config.color_mode,
            "family": label_names,
            "prompt_id": [f"{dataset_name}_{value}" for value in label_names],
            "prompt": [f"{display_name} {value}" for value in label_names],
            "tags": [
                [dataset_name, str(value), config.color_mode]
                for value in label_names
            ],
            "source_index": indices,
            "original_index": original_indices[indices],
            "sample_type": "dataset",
            "status": "success",
            **atlas_metadata,
        }
    )
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            source_files,
            extra=f"{config.split}:{config.color_mode}:{indices.tolist()}",
        ),
        source_description=f"{display_name} {config.split} split at {dataset_root}",
        total_rows=len(images),
        image_shape=image_shape,
        value_range=(0.0, 255.0),
    )


def _load_cinic10(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    source = _ensure_cinic10(
        dataset_root,
        split=config.split,
        download=config.download,
    )
    entries = _cinic10_entries(source, config.split)
    indices = _cinic10_sample_indices(
        entries,
        maximum=config.max_samples,
        seed=config.sample_seed,
        strategy=config.sample_strategy,
    )
    selected_entries = [entries[int(index)] for index in indices]
    selected_images, selected_labels, split_values, original_indices = (
        _load_cinic10_images(source, selected_entries)
    )
    label_names = np.asarray(
        [CINIC10_LABELS[value] for value in selected_labels],
        dtype=object,
    )
    dataset_name = "cinic10"
    display_name = "CINIC-10"
    vectors = selected_images.reshape(len(selected_images), -1)
    image_paths = [""] * len(indices)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_rgb_sprite_atlases(
            selected_images,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix=dataset_name,
        )
    elif config.thumbnail_mode == "files":
        image_paths = _export_rgb_thumbnails(
            selected_images,
            source_indices=indices,
            output_dir=thumbnail_dir,
            prefix=dataset_name,
        )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(indices)),
            "image_path": image_paths,
            "dataset": dataset_name,
            "split": split_values,
            "label": label_names,
            "label_id": selected_labels,
            "color_mode": config.color_mode,
            "family": label_names,
            "prompt_id": [f"{dataset_name}_{value}" for value in label_names],
            "prompt": [f"{display_name} {value}" for value in label_names],
            "tags": [
                [dataset_name, str(value), config.color_mode]
                for value in label_names
            ],
            "source_index": indices,
            "original_index": original_indices,
            "sample_type": "dataset",
            "status": "success",
            **atlas_metadata,
        }
    )
    source_files = [source] if source.is_file() else _cinic10_split_paths(source, config.split)
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            source_files,
            extra=f"{config.split}:{config.color_mode}:{indices.tolist()}",
        ),
        source_description=f"{display_name} {config.split} split at {dataset_root}",
        total_rows=len(entries),
        image_shape=(32, 32, 3),
        value_range=(0.0, 255.0),
    )


def _rgb_to_grayscale(images: np.ndarray) -> np.ndarray:
    weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    luminance = np.tensordot(
        np.asarray(images, dtype=np.float32),
        weights,
        axes=([-1], [0]),
    )
    return np.asarray(np.round(luminance), dtype=np.uint8)


def _ensure_fashion_mnist(
    dataset_root: Path,
    *,
    split: str,
    download: bool,
) -> list[Path]:
    expected = _fashion_mnist_paths(dataset_root, split)
    invalid = [
        path
        for path in expected
        if not path.is_file() or _file_md5(path) != FASHION_MNIST_FILES[path.name]
    ]
    if not invalid:
        return expected
    if not download:
        raise ConfigError(
            f"Fashion-MNIST does not exist or failed checksum under {dataset_root}. "
            "Set input.download: true to fetch it."
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    for destination in invalid:
        checksum = FASHION_MNIST_FILES[destination.name]
        if destination.exists():
            destination.unlink()
        partial = Path(f"{destination}.part")
        if partial.exists() and _file_md5(partial) == checksum:
            partial.replace(destination)
            continue
        LOGGER.info("Downloading Fashion-MNIST file %s", destination.name)
        _download_resumable(
            f"{FASHION_MNIST_URL_ROOT}/{destination.name}",
            partial,
        )
        if _file_md5(partial) != checksum:
            partial.unlink(missing_ok=True)
            _download_resumable(
                f"{FASHION_MNIST_URL_ROOT}/{destination.name}",
                partial,
            )
        if _file_md5(partial) != checksum:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded Fashion-MNIST file failed MD5 checksum: "
                f"{destination.name}"
            )
        partial.replace(destination)
    return expected


def _fashion_mnist_paths(dataset_root: Path, split: str) -> list[Path]:
    names = []
    if split in {"train", "all"}:
        names.extend(
            (
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
            )
        )
    if split in {"test", "all"}:
        names.extend(
            (
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            )
        )
    return [dataset_root / name for name in names]


def _load_fashion_mnist_arrays(
    config: InputConfig,
    dataset_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splits = ("train", "test") if config.split == "all" else (config.split,)
    image_parts = []
    label_parts = []
    split_parts = []
    index_parts = []
    offset = 0
    for split in splits:
        prefix = "train" if split == "train" else "t10k"
        images = _read_idx_images(
            dataset_root / f"{prefix}-images-idx3-ubyte.gz"
        )
        labels = _read_idx_labels(
            dataset_root / f"{prefix}-labels-idx1-ubyte.gz"
        )
        if len(images) != len(labels):
            raise ConfigError(
                f"Fashion-MNIST {split} image and label counts do not match."
            )
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


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(16)
        if len(header) != 16:
            raise ConfigError(f"Malformed IDX image header: {path}")
        magic, count, rows, columns = struct.unpack(">IIII", header)
        pixels = handle.read()
    expected = count * rows * columns
    if magic != 2051 or len(pixels) != expected:
        raise ConfigError(f"Malformed IDX image file: {path}")
    return np.frombuffer(pixels, dtype=np.uint8).reshape(count, rows, columns)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(8)
        if len(header) != 8:
            raise ConfigError(f"Malformed IDX label header: {path}")
        magic, count = struct.unpack(">II", header)
        labels = handle.read()
    if magic != 2049 or len(labels) != count:
        raise ConfigError(f"Malformed IDX label file: {path}")
    return np.frombuffer(labels, dtype=np.uint8)


def _ensure_cifar10(
    dataset_root: Path,
    *,
    split: str,
    download: bool,
) -> Path:
    data_dir = dataset_root / CIFAR10_DIRECTORY
    expected = _cifar10_batch_paths(data_dir, split)
    if all(path.is_file() for path in expected):
        return data_dir
    if not download:
        raise ConfigError(
            f"CIFAR-10 does not exist under {dataset_root}. "
            "Set input.download: true to fetch it."
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / "cifar-10-binary.tar.gz"
    if not archive_path.exists() or _file_md5(archive_path) != CIFAR10_MD5:
        partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
        if partial_path.exists() and _file_md5(partial_path) == CIFAR10_MD5:
            partial_path.replace(archive_path)
        else:
            LOGGER.info("Downloading CIFAR-10 from %s", CIFAR10_URL)
            _download_resumable(CIFAR10_URL, partial_path)
            if _file_md5(partial_path) != CIFAR10_MD5:
                partial_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Downloaded CIFAR-10 archive failed its MD5 checksum."
                )
            partial_path.replace(archive_path)
    _extract_cifar10_archive(archive_path, dataset_root)
    if not all(path.is_file() for path in expected):
        raise RuntimeError("CIFAR-10 archive did not contain the expected binary batches.")
    return data_dir


def _cifar10_batch_paths(data_dir: Path, split: str) -> list[Path]:
    paths = []
    if split in {"train", "all"}:
        paths.extend(data_dir / f"data_batch_{index}.bin" for index in range(1, 6))
    if split in {"test", "all"}:
        paths.append(data_dir / "test_batch.bin")
    return paths


def _extract_cifar10_archive(archive_path: Path, dataset_root: Path) -> None:
    allowed = {
        *(f"{CIFAR10_DIRECTORY}/data_batch_{index}.bin" for index in range(1, 6)),
        f"{CIFAR10_DIRECTORY}/test_batch.bin",
        f"{CIFAR10_DIRECTORY}/batches.meta.txt",
    }
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for name in allowed:
            member = members.get(name)
            if member is None or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            destination = dataset_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


def _load_cifar10_arrays(
    config: InputConfig,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    if config.split == "all":
        batches = [
            *((f"data_batch_{index}.bin", "train") for index in range(1, 6)),
            ("test_batch.bin", "test"),
        ]
    elif config.split == "train":
        batches = [
            (f"data_batch_{index}.bin", "train")
            for index in range(1, 6)
        ]
    else:
        batches = [("test_batch.bin", "test")]
    image_parts = []
    label_parts = []
    split_parts = []
    source_files = []
    for filename, split in batches:
        path = data_dir / filename
        if not path.exists():
            raise ConfigError(f"Missing CIFAR-10 batch: {path}")
        records = np.fromfile(path, dtype=np.uint8)
        if records.size % 3073:
            raise ConfigError(f"Malformed CIFAR-10 batch: {path}")
        records = records.reshape(-1, 3073)
        labels = records[:, 0]
        images = records[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        image_parts.append(images)
        label_parts.append(labels)
        split_parts.append(np.asarray([split] * len(records), dtype=object))
        source_files.append(path)
    images = np.concatenate(image_parts)
    return (
        images,
        np.concatenate(label_parts),
        np.concatenate(split_parts),
        np.arange(len(images), dtype=int),
        source_files,
    )


def _ensure_cifar100(
    dataset_root: Path,
    *,
    split: str,
    download: bool,
) -> Path:
    data_dir = dataset_root / CIFAR100_DIRECTORY
    expected = _cifar100_paths(data_dir, split)
    if all(path.is_file() for path in expected):
        return data_dir
    if not download:
        raise ConfigError(
            f"CIFAR-100 does not exist under {dataset_root}. "
            "Set input.download: true to fetch it."
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / "cifar-100-binary.tar.gz"
    if not archive_path.exists() or _file_md5(archive_path) != CIFAR100_MD5:
        partial_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
        if partial_path.exists() and _file_md5(partial_path) == CIFAR100_MD5:
            partial_path.replace(archive_path)
        else:
            LOGGER.info("Downloading CIFAR-100 from %s", CIFAR100_URL)
            _download_resumable(CIFAR100_URL, partial_path)
            if _file_md5(partial_path) != CIFAR100_MD5:
                partial_path.unlink(missing_ok=True)
                raise RuntimeError(
                    "Downloaded CIFAR-100 archive failed its MD5 checksum."
                )
            partial_path.replace(archive_path)
    _extract_cifar100_archive(archive_path, dataset_root)
    if not all(path.is_file() for path in expected):
        raise RuntimeError("CIFAR-100 archive did not contain the expected binary files.")
    return data_dir


def _cifar100_paths(data_dir: Path, split: str) -> list[Path]:
    paths = [data_dir / "fine_label_names.txt"]
    if split in {"train", "all"}:
        paths.append(data_dir / "train.bin")
    if split in {"test", "all"}:
        paths.append(data_dir / "test.bin")
    return paths


def _extract_cifar100_archive(archive_path: Path, dataset_root: Path) -> None:
    allowed = {
        f"{CIFAR100_DIRECTORY}/train.bin",
        f"{CIFAR100_DIRECTORY}/test.bin",
        f"{CIFAR100_DIRECTORY}/fine_label_names.txt",
        f"{CIFAR100_DIRECTORY}/coarse_label_names.txt",
    }
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for name in allowed:
            member = members.get(name)
            if member is None or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            destination = dataset_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)


def _load_cifar100_arrays(
    config: InputConfig,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    if config.split == "all":
        batches = [("train.bin", "train"), ("test.bin", "test")]
    else:
        batches = [(f"{config.split}.bin", config.split)]
    image_parts = []
    label_parts = []
    split_parts = []
    source_files = []
    for filename, split in batches:
        path = data_dir / filename
        if not path.exists():
            raise ConfigError(f"Missing CIFAR-100 batch: {path}")
        records = np.fromfile(path, dtype=np.uint8)
        if records.size % 3074:
            raise ConfigError(f"Malformed CIFAR-100 batch: {path}")
        records = records.reshape(-1, 3074)
        labels = records[:, 1]
        images = records[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        image_parts.append(images)
        label_parts.append(labels)
        split_parts.append(np.asarray([split] * len(records), dtype=object))
        source_files.append(path)
    images = np.concatenate(image_parts)
    return (
        images,
        np.concatenate(label_parts),
        np.concatenate(split_parts),
        np.arange(len(images), dtype=int),
        source_files,
    )


def _load_cifar100_label_names(data_dir: Path) -> list[str]:
    path = data_dir / "fine_label_names.txt"
    if not path.exists():
        raise ConfigError(f"Missing CIFAR-100 label names: {path}")
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != 100:
        raise ConfigError(f"CIFAR-100 label file must contain 100 labels: {path}")
    return labels


def _ensure_cinic10(
    dataset_root: Path,
    *,
    split: str,
    download: bool,
) -> Path:
    split_paths = _cinic10_split_paths(dataset_root, split)
    if all(_cinic10_split_complete(path) for path in split_paths):
        return dataset_root

    archive_path = dataset_root / CINIC10_ARCHIVE
    if archive_path.exists():
        return archive_path

    zip_path = dataset_root / CINIC10_ZIP
    if not zip_path.exists() and download:
        dataset_root.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Downloading CINIC-10 from %s", CINIC10_ZIP_URL)
        _download_resumable(CINIC10_ZIP_URL, zip_path)
    if zip_path.exists():
        _extract_cinic10_archive_from_zip(zip_path, archive_path)
        if archive_path.exists():
            return archive_path

    raise ConfigError(
        f"CINIC-10 does not exist under {dataset_root}. Expected extracted "
        f"split folders or {CINIC10_ARCHIVE}."
    )


def _cinic10_split_paths(dataset_root: Path, split: str) -> list[Path]:
    return [dataset_root / value for value in _cinic10_splits(split)]


def _cinic10_splits(split: str) -> tuple[str, ...]:
    return CINIC10_SPLITS if split == "all" else (split,)


def _cinic10_split_complete(path: Path) -> bool:
    return path.is_dir() and all((path / label).is_dir() for label in CINIC10_LABELS)


def _extract_cinic10_archive_from_zip(zip_path: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        try:
            with archive.open(CINIC10_ARCHIVE) as source, archive_path.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
        except KeyError as exc:
            raise ConfigError(f"{zip_path} does not contain {CINIC10_ARCHIVE}.") from exc


def _cinic10_entries(source: Path, split: str) -> list[tuple[str, str, int, int]]:
    splits = set(_cinic10_splits(split))
    label_ids = {label: index for index, label in enumerate(CINIC10_LABELS)}
    entries: list[tuple[str, str, int, int]] = []
    if source.is_file():
        with tarfile.open(source, mode="r:gz") as archive:
            names = sorted(member.name for member in archive.getmembers() if member.isfile())
        for name in names:
            parsed = _parse_cinic10_path(name, splits, label_ids)
            if parsed is not None:
                entries.append((*parsed, len(entries)))
    else:
        for split_name in _cinic10_splits(split):
            for label, label_id in label_ids.items():
                for path in sorted((source / split_name / label).glob("*.png")):
                    entries.append((str(path), split_name, label_id, len(entries)))
    if not entries:
        raise ConfigError(f"CINIC-10 split {split!r} has no images under {source}.")
    return entries


def _cinic10_sample_indices(
    entries: list[tuple[str, str, int, int]],
    *,
    maximum: int | None,
    seed: int,
    strategy: str,
) -> np.ndarray:
    if maximum is None or maximum >= len(entries):
        return np.arange(len(entries), dtype=int)
    if strategy != "stratified":
        return _sample_indices(len(entries), maximum, seed)
    labels = np.asarray([entry[2] for entry in entries])
    return _stratified_indices(labels, maximum=maximum, seed=seed)


def _stratified_indices(
    labels: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = {
        label: np.flatnonzero(labels == label)
        for label in sorted(set(labels.tolist()))
    }
    quotas = _balanced_quotas(
        {label: len(positions) for label, positions in groups.items()},
        maximum=maximum,
    )
    selected: list[int] = []
    for label, positions in groups.items():
        quota = quotas[label]
        if quota <= 0:
            continue
        shuffled = np.array(positions, copy=True)
        rng.shuffle(shuffled)
        selected.extend(int(value) for value in shuffled[:quota])
    return np.asarray(sorted(selected), dtype=int)


def _balanced_quotas(counts: dict[Any, int], *, maximum: int) -> dict[Any, int]:
    quotas = {label: 0 for label in counts}
    if not counts:
        return quotas
    base = maximum // len(counts)
    remainder = maximum % len(counts)
    for offset, label in enumerate(counts):
        quotas[label] = min(counts[label], base + int(offset < remainder))
    shortfall = maximum - sum(quotas.values())
    while shortfall > 0:
        progressed = False
        for label, count in counts.items():
            if shortfall <= 0:
                break
            available = count - quotas[label]
            if available <= 0:
                continue
            quotas[label] += 1
            shortfall -= 1
            progressed = True
        if not progressed:
            break
    return quotas


def _parse_cinic10_path(
    name: str,
    splits: set[str],
    label_ids: dict[str, int],
) -> tuple[str, str, int] | None:
    parts = Path(name).parts
    if len(parts) != 3 or parts[0] not in splits or parts[1] not in label_ids:
        return None
    if Path(parts[2]).suffix.lower() != ".png":
        return None
    return name, parts[0], label_ids[parts[1]]


def _load_cinic10_images(
    source: Path,
    entries: list[tuple[str, str, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images = np.empty((len(entries), 32, 32, 3), dtype=np.uint8)
    labels = np.empty(len(entries), dtype=np.uint8)
    split_values = np.empty(len(entries), dtype=object)
    original_indices = np.empty(len(entries), dtype=int)
    if source.is_file():
        with tarfile.open(source, mode="r:gz") as archive:
            for position, (name, split, label, original_index) in enumerate(entries):
                member = archive.extractfile(name)
                if member is None:
                    raise ConfigError(f"Missing CINIC-10 image in archive: {name}")
                with member:
                    images[position] = _read_cinic10_png(member)
                labels[position] = label
                split_values[position] = split
                original_indices[position] = original_index
    else:
        for position, (path, split, label, original_index) in enumerate(entries):
            with Path(path).open("rb") as handle:
                images[position] = _read_cinic10_png(handle)
            labels[position] = label
            split_values[position] = split
            original_indices[position] = original_index
    return images, labels, split_values, original_indices


def _read_cinic10_png(handle: Any) -> np.ndarray:
    from PIL import Image

    with Image.open(handle) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.shape != (32, 32, 3):
        raise ConfigError(f"CINIC-10 images must be 32x32 RGB, got {array.shape}.")
    return array


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


def _export_grayscale_sprite_atlases(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    output_dir: str | Path,
    prefix: str,
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
    digest.update(prefix.encode())
    digest.update(np.asarray(labels, dtype=np.uint8).tobytes())
    digest.update(str(len(vectors)).encode())
    stem = f"{prefix}_{digest.hexdigest()[:12]}"
    atlas_paths = [
        directory / f"{stem}_{index:02d}.png"
        for index in range(max(1, int(atlas_indices.max()) + 1 if len(vectors) else 1))
    ]
    if not all(path.exists() for path in atlas_paths):
        for path in directory.glob(f"{prefix}_*.png"):
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


def _export_rgb_sprite_atlases(
    images: np.ndarray,
    *,
    output_dir: str | Path,
    prefix: str,
    tile_size: int = 32,
    atlas_size: int = 2048,
) -> dict[str, object]:
    from PIL import Image

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    columns = atlas_size // tile_size
    capacity = columns * columns
    positions = np.arange(len(images), dtype=int)
    atlas_indices = positions // capacity
    local_positions = positions % capacity
    digest = hashlib.sha256()
    digest.update(b"rgb-v1")
    digest.update(np.asarray(images.shape, dtype=np.int64).tobytes())
    if len(images):
        digest.update(images[0].tobytes())
        digest.update(images[-1].tobytes())
    stem = f"{prefix}_{digest.hexdigest()[:12]}"
    count = max(1, int(atlas_indices.max()) + 1 if len(images) else 1)
    atlas_paths = [directory / f"{stem}_{index:02d}.png" for index in range(count)]
    if not all(path.exists() for path in atlas_paths):
        for path in directory.glob(f"{prefix}_*.png"):
            if path not in atlas_paths:
                path.unlink()
        for atlas_index, path in enumerate(atlas_paths):
            start = atlas_index * capacity
            end = min(len(images), start + capacity)
            atlas = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)
            for local_index, sample_index in enumerate(range(start, end)):
                row, column = divmod(local_index, columns)
                tile = atlas[
                    row * tile_size : (row + 1) * tile_size,
                    column * tile_size : (column + 1) * tile_size,
                ]
                tile[..., :3] = images[sample_index]
                tile[..., 3] = 255
            Image.fromarray(atlas, mode="RGBA").save(path, optimize=True)
    resolved = [str(path.resolve()) for path in atlas_paths]
    return {
        "sprite_atlas_path": [resolved[index] for index in atlas_indices],
        "sprite_atlas_index": atlas_indices,
        "sprite_atlas_column": local_positions % columns,
        "sprite_atlas_row": local_positions // columns,
        "sprite_tile_size": tile_size,
        "sprite_atlas_columns": columns,
        "sprite_atlas_size": atlas_size,
    }


def _export_rgb_thumbnails(
    images: np.ndarray,
    *,
    source_indices: np.ndarray,
    output_dir: str | Path | None,
    prefix: str,
) -> list[str]:
    if output_dir is None:
        return [""] * len(images)
    from PIL import Image

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for image, source_index in zip(images, source_indices, strict=False):
        path = directory / f"{prefix}_{int(source_index):05d}.png"
        if not path.exists():
            Image.fromarray(image, mode="RGB").save(path)
        paths.append(str(path.resolve()))
    return paths


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
        image_paths = _export_array_thumbnails(
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
        image_shape=config.image_shape,
        value_range=config.value_range,
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


def _export_array_thumbnails(
    vectors: np.ndarray,
    *,
    image_shape: tuple[int, ...],
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
            if pixels.ndim == 2:
                image = Image.fromarray(pixels, mode="L")
            elif pixels.ndim == 3 and pixels.shape[-1] == 1:
                image = Image.fromarray(pixels[..., 0], mode="L")
            elif pixels.ndim == 3 and pixels.shape[-1] in {3, 4}:
                mode = "RGB" if pixels.shape[-1] == 3 else "RGBA"
                image = Image.fromarray(pixels, mode=mode)
            else:
                raise ConfigError(f"Unsupported thumbnail image shape: {image_shape}")
            image.save(path)
        paths.append(str(path.resolve()))
    return paths


def _export_grayscale_thumbnails(
    vectors: np.ndarray,
    *,
    image_shape: tuple[int, int],
    value_range: tuple[float, float],
    source_indices: np.ndarray,
    output_dir: str | Path | None,
    prefix: str,
) -> list[str]:
    return _export_array_thumbnails(
        vectors,
        image_shape=image_shape,
        value_range=value_range,
        source_indices=source_indices,
        output_dir=output_dir,
        prefix=prefix,
    )


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


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_resumable(url: str, destination: Path) -> None:
    offset = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={offset}-"} if offset else {},
    )
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with destination.open(mode) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)


def _resolve(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
