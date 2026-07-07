"""Dataset adapters that produce feature-ready rows and optional thumbnails."""

from __future__ import annotations

import gzip
import hashlib
import logging
import pickle
import struct
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

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
TINY_IMAGENET_DIRECTORY = "tiny-imagenet-200"
TINY_IMAGENET_ZIP = "tiny-imagenet-200.zip"
TINY_IMAGENET_SPLITS = ("train", "val")
CELEBA_IMAGE_DIRECTORY = "img_align_celeba"
CELEBA_ATTRIBUTE_FILE = "list_attr_celeba.csv"
CELEBA_PARTITION_FILE = "list_eval_partition.csv"
CELEBA_SPLITS = {"train": 0, "valid": 1, "test": 2}
IMAGENET32_TRAIN_ZIP = "Imagenet32_train.zip"
IMAGENET32_VAL_ZIP = "Imagenet32_val.zip"
VOC2012_DIRECTORY = "VOC2012"
VOC2012_LABELS = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
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
    if config.type == "tiny_imagenet":
        return _load_tiny_imagenet(config, root, thumbnail_dir)
    if config.type == "celeba":
        return _load_celeba(config, root, thumbnail_dir)
    if config.type == "imagenet32":
        return _load_imagenet32(config, root, thumbnail_dir)
    if config.type == "voc2012":
        return _load_voc2012(config, root, thumbnail_dir)
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


def _load_tiny_imagenet(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    data_dir = _ensure_tiny_imagenet(dataset_root, split=config.split)
    label_names = _tiny_imagenet_label_names(data_dir)
    entries = _tiny_imagenet_entries(data_dir, config.split, label_names)
    indices = _classification_sample_indices(
        np.asarray([entry[2] for entry in entries], dtype=int),
        maximum=config.max_samples,
        seed=config.sample_seed,
        strategy=config.sample_strategy,
    )
    selected_entries = [entries[int(index)] for index in indices]
    selected_images, selected_labels, split_values, original_indices, wnids = (
        _load_tiny_imagenet_images(selected_entries)
    )
    selected_names = np.asarray([label_names[wnid] for wnid in wnids], dtype=object)
    vectors = selected_images.reshape(len(selected_images), -1)
    image_paths = [""] * len(indices)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_rgb_sprite_atlases(
            selected_images,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix="tiny_imagenet",
        )
    elif config.thumbnail_mode == "files":
        image_paths = _export_rgb_thumbnails(
            selected_images,
            source_indices=indices,
            output_dir=thumbnail_dir,
            prefix="tiny_imagenet",
        )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(indices)),
            "image_path": image_paths,
            "dataset": "tiny_imagenet",
            "split": split_values,
            "label": selected_names,
            "label_id": selected_labels,
            "wnid": wnids,
            "family": selected_names,
            "prompt_id": [f"tiny_imagenet_{value}" for value in wnids],
            "prompt": [f"Tiny ImageNet {value}" for value in selected_names],
            "tags": [
                ["tiny_imagenet", str(wnid), str(name)]
                for wnid, name in zip(wnids, selected_names, strict=False)
            ],
            "source_index": indices,
            "original_index": original_indices,
            "sample_type": "dataset",
            "status": "success",
            **atlas_metadata,
        }
    )
    source_files = [
        data_dir / "wnids.txt",
        data_dir / "words.txt",
        data_dir / "val" / "val_annotations.txt",
    ]
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            source_files,
            extra=f"{config.split}:{indices.tolist()}",
        ),
        source_description=f"Tiny ImageNet {config.split} split at {dataset_root}",
        total_rows=len(entries),
        image_shape=(64, 64, 3),
        value_range=(0.0, 255.0),
    )


def _load_celeba(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    image_dir = _celeba_image_dir(dataset_root)
    attributes = _celeba_attributes(dataset_root)
    partitions = _celeba_partitions(dataset_root, len(attributes))
    split_positions = _celeba_split_positions(partitions, config.split)
    label_attribute = config.label_attribute or "Male"
    if label_attribute not in attributes.columns:
        raise ConfigError(f"CelebA label attribute {label_attribute!r} not found.")
    labels = (attributes[label_attribute].to_numpy(dtype=int) > 0).astype(int)
    indices = _classification_sample_indices(
        labels[split_positions],
        maximum=config.max_samples,
        seed=config.sample_seed,
        strategy=config.sample_strategy,
    )
    selected_positions = split_positions[indices]
    image_size = int(config.image_size or 64)
    image_ids = attributes["image_id"].to_numpy(dtype=object)[selected_positions]
    images = np.stack(
        [
            _read_resized_rgb(
                image_dir / str(image_id),
                image_size=image_size,
                dataset_name="CelebA",
            )
            for image_id in image_ids
        ],
        axis=0,
    )
    selected_labels = labels[selected_positions]
    label_names = np.asarray(
        [
            _celeba_label_name(label_attribute, int(value))
            for value in selected_labels
        ],
        dtype=object,
    )
    split_values = np.asarray(
        [_celeba_split_name(int(value)) for value in partitions[selected_positions]],
        dtype=object,
    )
    vectors = images.reshape(len(images), -1)
    image_paths = [""] * len(images)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_rgb_sprite_atlases(
            images,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix="celeba",
        )
    elif config.thumbnail_mode == "files":
        image_paths = _export_rgb_thumbnails(
            images,
            source_indices=selected_positions,
            output_dir=thumbnail_dir,
            prefix="celeba",
        )
    selected_attributes = attributes.iloc[selected_positions].reset_index(drop=True)
    extra_columns = {
        f"attr_{column}": selected_attributes[column].to_numpy(dtype=int)
        for column in attributes.columns
        if column != "image_id"
    }
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(images)),
            "image_path": image_paths,
            "dataset": "celeba",
            "split": split_values,
            "label": label_names,
            "label_id": selected_labels,
            "label_attribute": label_attribute,
            "image_id": image_ids,
            "family": label_names,
            "prompt_id": [f"celeba_{value}" for value in image_ids],
            "prompt": [f"CelebA {value}" for value in label_names],
            "tags": [
                ["celeba", label_attribute, str(label)]
                for label in label_names
            ],
            "source_index": selected_positions,
            "original_index": selected_positions,
            "sample_type": "dataset",
            "status": "success",
            **extra_columns,
            **atlas_metadata,
        }
    )
    source_files = [
        dataset_root / CELEBA_ATTRIBUTE_FILE,
        dataset_root / CELEBA_PARTITION_FILE,
    ]
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            source_files,
            extra=f"{config.split}:{label_attribute}:{image_size}:{selected_positions.tolist()}",
        ),
        source_description=f"CelebA {config.split} split at {dataset_root}",
        total_rows=len(split_positions),
        image_shape=(image_size, image_size, 3),
        value_range=(0.0, 255.0),
    )


def _load_imagenet32(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    _ensure_imagenet32(dataset_root, split=config.split)
    batches = _imagenet32_batches(dataset_root, config.split)
    batch_labels, offsets = _imagenet32_batch_labels(batches)
    all_labels = np.concatenate(batch_labels)
    labels_zero_based = all_labels.astype(int) - 1
    indices = _classification_sample_indices(
        labels_zero_based,
        maximum=config.max_samples,
        seed=config.sample_seed,
        strategy=config.sample_strategy,
    )
    images, selected_labels, split_values, original_indices = _load_imagenet32_images(
        batches,
        offsets=offsets,
        selected_indices=indices,
    )
    vectors = images.reshape(len(images), -1)
    label_names = np.asarray(
        [f"imagenet_{int(value) + 1:04d}" for value in selected_labels],
        dtype=object,
    )
    image_paths = [""] * len(images)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_rgb_sprite_atlases(
            images,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix="imagenet32",
        )
    elif config.thumbnail_mode == "files":
        image_paths = _export_rgb_thumbnails(
            images,
            source_indices=indices,
            output_dir=thumbnail_dir,
            prefix="imagenet32",
        )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(images)),
            "image_path": image_paths,
            "dataset": "imagenet32",
            "split": split_values,
            "label": label_names,
            "label_id": selected_labels,
            "family": label_names,
            "prompt_id": label_names,
            "prompt": [f"ImageNet32 class {int(value) + 1}" for value in selected_labels],
            "tags": [["imagenet32", str(value)] for value in label_names],
            "source_index": indices,
            "original_index": original_indices,
            "sample_type": "dataset",
            "status": "success",
            **atlas_metadata,
        }
    )
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            batches,
            extra=f"{config.split}:{indices.tolist()}",
        ),
        source_description=f"ImageNet32 {config.split} split at {dataset_root}",
        total_rows=len(all_labels),
        image_shape=(32, 32, 3),
        value_range=(0.0, 255.0),
    )


def _load_voc2012(
    config: InputConfig,
    project_root: Path,
    thumbnail_dir: str | Path | None,
) -> DatasetBundle:
    dataset_root = _resolve(config.dataset_root, project_root)
    voc_root = _voc2012_root(dataset_root)
    entries = _voc2012_entries(voc_root, config.split)
    labels = np.asarray([entry["label_id"] for entry in entries], dtype=int)
    indices = _classification_sample_indices(
        labels,
        maximum=config.max_samples,
        seed=config.sample_seed,
        strategy=config.sample_strategy,
    )
    selected_entries = [entries[int(index)] for index in indices]
    image_size = int(config.image_size or 64)
    images = np.stack(
        [
            _read_resized_rgb(
                entry["image_path"],
                image_size=image_size,
                dataset_name="VOC2012",
            )
            for entry in selected_entries
        ],
        axis=0,
    )
    selected_labels = np.asarray([entry["label_id"] for entry in selected_entries], dtype=int)
    label_names = np.asarray([entry["label"] for entry in selected_entries], dtype=object)
    vectors = images.reshape(len(images), -1)
    image_paths = [""] * len(images)
    atlas_metadata: dict[str, object] = {}
    if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
        atlas_metadata = _export_rgb_sprite_atlases(
            images,
            output_dir=Path(thumbnail_dir).parent / "atlases",
            prefix="voc2012",
        )
    elif config.thumbnail_mode == "files":
        image_paths = _export_rgb_thumbnails(
            images,
            source_indices=indices,
            output_dir=thumbnail_dir,
            prefix="voc2012",
        )
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(images)),
            "image_path": image_paths,
            "dataset": "voc2012",
            "split": [entry["split"] for entry in selected_entries],
            "label": label_names,
            "label_id": selected_labels,
            "sample_id": [entry["sample_id"] for entry in selected_entries],
            "family": label_names,
            "prompt_id": [f"voc2012_{entry['sample_id']}" for entry in selected_entries],
            "prompt": [f"VOC2012 {entry['label']}" for entry in selected_entries],
            "tags": [
                ["voc2012", *entry["object_classes"]]
                for entry in selected_entries
            ],
            "source_index": indices,
            "original_index": [entry["original_index"] for entry in selected_entries],
            "sample_type": "dataset",
            "status": "success",
            "object_classes": [
                ",".join(entry["object_classes"])
                for entry in selected_entries
            ],
            "object_count": [entry["object_count"] for entry in selected_entries],
            "has_segmentation_mask": [
                entry["segmentation_mask_path"] != ""
                for entry in selected_entries
            ],
            "segmentation_mask_path": [
                entry["segmentation_mask_path"]
                for entry in selected_entries
            ],
            "source_width": [entry["width"] for entry in selected_entries],
            "source_height": [entry["height"] for entry in selected_entries],
            **atlas_metadata,
        }
    )
    source_split = "trainval" if config.split == "all" else config.split
    source_files = [
        voc_root / "ImageSets" / "Main" / f"{source_split}.txt",
        *(voc_root / "Annotations" / f"{entry['sample_id']}.xml" for entry in selected_entries),
    ]
    return DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id=_files_source_id(
            source_files,
            extra=f"{config.split}:{image_size}:{indices.tolist()}",
        ),
        source_description=f"VOC2012 {config.split} split at {voc_root}",
        total_rows=len(entries),
        image_shape=(image_size, image_size, 3),
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


def _ensure_tiny_imagenet(dataset_root: Path, *, split: str) -> Path:
    if split not in {*TINY_IMAGENET_SPLITS, "all"}:
        raise ConfigError("Tiny ImageNet supports split train, val, or all.")
    if (dataset_root / "wnids.txt").is_file() and (dataset_root / "train").is_dir():
        return dataset_root
    data_dir = dataset_root / TINY_IMAGENET_DIRECTORY
    if (data_dir / "wnids.txt").is_file() and (data_dir / "train").is_dir():
        return data_dir
    zip_path = dataset_root / TINY_IMAGENET_ZIP
    if zip_path.is_file():
        _extract_zip_safely(zip_path, dataset_root)
        if (data_dir / "wnids.txt").is_file() and (data_dir / "train").is_dir():
            return data_dir
    raise ConfigError(
        f"Tiny ImageNet does not exist under {dataset_root}. Expected "
        f"{TINY_IMAGENET_DIRECTORY}/ or {TINY_IMAGENET_ZIP}."
    )


def _tiny_imagenet_label_names(data_dir: Path) -> dict[str, str]:
    wnids = [
        line.strip()
        for line in (data_dir / "wnids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = {wnid: wnid for wnid in wnids}
    words_path = data_dir / "words.txt"
    if words_path.is_file():
        for line in words_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            wnid, _, words = stripped.partition("\t")
            if wnid in names and words:
                names[wnid] = words.split(",", 1)[0].strip()
    return names


def _tiny_imagenet_entries(
    data_dir: Path,
    split: str,
    label_names: dict[str, str],
) -> list[tuple[Path, str, int, int, str]]:
    splits = TINY_IMAGENET_SPLITS if split == "all" else (split,)
    wnids = list(label_names)
    label_ids = {wnid: index for index, wnid in enumerate(wnids)}
    entries: list[tuple[Path, str, int, int, str]] = []
    for split_name in splits:
        if split_name == "train":
            for wnid in wnids:
                image_dir = data_dir / "train" / wnid / "images"
                for path in sorted(image_dir.glob("*")):
                    if path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                        continue
                    entries.append((path, "train", label_ids[wnid], len(entries), wnid))
        elif split_name == "val":
            annotation_path = data_dir / "val" / "val_annotations.txt"
            if not annotation_path.is_file():
                raise ConfigError(
                    f"Missing Tiny ImageNet validation annotations: {annotation_path}"
                )
            for line in annotation_path.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                filename, wnid = fields[:2]
                if wnid not in label_ids:
                    continue
                path = data_dir / "val" / "images" / filename
                if path.is_file():
                    entries.append((path, "val", label_ids[wnid], len(entries), wnid))
    if not entries:
        raise ConfigError(f"Tiny ImageNet split {split!r} has no labeled images under {data_dir}.")
    return entries


def _load_tiny_imagenet_images(
    entries: list[tuple[Path, str, int, int, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images = np.empty((len(entries), 64, 64, 3), dtype=np.uint8)
    labels = np.empty(len(entries), dtype=np.int64)
    split_values = np.empty(len(entries), dtype=object)
    original_indices = np.empty(len(entries), dtype=int)
    wnids = np.empty(len(entries), dtype=object)
    for position, (path, split, label, original_index, wnid) in enumerate(entries):
        images[position] = _read_square_rgb(path, image_size=64, dataset_name="Tiny ImageNet")
        labels[position] = label
        split_values[position] = split
        original_indices[position] = original_index
        wnids[position] = wnid
    return images, labels, split_values, original_indices, wnids


def _classification_sample_indices(
    labels: np.ndarray,
    *,
    maximum: int | None,
    seed: int,
    strategy: str,
) -> np.ndarray:
    if maximum is None or maximum >= len(labels):
        return np.arange(len(labels), dtype=int)
    if strategy == "stratified":
        return _stratified_indices(labels, maximum=maximum, seed=seed)
    return _sample_indices(len(labels), maximum, seed)


def _read_square_rgb(path: Path, *, image_size: int, dataset_name: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.shape[:2] != (image_size, image_size):
        raise ConfigError(
            f"{dataset_name} images must be {image_size}x{image_size}, "
            f"got {array.shape} at {path}."
        )
    return array


def _celeba_image_dir(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / CELEBA_IMAGE_DIRECTORY / CELEBA_IMAGE_DIRECTORY,
        dataset_root / CELEBA_IMAGE_DIRECTORY,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ConfigError(f"CelebA aligned image directory not found under {dataset_root}.")


def _celeba_attributes(dataset_root: Path) -> pd.DataFrame:
    path = dataset_root / CELEBA_ATTRIBUTE_FILE
    if not path.is_file():
        raise ConfigError(f"CelebA attribute CSV not found: {path}")
    frame = pd.read_csv(path)
    if "image_id" not in frame:
        raise ConfigError(f"CelebA attribute CSV is missing image_id: {path}")
    return frame


def _celeba_partitions(dataset_root: Path, expected_rows: int) -> np.ndarray:
    path = dataset_root / CELEBA_PARTITION_FILE
    if not path.is_file():
        return np.zeros(expected_rows, dtype=int)
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ConfigError(f"CelebA partition CSV must have image_id and partition: {path}")
    values = frame.iloc[:, 1].to_numpy(dtype=int)
    if len(values) != expected_rows:
        raise ConfigError(
            f"CelebA partition row count {len(values)} does not match "
            f"attributes {expected_rows}."
        )
    return values


def _celeba_split_positions(partitions: np.ndarray, split: str) -> np.ndarray:
    if split == "all":
        return np.arange(len(partitions), dtype=int)
    if split not in CELEBA_SPLITS:
        supported = ", ".join(("all", *CELEBA_SPLITS))
        raise ConfigError(f"CelebA split must be one of {supported}.")
    return np.flatnonzero(partitions == CELEBA_SPLITS[split])


def _celeba_split_name(value: int) -> str:
    for name, partition_id in CELEBA_SPLITS.items():
        if value == partition_id:
            return name
    return "unknown"


def _celeba_label_name(attribute: str, value: int) -> str:
    normalized = attribute.lower()
    if value > 0:
        return normalized
    return f"not_{normalized}"


def _read_resized_rgb(path: Path, *, image_size: int, dataset_name: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        rgb = rgb.crop((left, top, left + side, top + side))
        rgb = rgb.resize((image_size, image_size), resample=_pil_resampling("bicubic"))
        array = np.asarray(rgb, dtype=np.uint8)
    if array.shape != (image_size, image_size, 3):
        raise ConfigError(
            f"{dataset_name} resize produced unexpected shape {array.shape} at {path}."
        )
    return array


def _extract_zip_safely(zip_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (output_root / member.filename).resolve()
            if output_root not in target.parents and target != output_root:
                raise ConfigError(f"Unsafe zip member path in {zip_path}: {member.filename}")
        archive.extractall(output_root)


def _pil_resampling(name: str) -> Any:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    return {
        "bicubic": resampling.BICUBIC,
        "box": resampling.BOX,
    }[name]


def _ensure_imagenet32(dataset_root: Path, *, split: str) -> None:
    if split not in {"train", "val", "all"}:
        raise ConfigError("ImageNet32 supports split train, val, or all.")
    required = _imagenet32_expected_files(dataset_root, split)
    if all(path.is_file() for path in required):
        return
    if split in {"train", "all"} and (dataset_root / IMAGENET32_TRAIN_ZIP).is_file():
        _extract_zip_safely(dataset_root / IMAGENET32_TRAIN_ZIP, dataset_root)
    if split in {"val", "all"} and (dataset_root / IMAGENET32_VAL_ZIP).is_file():
        _extract_zip_safely(dataset_root / IMAGENET32_VAL_ZIP, dataset_root)
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise ConfigError(f"ImageNet32 files missing under {dataset_root}: {missing[:3]}")


def _imagenet32_expected_files(dataset_root: Path, split: str) -> list[Path]:
    paths = []
    if split in {"train", "all"}:
        paths.extend(dataset_root / f"train_data_batch_{index}" for index in range(1, 11))
    if split in {"val", "all"}:
        paths.append(dataset_root / "val_data")
    return paths


def _imagenet32_batches(dataset_root: Path, split: str) -> list[Path]:
    return _imagenet32_expected_files(dataset_root, split)


def _imagenet32_batch_labels(batch_paths: list[Path]) -> tuple[list[np.ndarray], np.ndarray]:
    labels = []
    offsets = []
    offset = 0
    for path in batch_paths:
        payload = _read_imagenet32_batch(path)
        batch_labels = np.asarray(payload["labels"], dtype=np.int64)
        labels.append(batch_labels)
        offsets.append(offset)
        offset += len(batch_labels)
    return labels, np.asarray(offsets, dtype=int)


def _load_imagenet32_images(
    batch_paths: list[Path],
    *,
    offsets: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images = np.empty((len(selected_indices), 32, 32, 3), dtype=np.uint8)
    labels = np.empty(len(selected_indices), dtype=np.int64)
    split_values = np.empty(len(selected_indices), dtype=object)
    original_indices = np.asarray(selected_indices, dtype=int)
    output_position = 0
    for batch_index, path in enumerate(batch_paths):
        payload = _read_imagenet32_batch(path)
        data = np.asarray(payload["data"], dtype=np.uint8)
        batch_labels = np.asarray(payload["labels"], dtype=np.int64) - 1
        start = int(offsets[batch_index])
        end = start + len(batch_labels)
        mask = (selected_indices >= start) & (selected_indices < end)
        local_indices = selected_indices[mask] - start
        if len(local_indices) == 0:
            continue
        count = len(local_indices)
        images[output_position : output_position + count] = _imagenet32_pixels(
            data[local_indices]
        )
        labels[output_position : output_position + count] = batch_labels[local_indices]
        split = "val" if path.name == "val_data" else "train"
        split_values[output_position : output_position + count] = split
        output_position += count
    return images, labels, split_values, original_indices


def _read_imagenet32_batch(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict) or "data" not in payload or "labels" not in payload:
        raise ConfigError(f"Malformed ImageNet32 batch: {path}")
    return payload


def _imagenet32_pixels(data: np.ndarray) -> np.ndarray:
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def _voc2012_root(dataset_root: Path) -> Path:
    if (dataset_root / "JPEGImages").is_dir() and (dataset_root / "Annotations").is_dir():
        return dataset_root
    candidate = dataset_root / VOC2012_DIRECTORY
    if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
        return candidate
    raise ConfigError(f"VOC2012 directory not found under {dataset_root}.")


def _voc2012_entries(voc_root: Path, split: str) -> list[dict[str, Any]]:
    if split == "all":
        split = "trainval"
    if split not in {"train", "val", "trainval"}:
        raise ConfigError("VOC2012 supports split train, val, trainval, or all.")
    split_path = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    if not split_path.is_file():
        raise ConfigError(f"VOC2012 split file does not exist: {split_path}")
    sample_ids = [
        line.strip().split()[0]
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entries = []
    for original_index, sample_id in enumerate(sample_ids):
        annotation = _parse_voc_annotation(voc_root / "Annotations" / f"{sample_id}.xml")
        object_classes = annotation["object_classes"]
        label = object_classes[0] if object_classes else "unlabeled"
        label_id = VOC2012_LABELS.index(label) if label in VOC2012_LABELS else -1
        mask_path = voc_root / "SegmentationClass" / f"{sample_id}.png"
        entries.append(
            {
                "sample_id": sample_id,
                "split": split,
                "image_path": voc_root / "JPEGImages" / f"{sample_id}.jpg",
                "label": label,
                "label_id": label_id,
                "object_classes": object_classes,
                "object_count": len(object_classes),
                "segmentation_mask_path": str(mask_path) if mask_path.is_file() else "",
                "width": annotation["width"],
                "height": annotation["height"],
                "original_index": original_index,
            }
        )
    if not entries:
        raise ConfigError(f"VOC2012 split {split!r} has no images.")
    return entries


def _parse_voc_annotation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"VOC2012 annotation does not exist: {path}")
    root = ElementTree.parse(path).getroot()
    size = root.find("size")
    width = _xml_int(size.findtext("width") if size is not None else None)
    height = _xml_int(size.findtext("height") if size is not None else None)
    object_classes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name:
            object_classes.append(name)
    if not object_classes:
        object_classes = ["unlabeled"]
    return {
        "width": width,
        "height": height,
        "object_classes": object_classes,
    }


def _xml_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


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
    tile_size: int | None = None,
    atlas_size: int = 2048,
) -> dict[str, object]:
    from PIL import Image

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if tile_size is None:
        if images.ndim != 4 or images.shape[1] != images.shape[2] or images.shape[-1] != 3:
            raise ConfigError(f"RGB atlas images must be square RGB arrays: {images.shape}")
        tile_size = int(images.shape[1])
    columns = atlas_size // tile_size
    if columns <= 0:
        raise ConfigError(f"Atlas tile size {tile_size} exceeds atlas size {atlas_size}.")
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
    atlas_metadata: dict[str, object] = {}
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
        if config.thumbnail_mode == "atlas" and thumbnail_dir is not None:
            atlas_metadata = _export_array_sprite_atlases(
                selected,
                image_shape=config.image_shape,
                value_range=value_range,
                output_dir=Path(thumbnail_dir).parent / "atlases",
                prefix=data_path.stem,
            )
        elif config.thumbnail_mode == "files":
            image_paths = _export_array_thumbnails(
                selected,
                image_shape=config.image_shape,
                value_range=value_range,
                source_indices=indices,
                output_dir=thumbnail_dir,
                prefix=data_path.stem,
            )
    extra_metadata, metadata_source = _load_numpy_metadata(
        config,
        project_root,
        expected_rows=len(vectors),
        indices=indices,
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
            **atlas_metadata,
        }
    )
    if extra_metadata is not None:
        metadata = _merge_numpy_metadata(metadata, extra_metadata)
    source_files = [data_path]
    if config.labels_path:
        source_files.append(_resolve(config.labels_path, project_root))
    if metadata_source is not None:
        source_files.append(metadata_source)
    return DatasetBundle(
        metadata=metadata,
        vectors=selected,
        source_id=_files_source_id(source_files, extra=str(indices.tolist())),
        source_description=f"NumPy array {data_path}",
        total_rows=len(vectors),
        image_shape=config.image_shape,
        value_range=config.value_range,
    )


def _load_numpy_metadata(
    config: InputConfig,
    project_root: Path,
    *,
    expected_rows: int,
    indices: np.ndarray,
) -> tuple[pd.DataFrame | None, Path | None]:
    if not config.metadata_path:
        return None, None
    metadata_path = _resolve(config.metadata_path, project_root)
    if not metadata_path.exists():
        if config.metadata_path == "metadata/per_image_metadata.jsonl":
            return None, None
        raise ConfigError(f"NumPy metadata input does not exist: {metadata_path}")
    frame = _read_metadata_table(metadata_path)
    if len(frame) == expected_rows:
        return frame.iloc[indices].reset_index(drop=True), metadata_path
    if "source_index" in frame:
        lookup = frame.set_index(frame["source_index"].astype(int), drop=False)
        missing = [int(index) for index in indices if int(index) not in lookup.index]
        if missing:
            raise ConfigError(
                "NumPy metadata source_index is missing selected rows: "
                f"{missing[:5]}"
            )
        return lookup.loc[indices].reset_index(drop=True), metadata_path
    raise ConfigError(
        f"NumPy metadata row count {len(frame)} does not match sample count "
        f"{expected_rows}, and no source_index column was found."
    )


def _read_metadata_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ConfigError(f"Unsupported NumPy metadata format: {path}")


def _merge_numpy_metadata(
    base: pd.DataFrame,
    extra: pd.DataFrame,
) -> pd.DataFrame:
    if len(extra) != len(base):
        raise ConfigError(
            f"Selected NumPy metadata row count {len(extra)} does not match "
            f"selected sample count {len(base)}."
        )
    merged = base.copy()
    for column in extra.columns:
        if column == "row_id":
            continue
        merged[column] = extra[column].to_numpy()
    return merged


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


def _export_array_sprite_atlases(
    vectors: np.ndarray,
    *,
    image_shape: tuple[int, ...],
    value_range: tuple[float, float],
    output_dir: str | Path,
    prefix: str,
    atlas_size: int = 2048,
) -> dict[str, object]:
    from PIL import Image

    shape = tuple(int(value) for value in image_shape)
    if len(shape) == 2:
        tile_size = shape[0]
        if shape[0] != shape[1]:
            raise ConfigError(f"Atlas thumbnails require square images, got {shape}.")
    elif len(shape) == 3 and shape[-1] in {1, 3, 4}:
        tile_size = shape[0]
        if shape[0] != shape[1]:
            raise ConfigError(f"Atlas thumbnails require square images, got {shape}.")
    else:
        raise ConfigError(f"Unsupported atlas image shape: {shape}")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    columns = atlas_size // tile_size
    capacity = columns * columns
    positions = np.arange(len(vectors), dtype=int)
    atlas_indices = positions // capacity
    local_positions = positions % capacity
    digest = hashlib.sha256()
    digest.update(b"array-atlas-v1")
    digest.update(prefix.encode())
    digest.update(str(shape).encode())
    digest.update(str(tuple(float(value) for value in value_range)).encode())
    digest.update(str(len(vectors)).encode())
    if len(vectors):
        digest.update(np.asarray(vectors[0], dtype=np.float32).tobytes())
        digest.update(np.asarray(vectors[-1], dtype=np.float32).tobytes())
    stem = f"{prefix}_{digest.hexdigest()[:12]}"
    count = max(1, int(atlas_indices.max()) + 1 if len(vectors) else 1)
    atlas_paths = [directory / f"{stem}_{index:02d}.png" for index in range(count)]
    if not all(path.exists() for path in atlas_paths):
        for path in directory.glob(f"{prefix}_*.png"):
            if path not in atlas_paths:
                path.unlink()
        pixels = _array_pixels(vectors, image_shape=shape, value_range=value_range)
        for atlas_index, path in enumerate(atlas_paths):
            start = atlas_index * capacity
            end = min(len(vectors), start + capacity)
            atlas = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)
            for local_index, sample_index in enumerate(range(start, end)):
                row, column = divmod(local_index, columns)
                tile = atlas[
                    row * tile_size : (row + 1) * tile_size,
                    column * tile_size : (column + 1) * tile_size,
                ]
                tile_pixels = pixels[sample_index]
                if tile_pixels.ndim == 2:
                    tile[..., :3] = tile_pixels[..., None]
                    tile[..., 3] = 255
                elif tile_pixels.ndim == 3 and tile_pixels.shape[-1] == 1:
                    tile[..., :3] = tile_pixels[..., :1]
                    tile[..., 3] = 255
                elif tile_pixels.ndim == 3 and tile_pixels.shape[-1] == 3:
                    tile[..., :3] = tile_pixels
                    tile[..., 3] = 255
                elif tile_pixels.ndim == 3 and tile_pixels.shape[-1] == 4:
                    tile[...] = tile_pixels
                else:
                    raise ConfigError(f"Unsupported atlas image shape: {shape}")
            Image.fromarray(atlas, mode="RGBA").save(path, optimize=True)
    resolved_paths = [str(path.resolve()) for path in atlas_paths]
    return {
        "sprite_atlas_path": [resolved_paths[index] for index in atlas_indices],
        "sprite_atlas_index": atlas_indices,
        "sprite_atlas_column": local_positions % columns,
        "sprite_atlas_row": local_positions // columns,
        "sprite_tile_size": tile_size,
        "sprite_atlas_columns": columns,
        "sprite_atlas_size": atlas_size,
    }


def _array_pixels(
    vectors: np.ndarray,
    *,
    image_shape: tuple[int, ...],
    value_range: tuple[float, float],
) -> np.ndarray:
    low, high = value_range
    scale = max(high - low, np.finfo(np.float32).eps)
    normalized = np.clip((vectors.reshape((-1, *image_shape)) - low) / scale, 0.0, 1.0)
    return np.asarray(np.round(normalized * 255.0), dtype=np.uint8)


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
