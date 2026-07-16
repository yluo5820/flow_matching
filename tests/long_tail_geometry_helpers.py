"""Shared fixtures for long-tail geometry pipeline tests."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any

import numpy as np


def write_balanced_fashion_mnist(
    root: Path,
    *,
    examples_per_class: int,
) -> None:
    """Write a tiny balanced Fashion-MNIST-compatible IDX fixture."""

    labels = np.repeat(np.arange(10, dtype=np.uint8), examples_per_class)
    images = np.repeat(labels[:, None, None], 28 * 28, axis=1).reshape(-1, 28, 28)
    _write_idx_images(root / "train-images-idx3-ubyte.gz", images)
    _write_idx_labels(root / "train-labels-idx1-ubyte.gz", labels)
    _write_idx_images(root / "t10k-images-idx3-ubyte.gz", images)
    _write_idx_labels(root / "t10k-labels-idx1-ubyte.gz", labels)


def geometry_toy_config(root: Path, output_dir: Path) -> dict[str, Any]:
    """Return a minimal ordinary-FM config with counterfactual probes."""

    return {
        "experiment": {
            "name": "geometry_stage0_toy",
            "seed": 0,
            "output_dir": str(output_dir),
        },
        "data": {
            "name": "fashion_mnist_lt",
            "root": str(root),
            "train": True,
            "download": False,
            "normalize": "minus_one_one",
            "dequantize": True,
            "imbalance_type": "exp",
            "imbalance_factor": 0.1,
            "subset_seed": 7,
            "frequency_mapping": {
                "offset": 0,
                "multiplier": 3,
                "diagnostic_pool_per_class": 2,
            },
        },
        "source": {"name": "gaussian", "dim": 784},
        "coupling": {"name": "independent"},
        "path": {"name": "linear"},
        "model": {
            "name": "image_unet",
            "image_shape": [1, 28, 28],
            "base_channels": 8,
            "time_embedding_dim": 16,
            "zero_init_head": False,
            "capacity": {"enabled": False},
        },
        "conditioning": {
            "enabled": True,
            "num_classes": 10,
            "embedding_dim": 16,
            "dropout_probability": 0.0,
        },
        "objective": {
            "name": "flow_matching",
            "model_output": "target",
            "loss_space": "velocity",
            "min_denom": 0.05,
            "modifiers": [],
        },
        "training": {
            "batch_size": 10,
            "steps": 1,
            "lr": 1.0e-4,
            "early_stopping": {"enabled": False},
        },
        "sampling": {"n_samples": 10, "n_trajectories": 2, "nfe": 2},
    }


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, len(images), 28, 28))
        handle.write(images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, len(labels)))
        handle.write(labels.astype(np.uint8).tobytes())
