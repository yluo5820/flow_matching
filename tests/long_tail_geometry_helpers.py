"""Shared fixtures for long-tail geometry pipeline tests."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fm_lab.diagnostics.long_tail_geometry.manifest import build_probe_manifest
from fm_lab.experiments.factory import (
    build_model,
    build_path,
    build_source,
    build_target,
)
from fm_lab.training.losses import build_objective
from fm_lab.utils.checkpoints import save_checkpoint


def write_balanced_fashion_mnist(
    root: Path,
    *,
    examples_per_class: int,
) -> None:
    """Write a tiny balanced Fashion-MNIST-compatible IDX fixture."""

    root.mkdir(parents=True, exist_ok=True)
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


def build_probe_fixture(tmp_path: Path):
    """Build a deterministic ordinary-FM checkpoint-replay fixture."""

    root = tmp_path / "data"
    write_balanced_fashion_mnist(root, examples_per_class=10)
    config = geometry_toy_config(root, tmp_path / "run")
    target = build_target(config)
    source = build_source(config)
    path = build_path(config)
    torch.manual_seed(0)
    model = build_model(config, dim=source.dim)
    objective = build_objective(
        config["objective"],
        class_counts=target.class_counts,
    )
    _, labels, sample_ids = target.diagnostic_samples("a")
    manifest = build_probe_manifest(
        sample_ids.astype(np.int64),
        labels.numpy(),
        split="a",
        rows_per_class_per_stratum=1,
        batch_size=1,
        time_strata=((0.02, 0.10),),
        seed=19,
    )
    return config, target, source, path, objective, model, manifest


def write_geometry_toy_checkpoint(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    """Write a tiny ordinary-FM checkpoint and its Stage-0 validation config."""

    root = tmp_path / "data"
    write_balanced_fashion_mnist(root, examples_per_class=10)
    config = geometry_toy_config(root, tmp_path / "run")
    config["diagnostics"] = {
        "long_tail_geometry": {
            "pairing_check_offsets": [0, 1, 7],
            "probe_splits": ["a", "b"],
            "rows_per_class_per_stratum": 1,
            "microbatch_size": 1,
            "time_strata": [[0.02, 0.10]],
            "layers": [
                "input_block.conv2.weight",
                "output_block.2.weight",
            ],
            "sketch_dim": 4096,
            "max_sketch_dim": 4096,
            "sketch_seed": 20260716,
            "max_cosine_error": 0.02,
            "max_subspace_error": 0.03,
            "permutation_count": 99,
        }
    }
    torch.manual_seed(0)
    source = build_source(config)
    model = build_model(config, dim=source.dim)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=None,
        step=1,
        config=config,
        metrics={},
        prediction_contract={
            "path": "linear",
            "objective": "flow_matching",
            "model_output": "target",
            "loss_space": "velocity",
        },
    )
    return config, checkpoint_path


def _write_idx_images(path: Path, images: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, len(images), 28, 28))
        handle.write(images.astype(np.uint8).tobytes())


def _write_idx_labels(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, len(labels)))
        handle.write(labels.astype(np.uint8).tobytes())
