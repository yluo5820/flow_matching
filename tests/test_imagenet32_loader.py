from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import load_dataset


def test_imagenet32_loader_reads_batches_and_stratifies(tmp_path: Path) -> None:
    root = tmp_path / "imagenet32"
    root.mkdir()
    for batch_index in range(1, 11):
        _write_batch(
            root / f"train_data_batch_{batch_index}",
            labels=[1, 2, 1, 2],
            offset=batch_index * 10,
        )

    bundle = load_dataset(
        InputConfig(
            type="imagenet32",
            dataset_root=str(root),
            split="train",
            max_samples=4,
            sample_strategy="stratified",
            thumbnail_mode="none",
        ),
        project_root=tmp_path,
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (4, 32 * 32 * 3)
    assert bundle.image_shape == (32, 32, 3)
    assert bundle.total_rows == 40
    assert set(bundle.metadata["label"]) == {"imagenet_0001", "imagenet_0002"}
    assert set(bundle.metadata["label_id"]) == {0, 1}


def _write_batch(path: Path, *, labels: list[int], offset: int) -> None:
    data = np.zeros((len(labels), 3, 32, 32), dtype=np.uint8)
    for index in range(len(labels)):
        data[index, 0] = offset + index
        data[index, 1] = 255 - index
    payload = {
        "labels": labels,
        "data": data.reshape(len(labels), -1),
        "mean": np.zeros(3 * 32 * 32, dtype=np.float32),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
