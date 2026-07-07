from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import load_dataset


def test_celeba_loader_reads_attributes_splits_and_resizes(tmp_path: Path) -> None:
    root = tmp_path / "celeba"
    image_dir = root / "img_align_celeba" / "img_align_celeba"
    image_dir.mkdir(parents=True)
    rows = []
    partitions = []
    for index in range(6):
        image_id = f"{index + 1:06d}.jpg"
        _write_portrait(image_dir / image_id, value=index * 20)
        male = 1 if index % 2 == 0 else -1
        smiling = 1 if index >= 3 else -1
        rows.append(f"{image_id},{male},{smiling}")
        partitions.append(f"{image_id},{0 if index < 4 else 2}")
    (root / "list_attr_celeba.csv").write_text(
        "image_id,Male,Smiling\n" + "\n".join(rows),
        encoding="utf-8",
    )
    (root / "list_eval_partition.csv").write_text(
        "image_id,partition\n" + "\n".join(partitions),
        encoding="utf-8",
    )

    bundle = load_dataset(
        InputConfig(
            type="celeba",
            dataset_root=str(root),
            split="train",
            image_size=32,
            label_attribute="Male",
            max_samples=4,
            sample_strategy="stratified",
            thumbnail_mode="none",
        ),
        project_root=tmp_path,
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (4, 32 * 32 * 3)
    assert bundle.image_shape == (32, 32, 3)
    assert bundle.total_rows == 4
    assert set(bundle.metadata["label"]) == {"male", "not_male"}
    assert "attr_Smiling" in bundle.metadata


def _write_portrait(path: Path, *, value: int) -> None:
    image = np.zeros((80, 60, 3), dtype=np.uint8)
    image[..., 0] = value
    image[10:70, 5:55, 1] = 255 - value
    Image.fromarray(image, mode="RGB").save(path)
