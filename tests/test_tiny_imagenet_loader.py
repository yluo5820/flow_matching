from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import load_dataset


def test_tiny_imagenet_loader_reads_train_val_and_atlas(tmp_path: Path) -> None:
    root = tmp_path / "tiny_imagenet"
    data_dir = root / "tiny-imagenet-200"
    wnids = ["n00000001", "n00000002"]
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "val" / "images").mkdir(parents=True)
    (data_dir / "wnids.txt").write_text("\n".join(wnids), encoding="utf-8")
    (data_dir / "words.txt").write_text(
        "n00000001\tclass one\nn00000002\tclass two\n",
        encoding="utf-8",
    )
    val_rows = []
    for label_id, wnid in enumerate(wnids):
        train_image_dir = data_dir / "train" / wnid / "images"
        train_image_dir.mkdir(parents=True)
        for index in range(3):
            _write_image(train_image_dir / f"{wnid}_{index}.JPEG", value=label_id * 80 + index)
        val_name = f"val_{label_id}.JPEG"
        _write_image(data_dir / "val" / "images" / val_name, value=120 + label_id)
        val_rows.append(f"{val_name}\t{wnid}\t0\t0\t64\t64")
    (data_dir / "val" / "val_annotations.txt").write_text(
        "\n".join(val_rows),
        encoding="utf-8",
    )

    bundle = load_dataset(
        InputConfig(
            type="tiny_imagenet",
            dataset_root=str(root),
            split="all",
            max_samples=4,
            sample_seed=0,
            sample_strategy="stratified",
            thumbnail_mode="atlas",
        ),
        project_root=tmp_path,
        thumbnail_dir=tmp_path / "thumbs",
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (4, 64 * 64 * 3)
    assert bundle.image_shape == (64, 64, 3)
    assert bundle.total_rows == 8
    assert set(bundle.metadata["label"]) == {"class one", "class two"}
    assert bundle.metadata["sprite_tile_size"].nunique() == 1
    assert int(bundle.metadata["sprite_tile_size"].iloc[0]) == 64


def _write_image(path: Path, *, value: int) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = value
    image[..., 1] = 255 - value
    Image.fromarray(image, mode="RGB").save(path)
