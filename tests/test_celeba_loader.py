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


def test_celeba_loader_supports_landmark_bbox_crop(tmp_path: Path) -> None:
    root = tmp_path / "celeba"
    image_dir = root / "img_align_celeba" / "img_align_celeba"
    image_dir.mkdir(parents=True)
    rows = []
    partitions = []
    landmarks = []
    bboxes = []
    for index in range(4):
        image_id = f"{index + 1:06d}.jpg"
        _write_portrait(image_dir / image_id, value=index * 30)
        male = 1 if index % 2 == 0 else -1
        rows.append(f"{image_id},{male},1")
        partitions.append(f"{image_id},0")
        landmarks.append(f"{image_id},69,109,106,113,77,142,73,152,108,154")
        bboxes.append(f"{image_id},95,71,226,313")
    (root / "list_attr_celeba.csv").write_text(
        "image_id,Male,Smiling\n" + "\n".join(rows),
        encoding="utf-8",
    )
    (root / "list_eval_partition.csv").write_text(
        "image_id,partition\n" + "\n".join(partitions),
        encoding="utf-8",
    )
    (root / "list_landmarks_align_celeba.csv").write_text(
        (
            "image_id,lefteye_x,lefteye_y,righteye_x,righteye_y,nose_x,nose_y,"
            "leftmouth_x,leftmouth_y,rightmouth_x,rightmouth_y\n"
        )
        + "\n".join(landmarks),
        encoding="utf-8",
    )
    (root / "list_bbox_celeba.csv").write_text(
        "image_id,x_1,y_1,width,height\n" + "\n".join(bboxes),
        encoding="utf-8",
    )

    bundle = load_dataset(
        InputConfig(
            type="celeba",
            dataset_root=str(root),
            split="train",
            image_size=32,
            label_attribute="Male",
            crop_mode="landmark_bbox",
            max_samples=4,
            sample_strategy="stratified",
            thumbnail_mode="none",
        ),
        project_root=tmp_path,
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (4, 32 * 32 * 3)
    assert set(bundle.metadata["crop_source"]) == {"landmarks"}
    assert bundle.metadata["crop_x0"].min() >= 0
    assert bundle.metadata["crop_y0"].min() >= 0
    assert not bundle.metadata["annotation_bbox_valid_for_aligned"].any()


def _write_portrait(path: Path, *, value: int) -> None:
    image = np.zeros((80, 60, 3), dtype=np.uint8)
    image[..., 0] = value
    image[10:70, 5:55, 1] = 255 - value
    Image.fromarray(image, mode="RGB").save(path)
