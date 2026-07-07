from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fm_lab.geometry_explorer.segmentation_ablation import (
    SegmentationAblationConfig,
    build_segmentation_ablation,
)


def test_build_oxford_pet_segmentation_ablation(tmp_path: Path) -> None:
    root = tmp_path / "oxford-iiit-pet"
    images_dir = root / "images"
    trimaps_dir = root / "annotations" / "trimaps"
    images_dir.mkdir(parents=True)
    trimaps_dir.mkdir(parents=True)
    rows = [
        ("Abyssinian_1", 1, 1, 1),
        ("Abyssinian_2", 1, 1, 1),
        ("beagle_1", 2, 2, 1),
        ("beagle_2", 2, 2, 1),
    ]
    for index, (sample_id, *_rest) in enumerate(rows):
        image = np.zeros((24, 30, 3), dtype=np.uint8)
        image[..., 0] = index * 40
        image[5:20, 6:24, 1] = 255
        Image.fromarray(image, mode="RGB").save(images_dir / f"{sample_id}.jpg")
        trimap = np.full((24, 30), 2, dtype=np.uint8)
        trimap[5:20, 6:24] = 1
        trimap[4, 6:24] = 3
        Image.fromarray(trimap, mode="L").save(trimaps_dir / f"{sample_id}.png")
    _write_oxford_list(root / "annotations" / "list.txt", rows)
    _write_oxford_list(root / "annotations" / "trainval.txt", rows[:2])
    _write_oxford_list(root / "annotations" / "test.txt", rows[2:])

    result = build_segmentation_ablation(
        SegmentationAblationConfig(
            dataset="oxford_iiit_pet",
            dataset_root=str(root),
            output_root=str(tmp_path / "prepared"),
            split="all",
            image_size=16,
            overwrite=True,
        ),
        project_root=tmp_path,
    )

    assert result["mask_only"] is False
    variant_ids = {row["variant_id"] for row in result["datasets"]}
    assert "oxford_iiit_pet/segmentation_all_fg" in variant_ids
    assert "oxford_iiit_pet/segmentation_all_combined" in variant_ids
    foreground = tmp_path / "prepared" / "oxford_iiit_pet" / "all" / "fg"
    assert (foreground / "dataset.yaml").is_file()
    metadata = pd.read_parquet(foreground / "metadata.parquet")
    assert set(metadata["species"]) == {"cat", "dog"}
    assert "mask_area_ratio" in metadata
    values = np.load(foreground / "images.npy")
    assert values.shape == (4, 16 * 16 * 3)


def test_build_cub_segmentation_mask_only(tmp_path: Path) -> None:
    root = tmp_path / "segmentations"
    class_dir = root / "001.Black_footed_Albatross"
    class_dir.mkdir(parents=True)
    for index in range(3):
        mask = np.zeros((20, 18), dtype=np.uint8)
        mask[4:15, 5:14] = 255
        Image.fromarray(mask, mode="L").save(
            class_dir / f"Black_Footed_Albatross_000{index}_1.png"
        )

    result = build_segmentation_ablation(
        SegmentationAblationConfig(
            dataset="cub200_segmentations",
            dataset_root=str(root),
            output_root=str(tmp_path / "prepared"),
            split="all",
            image_size=16,
            overwrite=True,
        ),
        project_root=tmp_path,
    )

    assert result["mask_only"] is True
    assert result["missing_images"] == 3
    assert [row["variant_id"] for row in result["datasets"]] == [
        "cub200_segmentations/segmentation_all_mask"
    ]
    mask_dir = tmp_path / "prepared" / "cub200_segmentations" / "all" / "mask"
    values = np.load(mask_dir / "images.npy")
    assert values.shape == (3, 16 * 16 * 3)


def _write_oxford_list(path: Path, rows: list[tuple[str, int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id, class_id, species, breed_id in rows:
            handle.write(f"{sample_id} {class_id} {species} {breed_id}\n")
