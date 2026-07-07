from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fm_lab.geometry_explorer.cifar_background import (
    CIFAR_BACKGROUND_COMPONENTS,
    CifarBackgroundConfig,
    MaskResult,
    analyze_mask,
    build_cifar_background_ablation,
)
from fm_lab.image_diagnostics.dataset_loader import CIFAR10_LABELS, DatasetBundle


def test_analyze_mask_accepts_blob_and_rejects_empty() -> None:
    config = CifarBackgroundConfig()
    blob = np.zeros((32, 32), dtype=np.float32)
    blob[8:24, 9:23] = 1.0

    accepted = analyze_mask(blob, config=config)
    rejected = analyze_mask(np.zeros((32, 32), dtype=np.float32), config=config)

    assert accepted["valid_mask"] is True
    assert accepted["num_components"] == 1
    assert accepted["bbox_y0"] == 8
    assert accepted["bbox_y1"] == 24
    assert rejected["valid_mask"] is False
    assert "empty_mask" in rejected["failure_reason"]


def test_build_cifar_background_ablation_writes_variants_and_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    images = np.zeros((12, 32, 32, 3), dtype=np.uint8)
    for index in range(len(images)):
        images[index, :, :, 0] = index * 10
        images[index, 8:24, 8:24, 1] = 255
    labels = np.asarray([index % 3 for index in range(len(images))], dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(images)),
            "image_path": [""] * len(images),
            "dataset": "cifar10",
            "split": ["train"] * len(images),
            "label": [CIFAR10_LABELS[int(value)] for value in labels],
            "label_id": labels,
            "family": [CIFAR10_LABELS[int(value)] for value in labels],
            "prompt_id": [f"cifar10_{CIFAR10_LABELS[int(value)]}" for value in labels],
            "prompt": [f"CIFAR-10 {CIFAR10_LABELS[int(value)]}" for value in labels],
            "source_index": np.arange(len(images)),
            "original_index": np.arange(len(images)),
            "sample_type": "dataset",
            "status": "success",
        }
    )

    def fake_load_dataset(*args, **kwargs) -> DatasetBundle:
        return DatasetBundle(
            metadata=metadata,
            vectors=images.reshape(len(images), -1),
            source_id="fake",
            source_description="fake cifar10",
            total_rows=len(images),
            image_shape=(32, 32, 3),
            value_range=(0.0, 255.0),
        )

    def fake_segmenter(image: np.ndarray, row: pd.Series, index: int) -> MaskResult:
        del image, row, index
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[8:24, 8:24] = 1.0
        return MaskResult(mask=mask, confidence=0.9)

    monkeypatch.setattr(
        "fm_lab.geometry_explorer.cifar_background.load_dataset",
        fake_load_dataset,
    )
    result = build_cifar_background_ablation(
        CifarBackgroundConfig(
            output_root=str(tmp_path / "data" / "cifar_background"),
            split="train",
            max_samples=9,
            sample_strategy="stratified",
            pipelines=("fake",),
            metrics_max_samples=9,
            metrics_pairs=32,
            overwrite=True,
        ),
        project_root=tmp_path,
        segmenters={"fake": fake_segmenter},
    )

    assert result["metrics_path"].is_file()
    metrics = pd.read_csv(result["metrics_path"])
    assert set(metrics["metric_type"]) == {"component"}
    variant_ids = {row["variant_id"] for row in result["datasets"]}
    for component in (*CIFAR_BACKGROUND_COMPONENTS, "combined"):
        assert f"cifar10/background_fake_train_{component}" in variant_ids

    combined = tmp_path / "data" / "cifar_background" / "fake" / "train" / "combined"
    assert (combined / "dataset.yaml").is_file()
    combined_metadata = pd.read_parquet(combined / "metadata.parquet")
    assert set(combined_metadata["label"]) == set(CIFAR_BACKGROUND_COMPONENTS)
    assert "class_label" in combined_metadata
    assert "mask_area_ratio" in combined_metadata

    foreground = np.load(
        tmp_path / "data" / "cifar_background" / "fake" / "train" / "fg" / "images.npy"
    )
    raw = np.load(
        tmp_path / "data" / "cifar_background" / "fake" / "train" / "raw" / "images.npy"
    )
    assert foreground.shape == raw.shape == (9, 32 * 32 * 3)
    assert not np.allclose(foreground, raw)
