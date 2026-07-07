from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fm_lab.geometry_explorer.photometric import (
    PhotometricBuildConfig,
    _level_variants_per_base,
    _resolve_level,
    render_photometric_variant,
)
from fm_lab.image_diagnostics.canvas_explorer import _uses_label_tinted_thumbnail
from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import load_dataset


def test_render_photometric_variant_is_deterministic() -> None:
    image = np.linspace(0.0, 1.0, 28 * 28, dtype=np.float32)

    first, first_stats = render_photometric_variant(
        image,
        family="mnist",
        level="level_06_full",
        base_index=12,
        variant_index=2,
        seed=123,
    )
    second, second_stats = render_photometric_variant(
        image,
        family="mnist",
        level="level_06_full",
        base_index=12,
        variant_index=2,
        seed=123,
    )
    different, _ = render_photometric_variant(
        image,
        family="mnist",
        level="level_06_full",
        base_index=12,
        variant_index=3,
        seed=123,
    )

    assert first.shape == (28, 28)
    assert np.allclose(first, second)
    assert first_stats == second_stats
    assert not np.allclose(first, different)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    assert "mean_luminance" in first_stats
    assert "background_strength" in first_stats


def test_numpy_loader_merges_optional_metadata(tmp_path: Path) -> None:
    data_path = tmp_path / "images.npy"
    labels_path = tmp_path / "labels.npy"
    metadata_path = tmp_path / "metadata.parquet"
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    labels = np.asarray(["zero", "one", "two"], dtype="<U8")
    metadata = pd.DataFrame(
        {
            "row_id": [10, 11, 12],
            "dataset": ["photometric_mnist"] * 3,
            "label": ["zero", "one", "two"],
            "label_id": [0, 1, 2],
            "photometric_level": ["level_01_global"] * 3,
            "mean_luminance": [0.1, 0.2, 0.3],
        }
    )
    np.save(data_path, data)
    np.save(labels_path, labels)
    metadata.to_parquet(metadata_path, index=False)

    bundle = load_dataset(
        InputConfig(
            type="numpy",
            data_path=str(data_path),
            labels_path=str(labels_path),
            metadata_path=str(metadata_path),
            image_shape=(2, 2),
            value_range=(0.0, 11.0),
            thumbnail_mode="atlas",
        ),
        project_root=tmp_path,
        thumbnail_dir=tmp_path / "thumbnails",
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (3, 4)
    assert bundle.metadata["row_id"].tolist() == [0, 1, 2]
    assert bundle.metadata["dataset"].tolist() == ["photometric_mnist"] * 3
    assert bundle.metadata["label_id"].tolist() == [0, 1, 2]
    assert bundle.metadata["photometric_level"].tolist() == ["level_01_global"] * 3
    assert bundle.metadata["mean_luminance"].tolist() == [0.1, 0.2, 0.3]
    assert "sprite_atlas_path" in bundle.metadata
    assert Path(bundle.metadata["sprite_atlas_path"].iloc[0]).is_file()


def test_photometric_datasets_use_mnist_style_label_tinting() -> None:
    assert _uses_label_tinted_thumbnail("mnist")
    assert _uses_label_tinted_thumbnail("fashion_mnist")
    assert _uses_label_tinted_thumbnail("photometric_mnist")
    assert _uses_label_tinted_thumbnail("photometric_fashion_mnist")
    assert not _uses_label_tinted_thumbnail("cifar10")
    assert not _uses_label_tinted_thumbnail("cifar100")


def test_clean_level_defaults_to_single_variant_per_base() -> None:
    config = PhotometricBuildConfig(
        family="mnist",
        dataset_root="data/mnist",
        variants_per_base=5,
        clean_variants_per_base=1,
    )

    assert _level_variants_per_base(config, _resolve_level("level_00_clean")) == 1
    assert _level_variants_per_base(config, _resolve_level("level_06_full")) == 5
