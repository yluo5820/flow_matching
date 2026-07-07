from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fm_lab.geometry_explorer.background_dominance import (
    BackgroundDominanceConfig,
    build_background_dominance_experiments,
    render_foreground_background_components,
)
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle


def test_render_foreground_background_components_shapes_and_lambda() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[8:20, 9:19] = 1.0

    low = render_foreground_background_components(
        image,
        family="mnist",
        level="level_04_background",
        base_index=1,
        variant_index=0,
        seed=7,
        constant_gray=0.1,
        background_lambda=0.0,
        background_template_id=3,
    )
    high = render_foreground_background_components(
        image,
        family="mnist",
        level="level_04_background",
        base_index=1,
        variant_index=0,
        seed=7,
        constant_gray=0.1,
        background_lambda=1.0,
        background_template_id=3,
    )

    assert low.full.shape == (28, 28)
    assert low.foreground.shape == (28, 28)
    assert low.background.shape == (28, 28)
    assert low.mask.shape == (28, 28)
    assert not np.allclose(low.full, high.full)
    assert high.metadata["background_template_id"] == 3
    assert high.metadata["background_lambda"] == 1.0


def test_build_background_dominance_writes_datasets_and_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    vectors = np.zeros((6, 28 * 28), dtype=np.float32)
    for index in range(len(vectors)):
        image = vectors[index].reshape(28, 28)
        image[6 + index : 16 + index, 8:20] = 1.0
    labels = np.asarray([0, 1, 2, 0, 1, 2])
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(vectors)),
            "label": [str(value) for value in labels],
            "label_id": labels,
            "split": ["train"] * len(vectors),
            "source_index": np.arange(len(vectors)),
            "original_index": np.arange(len(vectors)),
        }
    )

    def fake_load_dataset(*args, **kwargs) -> DatasetBundle:
        return DatasetBundle(
            metadata=metadata,
            vectors=vectors,
            source_id="fake",
            source_description="fake mnist",
            total_rows=len(vectors),
            image_shape=(28, 28),
            value_range=(0.0, 1.0),
        )

    monkeypatch.setattr(
        "fm_lab.geometry_explorer.background_dominance.load_dataset",
        fake_load_dataset,
    )
    result = build_background_dominance_experiments(
        BackgroundDominanceConfig(
            family="mnist",
            dataset_root="unused",
            output_root=str(tmp_path / "data"),
            base_samples=6,
            variants_per_base=2,
            experiments=("a", "b", "c"),
            lambdas=(0.0, 1.0),
            metrics_max_samples=8,
            metrics_pairs=32,
            overwrite=True,
        ),
        project_root=tmp_path,
    )

    assert result["metrics_path"].is_file()
    metrics = pd.read_csv(result["metrics_path"])
    assert {"a", "b", "c"}.issubset(set(metrics["experiment"]))
    variant_ids = {row["variant_id"] for row in result["datasets"]}
    assert "mnist/background_a_level_04_background_full" in variant_ids
    assert "mnist/background_a_level_04_background_combined" in variant_ids
    assert "mnist/background_c_lambda_0" in variant_ids
    combined = (
        tmp_path
        / "data"
        / "background_dominance_mnist"
        / "experiment_a"
        / "level_04_background"
        / "combined"
    )
    assert (combined / "dataset.yaml").is_file()
    combined_metadata = pd.read_parquet(combined / "metadata.parquet")
    assert set(combined_metadata["label"]) == {"full", "foreground", "background", "mask"}
    assert "class_label" in combined_metadata
