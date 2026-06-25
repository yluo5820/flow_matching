from __future__ import annotations

import gzip
import io
import json
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import fm_lab.image_diagnostics.dataset_loader as dataset_loader
from fm_lab.image_diagnostics.canvas_explorer import (
    AtlasBundle,
    _compact_atlas_bundle,
    _compact_atlas_path,
    atlas_data_url,
    build_canvas_html,
    prepare_sprite_atlases,
)
from fm_lab.image_diagnostics.config import (
    FeatureConfig,
    InputConfig,
    LocalDiagnosticsConfig,
    ProjectionConfig,
    apply_diagnostics_overrides,
    diagnostics_config_from_dict,
)
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle, load_dataset
from fm_lab.image_diagnostics.explorer_data import build_explorer_data
from fm_lab.image_diagnostics.explorer_merge import (
    combine_explorer_tables,
    discover_explorer_groups,
    load_discovered_explorer_group,
)
from fm_lab.image_diagnostics.feature_runner import compute_or_load_features
from fm_lab.image_diagnostics.id_config import id_config_from_dict
from fm_lab.image_diagnostics.label_store import load_manual_labels, save_manual_label
from fm_lab.image_diagnostics.local_diagnostics import compute_local_diagnostics
from fm_lab.image_diagnostics.metadata_loader import load_image_metadata
from fm_lab.image_diagnostics.mnist_reference_projections import (
    compute_mnist_reference_projections,
)
from fm_lab.image_diagnostics.projection_diagnostics import (
    compute_projection_diagnostics,
)
from fm_lab.image_diagnostics.projections import (
    compute_or_load_projections,
    compute_projection,
    projection_variants,
)
from fm_lab.image_diagnostics.runner import run_diagnostics_build
from fm_lab.image_diagnostics.three_explorer import build_three_html


def test_config_defaults_to_raw_features_without_model_download() -> None:
    raw = _raw_config("data/mnist")
    updated = apply_diagnostics_overrides(
        raw,
        input_path="other/mnist",
        feature_mode="raw",
        recompute_features=True,
        recompute_projection=True,
        recompute_diagnostics=True,
        no_explorer=True,
    )
    config = diagnostics_config_from_dict(updated)

    assert config.explorer_name == "test_explorer"
    assert config.input.dataset_root == "other/mnist"
    assert config.features.mode == "raw"
    assert config.features.skip_existing is False
    assert config.projection.skip_existing is False
    assert config.diagnostics.skip_existing is False
    assert config.explorer.enabled is False
    assert raw["input"]["dataset_root"] == "data/mnist"


def test_input_config_does_not_sample_by_default() -> None:
    assert InputConfig().max_samples is None


def test_mnist_loader_selects_vectors_labels_and_thumbnails(tmp_path: Path) -> None:
    mnist_root = tmp_path / "mnist"
    images = np.zeros((6, 28, 28), dtype=np.uint8)
    for index in range(6):
        images[index, 3 + index : 8 + index, 4:12] = 255
    labels = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    _write_mnist_split(mnist_root, images, labels, split="test")

    bundle = load_dataset(
        InputConfig(
            type="mnist",
            dataset_root=str(mnist_root),
            split="test",
            max_samples=4,
            sample_seed=7,
        ),
        project_root=tmp_path,
        thumbnail_dir=tmp_path / "thumbnails",
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (4, 784)
    assert bundle.metadata["label"].map(type).eq(str).all()
    assert bundle.metadata["image_path"].map(lambda value: Path(value).is_file()).all()
    assert bundle.total_rows == 6


def test_mnist_loader_recreates_fetch_mldata_order(tmp_path: Path) -> None:
    mnist_root = tmp_path / "mnist"
    train_images = np.arange(4 * 28 * 28, dtype=np.uint8).reshape(4, 28, 28)
    train_labels = np.asarray([2, 0, 1, 0], dtype=np.uint8)
    test_images = np.arange(3 * 28 * 28, dtype=np.uint8).reshape(3, 28, 28)
    test_labels = np.asarray([1, 0, 1], dtype=np.uint8)
    _write_mnist_split(
        mnist_root,
        train_images,
        train_labels,
        split="train",
    )
    _write_mnist_split(
        mnist_root,
        test_images,
        test_labels,
        split="test",
    )

    bundle = load_dataset(
        InputConfig(
            type="mnist",
            dataset_root=str(mnist_root),
            split="all",
            order="mldata",
            thumbnail_mode="atlas",
            max_samples=None,
        ),
        project_root=tmp_path,
        thumbnail_dir=tmp_path / "assets" / "thumbnails",
    )

    assert bundle.metadata["label"].tolist() == ["0", "0", "1", "2", "0", "1", "1"]
    assert bundle.metadata["split"].tolist() == [
        "train",
        "train",
        "train",
        "train",
        "test",
        "test",
        "test",
    ]
    assert bundle.metadata["original_index"].tolist() == [1, 3, 2, 0, 5, 4, 6]
    assert bundle.metadata["source_index"].tolist() == list(range(7))
    atlas_path = Path(bundle.metadata["sprite_atlas_path"].iloc[0])
    with Image.open(atlas_path) as atlas:
        assert atlas.size == (2048, 2048)


def test_fashion_mnist_loader_uses_named_classes_and_grayscale_atlas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_root = tmp_path / "fashion_mnist"
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    images[0, 4:12, 5:20] = 255
    images[1, 6:24, 9:18] = 180
    images[2, 12:21, 3:25] = 120
    images[3, 3:25, 8:21] = 220
    labels = np.asarray([0, 1, 5, 9], dtype=np.uint8)
    _write_mnist_split(dataset_root, images, labels, split="test")
    for path in dataset_root.glob("*.gz"):
        monkeypatch.setitem(
            dataset_loader.FASHION_MNIST_FILES,
            path.name,
            dataset_loader._file_md5(path),
        )

    bundle = load_dataset(
        InputConfig(
            type="fashion_mnist",
            dataset_root=str(dataset_root),
            split="test",
            thumbnail_mode="atlas",
            max_samples=None,
        ),
        project_root=tmp_path,
        thumbnail_dir=tmp_path / "assets" / "thumbnails",
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (4, 784)
    assert bundle.image_shape == (28, 28)
    assert bundle.value_range == (0.0, 1.0)
    assert bundle.metadata["dataset"].unique().tolist() == ["fashion_mnist"]
    assert bundle.metadata["label"].tolist() == [
        "T-shirt/top",
        "Trouser",
        "Sandal",
        "Ankle boot",
    ]
    assert bundle.metadata["label_id"].tolist() == [0, 1, 5, 9]
    atlas_path = Path(bundle.metadata["sprite_atlas_path"].iloc[0])
    assert atlas_path.name.startswith("fashion_mnist_")
    with Image.open(atlas_path) as atlas:
        assert atlas.size == (2048, 2048)


def test_cifar10_loader_preserves_rgb_images_and_class_names(tmp_path: Path) -> None:
    cifar_root = tmp_path / "cifar10"
    data_dir = cifar_root / "cifar-10-batches-bin"
    data_dir.mkdir(parents=True)
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    images[0, :, :, 0] = 255
    images[1, :, :, 1] = 128
    images[2, :, :, 2] = 64
    images[3] = np.asarray([12, 34, 56], dtype=np.uint8)
    labels = np.asarray([0, 3, 8, 9], dtype=np.uint8)
    _write_cifar10_batch(data_dir / "test_batch.bin", images, labels)

    bundle = load_dataset(
        InputConfig(
            type="cifar10",
            dataset_root=str(cifar_root),
            split="test",
            thumbnail_mode="atlas",
            max_samples=None,
        ),
        project_root=tmp_path,
        thumbnail_dir=tmp_path / "assets" / "thumbnails",
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (4, 32 * 32 * 3)
    assert bundle.image_shape == (32, 32, 3)
    assert bundle.value_range == (0.0, 255.0)
    assert bundle.metadata["label"].tolist() == [
        "airplane",
        "cat",
        "ship",
        "truck",
    ]
    atlas_path = Path(bundle.metadata["sprite_atlas_path"].iloc[0])
    with Image.open(atlas_path) as atlas:
        assert atlas.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_cifar10_loader_supports_grayscale_vectors_and_atlas(
    tmp_path: Path,
) -> None:
    cifar_root = tmp_path / "cifar10"
    data_dir = cifar_root / "cifar-10-batches-bin"
    data_dir.mkdir(parents=True)
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    images[0] = np.asarray([255, 0, 0], dtype=np.uint8)
    images[1] = np.asarray([0, 128, 0], dtype=np.uint8)
    images[2] = np.asarray([0, 0, 64], dtype=np.uint8)
    images[3] = np.asarray([12, 34, 56], dtype=np.uint8)
    labels = np.asarray([0, 3, 8, 9], dtype=np.uint8)
    _write_cifar10_batch(data_dir / "test_batch.bin", images, labels)

    bundle = load_dataset(
        InputConfig(
            type="cifar10",
            dataset_root=str(cifar_root),
            split="test",
            color_mode="grayscale",
            thumbnail_mode="atlas",
            max_samples=None,
        ),
        project_root=tmp_path,
        thumbnail_dir=tmp_path / "assets" / "thumbnails",
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (4, 32 * 32)
    assert bundle.image_shape == (32, 32)
    assert bundle.metadata["dataset"].unique().tolist() == ["cifar10_grayscale"]
    assert bundle.metadata["color_mode"].unique().tolist() == ["grayscale"]
    assert bundle.vectors[:, 0].tolist() == [76, 75, 7, 30]
    atlas_path = Path(bundle.metadata["sprite_atlas_path"].iloc[0])
    with Image.open(atlas_path) as atlas:
        assert atlas.convert("RGB").getpixel((0, 0)) == (76, 76, 76)


def test_large_prepacked_atlases_are_compacted_to_webp(tmp_path: Path) -> None:
    source = tmp_path / "atlas.png"
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[..., :3] = [120, 40, 200]
    pixels[..., 3] = 255
    Image.fromarray(pixels, mode="RGBA").save(source)
    bundle = AtlasBundle(
        frame=pd.DataFrame({"row_id": [0]}),
        atlas_paths=[source],
        palette={},
        tile_size=32,
        atlas_columns=2,
    )

    compact = _compact_atlas_bundle(bundle, threshold_bytes=0)

    assert compact.atlas_paths[0].suffix == ".webp"
    assert compact.atlas_paths[0].is_file()
    assert compact.atlas_paths[0].stat().st_size < source.stat().st_size
    assert atlas_data_url(compact.atlas_paths[0]).startswith(
        "data:image/webp;base64,"
    )


def test_atlas_compaction_is_safe_across_concurrent_renders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "atlas.png"
    Image.new("RGBA", (32, 32), (120, 40, 200, 255)).save(source)
    legacy_temporary = tmp_path / ".atlas_q90.webp.tmp"
    legacy_temporary.write_bytes(b"incomplete")
    encoded = io.BytesIO()
    Image.new("RGBA", (32, 32), (120, 40, 200, 255)).save(
        encoded,
        format="WEBP",
    )
    webp_bytes = encoded.getvalue()
    barrier = threading.Barrier(2)

    def synchronized_save(self, path, **kwargs):
        del self, kwargs
        Path(path).write_bytes(webp_bytes)
        barrier.wait(timeout=5)

    monkeypatch.setattr(Image.Image, "save", synchronized_save)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(_compact_atlas_path, [source, source]))

    assert results[0] == results[1]
    assert results[0].is_file()
    assert not legacy_temporary.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_numpy_loader_supports_vectors_and_image_preview(tmp_path: Path) -> None:
    data_path = tmp_path / "digits.npy"
    labels_path = tmp_path / "labels.npy"
    np.save(data_path, np.arange(5 * 16, dtype=np.float32).reshape(5, 16))
    np.save(labels_path, np.asarray([0, 1, 0, 1, 2]))

    bundle = load_dataset(
        InputConfig(
            type="numpy",
            data_path=str(data_path),
            labels_path=str(labels_path),
            image_shape=(4, 4),
            max_samples=3,
        ),
        project_root=tmp_path,
        thumbnail_dir=tmp_path / "previews",
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (3, 16)
    assert len(bundle.metadata) == 3
    assert bundle.metadata["image_path"].map(lambda value: Path(value).is_file()).all()


def test_metadata_loader_filters_resolves_and_deduplicates(tmp_path: Path) -> None:
    experiment = tmp_path / "outputs" / "run"
    image_path = experiment / "images" / "p1" / "image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="red").save(image_path)
    metadata_path = experiment / "metadata" / "per_image_metadata.jsonl"
    metadata_path.parent.mkdir()
    valid = {
        "output_path": str(image_path.relative_to(tmp_path)),
        "prompt_id": "p1",
        "family": "color",
        "status": "success",
    }
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(valid),
                json.dumps(valid),
                json.dumps({**valid, "output_path": "missing.png"}),
                json.dumps({**valid, "status": "failed"}),
                "{bad json",
            ]
        ),
        encoding="utf-8",
    )

    result = load_image_metadata(
        InputConfig(
            type="image_metadata",
            experiment_dir=str(experiment),
        ),
        project_root=tmp_path,
    )

    assert len(result.frame) == 1
    assert result.frame.iloc[0]["image_path"] == str(image_path.resolve())
    assert result.duplicate_rows == 1
    assert result.missing_images == 1
    assert result.malformed_rows == 1


def test_raw_feature_runner_uses_dataset_vectors_without_model(tmp_path: Path) -> None:
    vectors = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1],
            "image_path": ["", ""],
            "label": ["a", "b"],
        }
    )
    bundle = DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id="test-source",
        source_description="test vectors",
        total_rows=2,
    )

    result = compute_or_load_features(
        config=FeatureConfig(mode="raw", name="input_vectors"),
        dataset=bundle,
        output_dir=tmp_path / "explorer",
        save=False,
        model_loader=lambda _: (_ for _ in ()).throw(AssertionError("model loaded")),
    )

    assert np.array_equal(result.features, vectors)
    assert result.metadata["feature_mode"].tolist() == ["raw", "raw"]
    assert not (tmp_path / "explorer" / "features").exists()


def test_dinov2_feature_runner_converts_mnist_vectors_to_rgb(tmp_path: Path) -> None:
    vectors = np.zeros((3, 28 * 28), dtype=np.float32)
    vectors[1, 100:200] = 1.0
    vectors[2, 300:500] = 0.5
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "image_path": ["", "", ""],
            "label": ["0", "1", "2"],
        }
    )
    bundle = DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id="mnist-source",
        source_description="MNIST vectors",
        total_rows=3,
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
    )

    class FakeExtractor:
        def extract(self, images):
            assert all(image.mode == "RGB" for image in images)
            assert all(image.size == (28, 28) for image in images)
            return np.asarray(
                [
                    [np.asarray(image)[..., 0].mean(), index]
                    for index, image in enumerate(images)
                ],
                dtype=np.float32,
            )

    result = compute_or_load_features(
        config=FeatureConfig(
            mode="dinov2",
            name="dinov2_test",
            batch_size=2,
            normalize=False,
        ),
        dataset=bundle,
        output_dir=tmp_path / "explorer",
        save=False,
        model_loader=lambda _: FakeExtractor(),
    )

    assert result.features.shape == (3, 2)
    assert result.metadata["feature_mode"].tolist() == ["dinov2"] * 3


def test_dinov2_feature_runner_preserves_cifar_rgb_vectors(tmp_path: Path) -> None:
    images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    images[0] = np.asarray([255, 0, 0], dtype=np.uint8)
    images[1] = np.asarray([0, 128, 255], dtype=np.uint8)
    bundle = DatasetBundle(
        metadata=pd.DataFrame(
            {
                "row_id": [0, 1],
                "image_path": ["", ""],
                "label": ["airplane", "ship"],
            }
        ),
        vectors=images.reshape(2, -1),
        source_id="cifar10-source",
        source_description="CIFAR-10 vectors",
        total_rows=2,
        image_shape=(32, 32, 3),
        value_range=(0.0, 255.0),
    )

    class FakeExtractor:
        def extract(self, batch):
            pixels = [np.asarray(image)[0, 0].tolist() for image in batch]
            assert pixels == [[255, 0, 0], [0, 128, 255]]
            return np.asarray(pixels, dtype=np.float32)

    result = compute_or_load_features(
        config=FeatureConfig(
            mode="dinov2",
            name="dinov2_test",
            batch_size=2,
            normalize=False,
        ),
        dataset=bundle,
        output_dir=tmp_path / "explorer",
        save=False,
        model_loader=lambda _: FakeExtractor(),
    )

    assert result.features.tolist() == [[255.0, 0.0, 0.0], [0.0, 128.0, 255.0]]


def test_dinov2_feature_runner_replicates_grayscale_cifar_channels(
    tmp_path: Path,
) -> None:
    vectors = np.asarray(
        [
            np.full(32 * 32, 76, dtype=np.uint8),
            np.full(32 * 32, 150, dtype=np.uint8),
        ]
    )
    bundle = DatasetBundle(
        metadata=pd.DataFrame(
            {
                "row_id": [0, 1],
                "image_path": ["", ""],
                "label": ["airplane", "ship"],
            }
        ),
        vectors=vectors,
        source_id="cifar10-grayscale-source",
        source_description="CIFAR-10 grayscale vectors",
        total_rows=2,
        image_shape=(32, 32),
        value_range=(0.0, 255.0),
    )

    class FakeExtractor:
        def extract(self, batch):
            pixels = [np.asarray(image)[0, 0].tolist() for image in batch]
            assert pixels == [[76, 76, 76], [150, 150, 150]]
            return np.asarray(pixels, dtype=np.float32)

    result = compute_or_load_features(
        config=FeatureConfig(
            mode="dinov2",
            name="dinov2_grayscale_test",
            batch_size=2,
            normalize=False,
        ),
        dataset=bundle,
        output_dir=tmp_path / "explorer",
        save=False,
        model_loader=lambda _: FakeExtractor(),
    )

    assert result.features.tolist() == [
        [76.0, 76.0, 76.0],
        [150.0, 150.0, 150.0],
    ]


def test_feature_cache_reattaches_current_dataset_metadata(tmp_path: Path) -> None:
    vectors = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    first = DatasetBundle(
        metadata=pd.DataFrame(
            {"row_id": [0, 1], "image_path": ["first-0.png", "first-1.png"]}
        ),
        vectors=vectors,
        source_id="shared-source",
        source_description="first",
        total_rows=2,
    )
    second = DatasetBundle(
        metadata=pd.DataFrame(
            {"row_id": [0, 1], "image_path": ["second-0.png", "second-1.png"]}
        ),
        vectors=vectors,
        source_id="shared-source",
        source_description="second",
        total_rows=2,
    )
    config = FeatureConfig(mode="raw", name="shared")

    compute_or_load_features(
        config=config,
        dataset=first,
        output_dir=tmp_path / "cache",
    )
    loaded = compute_or_load_features(
        config=config,
        dataset=second,
        output_dir=tmp_path / "cache",
    )

    assert loaded.loaded_from_cache is True
    assert loaded.metadata["image_path"].tolist() == [
        "second-0.png",
        "second-1.png",
    ]


def test_projection_and_local_diagnostics_handle_small_dataset() -> None:
    features = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "row_id": range(6),
            "image_path": [f"{index}.png" for index in range(6)],
            "label": ["0", "0", "1", "1", "2", "2"],
            "prompt_id": ["a", "a", "b", "b", "c", "c"],
            "family": ["x", "x", "x", "x", "y", "y"],
            "status": ["success"] * 6,
        }
    )

    projection = compute_projection(
        features,
        ProjectionConfig(method="pca"),
        method="pca",
    )
    diagnostics = compute_local_diagnostics(
        features,
        metadata,
        LocalDiagnosticsConfig(k_neighbors=15),
    )

    assert projection.shape == (6, 2)
    assert diagnostics["knn_radius_k15"].notna().all()
    assert diagnostics["participation_ratio_k15"].between(1, 5).all()
    assert diagnostics["distance_to_label_centroid"].gt(0).all()
    assert diagnostics["outlier_score"].notna().all()


def test_projection_runner_persists_three_components(tmp_path: Path) -> None:
    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    config = ProjectionConfig(
        variants=(
            diagnostics_config_from_dict(
                {
                    "explorer_name": "three",
                    "input": {"type": "numpy", "data_path": "unused.npy"},
                    "projection": {
                        "variants": [
                            {
                                "name": "PCA 3D",
                                "key": "pca_3d",
                                "method": "pca",
                                "n_components": 3,
                            }
                        ]
                    },
                    "explorer": {"renderer": "three3d"},
                }
            ).projection.variants[0],
        )
    )

    result = compute_or_load_projections(
        features,
        pd.Series(range(6)),
        config,
        tmp_path,
        feature_name="raw",
        save=False,
    )

    assert {"pca_3d_x", "pca_3d_y", "pca_3d_z"} <= set(result.columns)


def test_three_renderer_accepts_mixed_projection_dimensions() -> None:
    config = diagnostics_config_from_dict(
        {
            "explorer_name": "mixed",
            "input": {"type": "numpy", "data_path": "unused.npy"},
            "projection": {
                "variants": [
                    {
                        "name": "UMAP 2D",
                        "key": "umap_2d",
                        "method": "umap",
                        "n_components": 2,
                    },
                    {
                        "name": "UMAP 3D",
                        "key": "umap_3d",
                        "method": "umap",
                        "n_components": 3,
                    },
                ]
            },
            "explorer": {"renderer": "three3d"},
        }
    )

    assert [variant.n_components for variant in config.projection.variants] == [2, 3]


def test_projection_diagnostics_follow_each_projection() -> None:
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "label": ["a", "a", "b", "b"],
        }
    )
    projections = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "umap_x": [0.0, 0.1, 10.0, 10.1],
            "umap_y": [0.0, 0.0, 0.0, 0.0],
            "tsne_x": [0.0, 10.0, 0.1, 10.1],
            "tsne_y": [0.0, 0.0, 0.0, 0.0],
        }
    )

    diagnostics = compute_projection_diagnostics(
        projections,
        metadata,
        k_neighbors=1,
    )

    assert diagnostics["umap_label_agreement_k1"].tolist() == [1.0] * 4
    assert diagnostics["tsne_label_agreement_k1"].tolist() == [0.0] * 4
    assert diagnostics["umap_nearest_row_id"].tolist() == [1, 0, 3, 2]


def test_projection_diagnostics_include_z_distance() -> None:
    metadata = pd.DataFrame({"row_id": [0, 1, 2], "label": ["a", "a", "b"]})
    projections = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "umap_3d_x": [0.0, 0.0, 0.0],
            "umap_3d_y": [0.0, 0.0, 0.0],
            "umap_3d_z": [0.0, 2.0, 10.0],
        }
    )

    diagnostics = compute_projection_diagnostics(
        projections,
        metadata,
        k_neighbors=1,
    )

    assert diagnostics["umap_3d_knn_radius_k1"].tolist() == [2.0, 2.0, 8.0]


def test_named_projection_variants_load_precomputed_coordinates(tmp_path: Path) -> None:
    coordinates = [[-1.0, 2.0], [0.0, 1.0], [1.0, 0.0]]
    projection_path = tmp_path / "reference.json"
    projection_path.write_text(json.dumps(coordinates), encoding="utf-8")
    config = diagnostics_config_from_dict(
        {
            "explorer_name": "variants",
            "input": {
                "type": "numpy",
                "data_path": "unused.npy",
            },
            "projection": {
                "variants": [
                    {
                        "name": "UMAP min_dist=0.8",
                        "key": "umap_min_dist_0_8",
                        "method": "umap",
                        "min_dist": 0.8,
                        "source_path": str(projection_path),
                    }
                ]
            },
        }
    )

    variants = projection_variants(config.projection)
    result = compute_or_load_projections(
        np.zeros((3, 4), dtype=np.float32),
        pd.Series([0, 1, 2]),
        config.projection,
        tmp_path / "output",
        feature_name="raw",
        save=False,
        project_root=tmp_path,
    )

    assert variants[0].name == "UMAP min_dist=0.8"
    assert result["umap_min_dist_0_8_x"].tolist() == [-1.0, 0.0, 1.0]


def test_local_reference_projection_writer_uses_mldata_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mnist_root = tmp_path / "mnist"
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    _write_mnist_split(
        mnist_root,
        images,
        np.asarray([2, 0, 1, 0], dtype=np.uint8),
        split="train",
    )
    _write_mnist_split(
        mnist_root,
        images,
        np.asarray([1, 0, 2, 1], dtype=np.uint8),
        split="test",
    )

    def fake_projection(pixels, *, method, n_jobs):
        del n_jobs
        assert method == "umap"
        assert pixels.shape == (8, 784)
        return np.column_stack(
            [np.arange(len(pixels)), -np.arange(len(pixels))]
        ).astype(np.float32)

    monkeypatch.setattr(
        "fm_lab.image_diagnostics.mnist_reference_projections._compute_method",
        fake_projection,
    )
    output_dir = tmp_path / "coordinates"
    manifest = compute_mnist_reference_projections(
        dataset_root=mnist_root,
        output_dir=output_dir,
        methods=["umap"],
    )

    labels = json.loads((output_dir / "mnist_labels.json").read_text())
    coordinates = json.loads((output_dir / "mnist_embeddings.json").read_text())
    assert labels == [0, 0, 1, 2, 0, 1, 1, 2]
    assert coordinates[-1] == [7.0, -7.0]
    assert manifest["samples"] == 8


def test_labels_merge_into_explorer_data(tmp_path: Path) -> None:
    labels_path = tmp_path / "manual_labels.csv"
    save_manual_label(
        labels_path,
        row_id=1,
        manual_label="outlier",
        manual_notes="between clusters",
    )
    metadata = pd.DataFrame(
        [
            {"row_id": 0, "image_path": "a.png"},
            {"row_id": 1, "image_path": "b.png"},
        ]
    )
    projections = pd.DataFrame(
        {"row_id": [0, 1], "umap_x": [0.0, 1.0], "umap_y": [1.0, 0.0]}
    )
    diagnostics = pd.DataFrame({"row_id": [0, 1], "outlier_score": [1.0, 2.0]})

    explorer = build_explorer_data(
        metadata,
        projections,
        diagnostics,
        labels_path=labels_path,
    )

    assert explorer.loc[0, "manual_label"] == "unlabeled"
    assert explorer.loc[1, "manual_label"] == "outlier"
    assert len(load_manual_labels(labels_path)) == 1


def test_combined_explorer_aligns_and_renames_precomputed_projections(
    tmp_path: Path,
) -> None:
    base = pd.DataFrame(
        {
            "row_id": [0, 1],
            "split": ["test", "test"],
            "source_index": [10, 20],
            "label": ["1", "2"],
        }
    )
    source = pd.DataFrame(
        {
            "row_id": [99, 98],
            "split": ["test", "test"],
            "source_index": [20, 10],
            "label": ["2", "1"],
            "umap_x": [2.0, 1.0],
            "umap_y": [4.0, 3.0],
            "umap_z": [6.0, 5.0],
        }
    )
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)

    combined = combine_explorer_tables(
        base,
        [
            {
                "data_path": str(source_path),
                "projections": {"umap": "raw_umap_3d"},
            }
        ],
        align_on=["split", "source_index"],
        project_root=tmp_path,
    )

    assert combined["raw_umap_3d_x"].tolist() == [1.0, 2.0]
    assert combined["raw_umap_3d_y"].tolist() == [3.0, 4.0]
    assert combined["raw_umap_3d_z"].tolist() == [5.0, 6.0]


def test_combined_explorer_rejects_different_sample_sets(tmp_path: Path) -> None:
    base = pd.DataFrame(
        {
            "row_id": [0, 1],
            "source_index": [10, 20],
        }
    )
    source = pd.DataFrame(
        {
            "row_id": [0, 1],
            "source_index": [10, 30],
            "umap_x": [0.0, 1.0],
            "umap_y": [1.0, 0.0],
        }
    )
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path, index=False)

    with np.testing.assert_raises_regex(ValueError, "does not match base samples"):
        combine_explorer_tables(
            base,
            [
                {
                    "data_path": str(source_path),
                    "projections": {"umap": "raw_umap_2d"},
                }
            ],
            align_on=["source_index"],
            project_root=tmp_path,
        )


def test_auto_explorer_discovers_compatible_views_and_prefers_id_table(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset_explorer"
    first_dir = root / "raw_2d" / "explorer"
    second_dir = root / "learned_3d" / "explorer"
    incompatible_dir = root / "other_samples" / "explorer"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)
    incompatible_dir.mkdir(parents=True)
    metadata = {
        "row_id": [0, 1, 2],
        "dataset": ["mnist"] * 3,
        "split": ["test"] * 3,
        "source_index": [10, 20, 30],
        "label": ["1", "2", "3"],
        "sprite_atlas_path": ["atlas.webp"] * 3,
        "sprite_atlas_index": [0, 1, 2],
        "sprite_atlas_column": [0, 1, 2],
        "sprite_atlas_row": [0, 0, 0],
        "sprite_tile_size": [28] * 3,
        "sprite_atlas_columns": [3] * 3,
    }
    pd.DataFrame(
        metadata
        | {
            "umap_x": [0.0, 1.0, 2.0],
            "umap_y": [2.0, 1.0, 0.0],
        }
    ).to_parquet(first_dir / "explorer_data.parquet", index=False)
    pd.DataFrame(
        metadata
        | {
            "umap_x": [0.0, 1.0, 2.0],
            "umap_y": [2.0, 1.0, 0.0],
            "mle_lid_k15": [1.0, 2.0, 3.0],
        }
    ).to_parquet(first_dir / "explorer_data_with_id.parquet", index=False)
    pd.DataFrame(
        metadata
        | {
            "umap_3d_x": [1.0, 2.0, 3.0],
            "umap_3d_y": [3.0, 2.0, 1.0],
            "umap_3d_z": [0.5, 0.0, -0.5],
        }
    ).iloc[::-1].to_parquet(second_dir / "explorer_data.parquet", index=False)
    pd.DataFrame(
        {
            **metadata,
            "source_index": [11, 21, 31],
            "pca_x": [0.0, 1.0, 2.0],
            "pca_y": [0.0, 1.0, 2.0],
        }
    ).to_parquet(incompatible_dir / "explorer_data.parquet", index=False)

    groups = discover_explorer_groups(root)
    compatible = next(group for group in groups if group.projection_count == 2)
    loaded = load_discovered_explorer_group(compatible)

    assert len(groups) == 2
    assert compatible.sample_count == 3
    assert len(loaded.projection_names) == 2
    assert loaded.explorer_config.renderer == "three3d"
    assert loaded.frame["mle_lid_k15"].tolist() == [1.0, 2.0, 3.0]
    assert loaded.data_path == first_dir / "explorer_data_with_id.parquet"
    assert any(
        path.name == "explorer_data_with_id.parquet"
        for path in loaded.source_paths
    )


def test_mnist_dry_run_requires_no_model(tmp_path: Path) -> None:
    mnist_root = tmp_path / "mnist"
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    labels = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    _write_mnist_split(mnist_root, images, labels, split="test")
    config = diagnostics_config_from_dict(_raw_config(str(mnist_root)))

    result = run_diagnostics_build(config, project_root=tmp_path, dry_run=True)

    assert result["input_type"] == "mnist"
    assert result["selected_samples"] == 4
    assert result["feature_mode"] == "raw"
    assert result["requires_model_download"] is False
    assert not (tmp_path / "outputs").exists()


def test_build_runs_configured_id_estimation(tmp_path: Path, monkeypatch) -> None:
    mnist_root = tmp_path / "mnist"
    images = np.arange(4 * 28 * 28, dtype=np.uint8).reshape(4, 28, 28)
    labels = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    _write_mnist_split(mnist_root, images, labels, split="test")
    raw = _raw_config(str(mnist_root))
    raw["output"]["root_dir"] = str(tmp_path / "outputs")
    raw["projection"] = {
        "method": "pca",
        "also_compute_pca": False,
    }
    raw["diagnostics"] = {"enabled": False}
    raw["id_estimation"] = {
        "enabled": True,
        "config_path": "configs/id.yaml",
    }
    captured = {}
    fake_id_config = id_config_from_dict(
        {
            "id_estimation_name": "test_id",
            "input": {
                "diagnostics_dir": "unused",
                "embedding_source": "features/raw_pixels_features.npy",
                "source_type": "npy",
            },
        }
    )

    def fake_load(path: Path):
        captured["config_path"] = path
        return fake_id_config

    def fake_run(config, *, project_root):
        captured["config"] = config
        captured["project_root"] = project_root
        return {"merged_explorer_path": "explorer_data_with_id.parquet"}

    monkeypatch.setattr(
        "fm_lab.image_diagnostics.runner.load_id_config",
        fake_load,
    )
    monkeypatch.setattr(
        "fm_lab.image_diagnostics.runner.run_id_estimation",
        fake_run,
    )

    result = run_diagnostics_build(
        diagnostics_config_from_dict(raw),
        project_root=tmp_path,
    )

    assert captured["config_path"] == tmp_path / "configs" / "id.yaml"
    assert captured["config"].input.diagnostics_dir == str(
        (tmp_path / "outputs" / "test_explorer").resolve()
    )
    assert captured["project_root"] == tmp_path
    assert result["id_estimation"]["merged_explorer_path"].endswith(
        "explorer_data_with_id.parquet"
    )


def test_sprite_atlas_packs_and_tints_mnist_thumbnails(tmp_path: Path) -> None:
    image_paths = []
    for index in range(3):
        path = tmp_path / f"digit_{index}.png"
        pixels = np.zeros((8, 8), dtype=np.uint8)
        pixels[2:6, 3:5] = 255
        Image.fromarray(pixels, mode="L").save(path)
        image_paths.append(str(path))
    frame = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "source_index": [10, 11, 12],
            "image_path": image_paths,
            "dataset": ["mnist"] * 3,
            "label": ["0", "1", "2"],
            "umap_x": [0.0, 1.0, 2.0],
            "umap_y": [2.0, 1.0, 0.0],
            "pca_x": [1.0, 0.0, -1.0],
            "pca_y": [0.0, 1.0, 0.0],
        }
    )

    bundle = prepare_sprite_atlases(
        frame,
        output_dir=tmp_path / "atlases",
        tile_size=8,
        max_atlas_size=32,
    )

    assert len(bundle.atlas_paths) == 1
    assert bundle.frame["atlas_column"].tolist() == [0, 1, 2]
    with Image.open(bundle.atlas_paths[0]) as atlas:
        assert atlas.mode == "RGBA"
        assert atlas.getpixel((3, 2))[3] == 255
        assert atlas.getpixel((0, 0))[3] == 0


def test_canvas_html_contains_thumbnail_interactions(tmp_path: Path) -> None:
    path = tmp_path / "digit.png"
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(path)
    frame = pd.DataFrame(
        {
            "row_id": [0],
            "source_index": [7],
            "image_path": [str(path)],
            "dataset": ["mnist"],
            "label": ["4"],
            "umap_x": [0.0],
            "umap_y": [0.0],
            "pca_x": [1.0],
            "pca_y": [1.0],
            "mle_lid_k15": [2.5],
            "pca_dim_95_k15": [3.0],
        }
    )
    bundle = prepare_sprite_atlases(frame, output_dir=tmp_path / "atlases")

    html = build_canvas_html(
        bundle,
        height=600,
        config=diagnostics_config_from_dict(
            {
                "explorer_name": "canvas",
                "input": {"type": "numpy", "data_path": "unused.npy"},
                "explorer": {
                    "preview_mode": "original",
                    "compute_projection_diagnostics": True,
                    "show_workspace": False,
                },
            }
        ).explorer,
    )

    assert 'id="plot"' in html
    assert 'id="preview"' in html
    assert "pointermove" in html
    assert 'addEventListener("wheel"' in html
    assert '"projections":["UMAP","PCA"]' in html
    assert "data:image/png;base64," in html
    assert '"previewMode":"original"' in html
    assert "projectionDiagnostics" in html
    assert "previewTileContext.getImageData" in html
    assert '"mle_lid_k15":2.5' in html
    assert '"pca_dim_95_k15":3.0' in html
    assert 'id="class-filter"' in html
    assert "All classes" in html
    assert "syncClassFilter" in html
    assert "for (const index of visibleIndices)" in html
    assert "${visibleIndices.length.toLocaleString()} samples" in html


def test_three_html_contains_mixed_dimensions_and_thumbnail_shader(
    tmp_path: Path,
) -> None:
    path = tmp_path / "digit.png"
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(path)
    frame = pd.DataFrame(
        {
            "row_id": [0],
            "source_index": [7],
            "image_path": [str(path)],
            "dataset": ["mnist"],
            "label": ["4"],
            "umap_2d_x": [0.5],
            "umap_2d_y": [1.5],
            "umap_3d_x": [1.0],
            "umap_3d_y": [2.0],
            "umap_3d_z": [3.0],
            "mle_lid_k15": [2.5],
        }
    )
    bundle = prepare_sprite_atlases(frame, output_dir=tmp_path / "atlases")
    config = diagnostics_config_from_dict(
        {
            "explorer_name": "three",
            "input": {"type": "numpy", "data_path": "unused.npy"},
            "projection": {
                "variants": [
                    {
                        "name": "UMAP 2D",
                        "key": "umap_2d",
                        "method": "umap",
                        "n_components": 2,
                    },
                    {
                        "name": "UMAP 3D",
                        "key": "umap_3d",
                        "method": "umap",
                        "n_components": 3,
                    }
                ]
            },
            "explorer": {"renderer": "three3d"},
        }
    )

    html = build_three_html(
        bundle,
        height=600,
        config=config.explorer,
        projection_names={"umap_2d": "UMAP 2D", "umap_3d": "UMAP 3D"},
        three_source="window.THREE = {};",
    )

    assert "THREE.PerspectiveCamera" in html
    assert "THREE.WebGLRenderer" in html
    assert "Map Z" in html
    assert '"projectionDimensions":{"UMAP 2D":2,"UMAP 3D":3}' in html
    assert '"coordinates":{"UMAP 2D":[0.0,0.0,0.0],"UMAP 3D":[0.0,0.0,0.0]}' in html
    assert '"details":{"mle_lid_k15":2.5}' in html
    assert "texture2D(textureAtlas" in html
    assert "atlasImages[atlas] = texture.image" in html
    assert "Loading ${DATA.points.length.toLocaleString()} samples..." in html
    assert "new Image()" not in html
    assert 'id="class-filter"' in html
    assert "All classes" in html
    assert "syncClassFilter" in html
    assert '"fashion_mnist"' in html
    assert "if (loadedTextures.length) buildPointClouds(loadedTextures)" in html
    assert "${visibleIndices.length.toLocaleString()} samples" in html


def _raw_config(dataset_root: str) -> dict:
    return {
        "explorer_name": "test_explorer",
        "input": {
            "type": "mnist",
            "dataset_root": dataset_root,
            "split": "test",
            "max_samples": 10,
        },
        "features": {
            "mode": "raw",
            "name": "raw_pixels",
            "normalize": False,
        },
        "projection": {"method": "pca"},
        "output": {"root_dir": "outputs"},
    }


def _write_mnist_split(
    root: Path,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if split == "test":
        image_name = "t10k-images-idx3-ubyte.gz"
        label_name = "t10k-labels-idx1-ubyte.gz"
    else:
        image_name = "train-images-idx3-ubyte.gz"
        label_name = "train-labels-idx1-ubyte.gz"
    with gzip.open(root / image_name, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, len(images), 28, 28))
        handle.write(np.asarray(images, dtype=np.uint8).tobytes())
    with gzip.open(root / label_name, "wb") as handle:
        handle.write(struct.pack(">II", 2049, len(labels)))
        handle.write(np.asarray(labels, dtype=np.uint8).tobytes())


def _write_cifar10_batch(
    path: Path,
    images: np.ndarray,
    labels: np.ndarray,
) -> None:
    pixels = images.transpose(0, 3, 1, 2).reshape(len(images), -1)
    records = np.concatenate([labels.reshape(-1, 1), pixels], axis=1)
    path.write_bytes(np.asarray(records, dtype=np.uint8).tobytes())
