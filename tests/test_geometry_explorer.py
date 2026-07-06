from __future__ import annotations

import gzip
import io
import struct
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fm_lab.experiments.factory import build_model, build_target
from fm_lab.experiments.run_explorer import main as explorer_cli_main
from fm_lab.geometry_explorer.app import _model_run_labels, _trajectory_views_by_run
from fm_lab.geometry_explorer.bundles import (
    load_projection_payload,
    load_trajectory_payload,
)
from fm_lab.geometry_explorer.display import metric_label, model_run_label
from fm_lab.geometry_explorer.mnist_labeling import (
    label_fashion_mnist_dataset_variant,
    label_mnist_dataset_variant,
)
from fm_lab.geometry_explorer.model_diagnostics import build_model_diagnostics
from fm_lab.geometry_explorer.registry import GeometryRegistry
from fm_lab.geometry_explorer.variants import (
    DatasetVariantConfig,
    build_dataset_variant,
)
from fm_lab.geometry_explorer.viewer import build_geometry_html
from fm_lab.geometry_explorer.views import _sample_view_dataset, build_projection_view
from fm_lab.image_diagnostics.dataset_loader import DatasetBundle
from fm_lab.image_diagnostics.explorer_payload import sample_metric_columns
from fm_lab.image_diagnostics.save_utils import read_parquet, write_parquet
from fm_lab.utils.checkpoints import save_checkpoint
from fm_lab.utils.config import ConfigError


def test_registry_registers_dataset_projection_and_trajectory(tmp_path: Path) -> None:
    registry = GeometryRegistry(tmp_path / "workspace")
    dataset_path = tmp_path / "dataset.parquet"
    write_parquet(pd.DataFrame({"row_id": [0], "label": ["1"]}), dataset_path)
    coordinates_path = tmp_path / "coords.npz"
    trajectory_path = tmp_path / "trajectory.npy"
    np.savez(
        coordinates_path,
        trajectory=np.zeros((2, 1, 3), dtype=np.float32),
        target=np.zeros((1, 3), dtype=np.float32),
        generated=np.zeros((1, 3), dtype=np.float32),
    )
    np.save(trajectory_path, np.zeros((2, 1, 784), dtype=np.float32))

    registry.register_dataset_variant(
        variant_id="mnist/test",
        family="mnist",
        variant="test",
        base="original",
        split="train",
        dataset_path=dataset_path,
        data_path=None,
        labels_path=None,
        config_path=None,
        row_count=1,
        label_counts={"1": 1},
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
    )
    registry.register_projection_view(
        view_id="view",
        variant_id="mnist/test",
        feature_name="raw_pixels",
        feature_mode="raw",
        explorer_data_path=dataset_path,
        output_dir=tmp_path,
        projection_names={"pca_3d": "PCA 3D"},
        renderer="three3d",
        row_count=1,
    )
    registry.register_model_run(
        run_id="run",
        run_dir=tmp_path / "run",
        variant_id="mnist/test",
        family="mnist",
        variant="test",
        config_path=None,
        metrics_path=None,
    )
    registry.register_trajectory_view(
        view_id="traj",
        run_id="run",
        variant_id="mnist/test",
        solver="euler",
        nfe=4,
        coordinates_path=coordinates_path,
        trajectory_path=trajectory_path,
        generated_path=None,
        target_path=None,
        labels_path=None,
        output_dir=tmp_path,
        interactive_path=None,
        n_steps=2,
        n_trajectories=1,
    )

    assert registry.dataset_variants()[0].variant_id == "mnist/test"
    assert registry.projection_views("mnist/test")[0].view_id == "view"
    assert registry.model_runs("mnist/test")[0].run_id == "run"
    assert registry.trajectory_views(variant_id="mnist/test")[0].view_id == "traj"
    assert registry.trajectory_views(run_id="run")[0].view_id == "traj"
    grouped = _trajectory_views_by_run(registry.trajectory_views(variant_id="mnist/test"))
    assert list(grouped) == ["run"]
    assert grouped["run"][0].view_id == "traj"


def test_model_run_labels_describe_model_and_prediction_target(tmp_path: Path) -> None:
    registry = GeometryRegistry(tmp_path / "workspace")
    dataset_path = tmp_path / "dataset.parquet"
    write_parquet(pd.DataFrame({"row_id": [0], "label": ["1"]}), dataset_path)
    config_path = tmp_path / "run" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
model:
  name: image_unet
objective:
  name: flow_matching
  model_output: x
""",
        encoding="utf-8",
    )
    registry.register_dataset_variant(
        variant_id="mnist/test",
        family="mnist",
        variant="test",
        base="original",
        split="train",
        dataset_path=dataset_path,
        data_path=None,
        labels_path=None,
        config_path=None,
        row_count=1,
        label_counts={"1": 1},
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
    )
    registry.register_model_run(
        run_id="run_xpred",
        run_dir=config_path.parent,
        variant_id="mnist/test",
        family="mnist",
        variant="test",
        config_path=config_path,
        metrics_path=None,
    )

    labels = _model_run_labels(registry.model_runs("mnist/test"))

    assert labels["run_xpred"] == "Image U-Net · FM x-pred · Run Xpred"
    assert model_run_label(run_id="run_velocity") == "Run Velocity"


def test_mnist_long_tail_variant_has_exact_counts_and_training_target(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "mnist"
    _write_fake_mnist(data_root, split="train", count=30)
    config = DatasetVariantConfig(
        family="mnist",
        variant="long_tail_test",
        base="original",
        split="train",
        seed=7,
        input={"dataset_root": str(data_root), "split": "train", "order": "source"},
        selection={"per_class_counts": {"0": 3, "1": 2, "2": 1}},
    )

    result = build_dataset_variant(config, workspace=tmp_path / "workspace")
    assert result["label_counts"] == {"0": 3, "1": 2, "2": 1}
    assert np.load(result["data_path"]).shape == (6, 784)

    target = build_target(
        {
            "data": {
                "name": "mnist",
                "variant_id": "mnist/long_tail_test",
                "workspace": str(tmp_path / "workspace"),
                "normalize": "zero_one",
            }
        }
    )
    samples, labels = target.sample_with_labels(12)
    assert samples.shape == (12, 784)
    assert set(labels.tolist()) <= {0, 1, 2}
    assert target.metadata()["variant_id"] == "mnist/long_tail_test"


def test_label_mnist_dataset_variant_replaces_generated_labels(tmp_path: Path) -> None:
    data_root = tmp_path / "mnist"
    _write_fake_mnist(data_root, split="train", count=24)
    _write_fake_mnist(data_root, split="test", count=12)
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    dataset_dir = workspace / "datasets" / "mnist" / "generated"
    dataset_dir.mkdir(parents=True)

    samples = np.zeros((6, 28 * 28), dtype=np.float32)
    dataset_path = write_parquet(
        pd.DataFrame(
            {
                "row_id": np.arange(len(samples), dtype=int),
                "label": ["generated"] * len(samples),
                "family": ["generated"] * len(samples),
            }
        ),
        dataset_dir / "dataset_index.parquet",
    )
    data_path = dataset_dir / "data.npy"
    labels_path = dataset_dir / "labels.npy"
    np.save(data_path, samples)
    np.save(labels_path, np.asarray(["generated"] * len(samples)))
    registry.register_dataset_variant(
        variant_id="mnist/generated",
        family="mnist",
        variant="generated",
        base="generated",
        split="generated",
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=None,
        row_count=len(samples),
        label_counts={"generated": len(samples)},
        image_shape=(28, 28),
        value_range=(-1.0, 1.0),
    )

    result = label_mnist_dataset_variant(
        variant_id="mnist/generated",
        workspace=workspace,
        data_root=data_root,
        classifier_checkpoint=tmp_path / "mnist_classifier.pt",
        classifier_steps=1,
        classifier_batch_size=4,
        classifier_eval_samples=4,
        device=torch.device("cpu"),
    )

    frame = read_parquet(dataset_path)
    labels = np.load(labels_path)
    row = registry.get_dataset_variant("mnist/generated")

    assert result["rows"] == len(samples)
    assert set(frame["label"].astype(str)) <= {str(value) for value in range(10)}
    assert frame["generated_group"].tolist() == ["generated"] * len(samples)
    assert "classifier_confidence" in frame
    assert frame["label_source"].tolist() == ["mnist_classifier"] * len(samples)
    assert labels.dtype == np.int64
    assert "generated" not in row["label_counts_json"]


def test_label_fashion_mnist_dataset_variant_replaces_generated_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "fashion_mnist"
    _write_fake_idx_dataset(data_root, split="train", count=24)
    _write_fake_idx_dataset(data_root, split="test", count=12)
    _patch_fashion_mnist_checksums(monkeypatch, data_root)
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    dataset_dir = workspace / "datasets" / "fashion_mnist" / "generated"
    dataset_dir.mkdir(parents=True)

    samples = np.zeros((6, 28 * 28), dtype=np.float32)
    dataset_path = write_parquet(
        pd.DataFrame(
            {
                "row_id": np.arange(len(samples), dtype=int),
                "dataset": ["fashion_mnist"] * len(samples),
                "label": ["generated"] * len(samples),
                "family": ["generated"] * len(samples),
            }
        ),
        dataset_dir / "dataset_index.parquet",
    )
    data_path = dataset_dir / "data.npy"
    labels_path = dataset_dir / "labels.npy"
    np.save(data_path, samples)
    np.save(labels_path, np.asarray(["generated"] * len(samples)))
    registry.register_dataset_variant(
        variant_id="fashion_mnist/generated",
        family="fashion_mnist",
        variant="generated",
        base="generated",
        split="generated",
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=None,
        row_count=len(samples),
        label_counts={"generated": len(samples)},
        image_shape=(28, 28),
        value_range=(-1.0, 1.0),
    )

    result = label_fashion_mnist_dataset_variant(
        variant_id="fashion_mnist/generated",
        workspace=workspace,
        data_root=data_root,
        classifier_checkpoint=tmp_path / "fashion_mnist_classifier.pt",
        classifier_steps=1,
        classifier_batch_size=4,
        classifier_eval_samples=4,
        device=torch.device("cpu"),
    )

    frame = read_parquet(dataset_path)
    labels = np.load(labels_path)
    row = registry.get_dataset_variant("fashion_mnist/generated")

    assert result["rows"] == len(samples)
    assert set(frame["label"].astype(str)) <= {
        "Ankle boot",
        "Bag",
        "Coat",
        "Dress",
        "Pullover",
        "Sandal",
        "Shirt",
        "Sneaker",
        "T-shirt/top",
        "Trouser",
    }
    assert frame["generated_group"].tolist() == ["generated"] * len(samples)
    assert "classifier_confidence" in frame
    assert frame["label_source"].tolist() == ["fashion_mnist_classifier"] * len(samples)
    assert labels.dtype == np.int64
    assert "generated" not in row["label_counts_json"]


def test_dataset_variant_builder_supports_fashion_mnist_cifar10_cifar100_and_cinic10(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fashion_root = tmp_path / "fashion_mnist"
    cifar_root = tmp_path / "cifar10"
    cifar100_root = tmp_path / "cifar100"
    cinic_root = tmp_path / "cinic10"
    _write_fake_idx_dataset(fashion_root, split="train", count=20)
    _write_fake_cifar10(cifar_root, count=20)
    _write_fake_cifar100(cifar100_root, count=20)
    _write_fake_cinic10_tar(cinic_root, count=20)
    _patch_fashion_mnist_checksums(monkeypatch, fashion_root)
    workspace = tmp_path / "workspace"

    fashion = build_dataset_variant(
        DatasetVariantConfig(
            family="fashion_mnist",
            variant="tiny",
            split="train",
            input={
                "dataset_root": str(fashion_root),
                "split": "train",
                "download": False,
            },
            selection={"per_class_counts": {"0": 2, "1": 1}},
        ),
        workspace=workspace,
    )
    cifar = build_dataset_variant(
        DatasetVariantConfig(
            family="cifar10",
            variant="tiny",
            split="train",
            input={
                "dataset_root": str(cifar_root),
                "split": "train",
                "download": False,
            },
            selection={"per_class_counts": {"airplane": 2, "automobile": 1}},
        ),
        workspace=workspace,
    )
    cifar_gray = build_dataset_variant(
        DatasetVariantConfig(
            family="cifar10_grayscale",
            variant="tiny",
            split="train",
            input={
                "dataset_root": str(cifar_root),
                "split": "train",
                "download": False,
            },
            selection={"per_class_counts": {"0": 2, "1": 1}},
        ),
        workspace=workspace,
    )
    cifar100 = build_dataset_variant(
        DatasetVariantConfig(
            family="cifar100",
            variant="tiny",
            split="train",
            input={
                "dataset_root": str(cifar100_root),
                "split": "train",
                "download": False,
            },
            selection={"per_class_counts": {"apple": 2, "aquarium_fish": 1}},
        ),
        workspace=workspace,
    )
    cinic = build_dataset_variant(
        DatasetVariantConfig(
            family="cinic10",
            variant="tiny",
            split="train",
            input={
                "dataset_root": str(cinic_root),
                "split": "train",
                "download": False,
            },
            selection={"per_class_counts": {"airplane": 2, "automobile": 1}},
        ),
        workspace=workspace,
    )
    cinic_sampled = build_dataset_variant(
        DatasetVariantConfig(
            family="cinic10",
            variant="tiny_sampled",
            split="train",
            input={
                "dataset_root": str(cinic_root),
                "split": "train",
                "max_samples": 10,
                "sample_strategy": "stratified",
                "download": False,
            },
        ),
        workspace=workspace,
    )

    assert fashion["variant_id"] == "fashion_mnist/tiny"
    assert fashion["label_counts"] == {"Trouser": 1, "T-shirt/top": 2}
    assert np.load(fashion["data_path"]).shape == (3, 784)
    assert cifar["variant_id"] == "cifar10/tiny"
    assert cifar["label_counts"] == {"airplane": 2, "automobile": 1}
    assert np.load(cifar["data_path"]).shape == (3, 32 * 32 * 3)
    assert cifar_gray["variant_id"] == "cifar10_grayscale/tiny"
    assert cifar_gray["label_counts"] == {"airplane": 2, "automobile": 1}
    assert np.load(cifar_gray["data_path"]).shape == (3, 32 * 32)
    assert cifar100["variant_id"] == "cifar100/tiny"
    assert cifar100["label_counts"] == {"apple": 2, "aquarium_fish": 1}
    assert np.load(cifar100["data_path"]).shape == (3, 32 * 32 * 3)
    assert cinic["variant_id"] == "cinic10/tiny"
    assert cinic["label_counts"] == {"airplane": 2, "automobile": 1}
    assert np.load(cinic["data_path"]).shape == (3, 32 * 32 * 3)
    assert cinic_sampled["variant_id"] == "cinic10/tiny_sampled"
    assert cinic_sampled["label_counts"] == {"airplane": 5, "automobile": 5}
    assert np.load(cinic_sampled["data_path"]).shape == (10, 32 * 32 * 3)

    fashion_target = build_target(
        {
            "data": {
                "name": "fashion_mnist",
                "variant_id": "fashion_mnist/tiny",
                "workspace": str(workspace),
                "normalize": "zero_one",
            }
        }
    )
    fashion_samples, fashion_labels = fashion_target.sample_with_labels(5)
    assert fashion_samples.shape == (5, 784)
    assert set(fashion_labels.tolist()) <= {0, 1}
    assert fashion_target.metadata()["image_shape"] == [28, 28]

    cifar_target = build_target(
        {
            "data": {
                "name": "cifar10",
                "variant_id": "cifar10/tiny",
                "workspace": str(workspace),
                "normalize": "minus_one_one",
            }
        }
    )
    cifar_samples, cifar_labels = cifar_target.sample_with_labels(5)
    assert cifar_samples.shape == (5, 32 * 32 * 3)
    assert set(cifar_labels.tolist()) <= {0, 1}
    assert cifar_target.metadata()["image_shape"] == [32, 32, 3]
    assert cifar_target.metadata()["image_value_range"] == [-1.0, 1.0]
    assert float(cifar_samples.min()) >= -1.0
    assert float(cifar_samples.max()) <= 1.0

    cifar_gray_target = build_target(
        {
            "data": {
                "name": "cifar10_grayscale",
                "variant_id": "cifar10_grayscale/tiny",
                "workspace": str(workspace),
                "normalize": "zero_one",
            }
        }
    )
    cifar_gray_samples, _ = cifar_gray_target.sample_with_labels(5)
    assert cifar_gray_samples.shape == (5, 32 * 32)
    assert cifar_gray_target.metadata()["image_shape"] == [32, 32]

    cifar100_target = build_target(
        {
            "data": {
                "name": "cifar100",
                "variant_id": "cifar100/tiny",
                "workspace": str(workspace),
                "normalize": "minus_one_one",
            }
        }
    )
    cifar100_samples, cifar100_labels = cifar100_target.sample_with_labels(5)
    assert cifar100_samples.shape == (5, 32 * 32 * 3)
    assert set(cifar100_labels.tolist()) <= {0, 1}
    assert cifar100_target.metadata()["image_shape"] == [32, 32, 3]

    cinic_target = build_target(
        {
            "data": {
                "name": "cinic10",
                "variant_id": "cinic10/tiny",
                "workspace": str(workspace),
                "normalize": "minus_one_one",
            }
        }
    )
    cinic_samples, cinic_labels = cinic_target.sample_with_labels(5)
    assert cinic_samples.shape == (5, 32 * 32 * 3)
    assert set(cinic_labels.tolist()) <= {0, 1}
    assert cinic_target.metadata()["image_shape"] == [32, 32, 3]


def test_projection_view_sampling_can_stratify_by_label() -> None:
    labels = np.asarray(["a"] * 10 + ["b"] * 10 + ["c"] * 10, dtype=object)
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(len(labels)),
            "label": labels,
        }
    )
    vectors = np.arange(len(labels) * 2, dtype=np.float32).reshape(len(labels), 2)
    dataset = DatasetBundle(
        metadata=metadata,
        vectors=vectors,
        source_id="full",
        source_description="full dataset",
        total_rows=len(metadata),
        image_shape=(2,),
        value_range=(0.0, 1.0),
    )

    sampled = _sample_view_dataset(
        dataset,
        max_samples=9,
        seed=123,
        strategy="stratified",
    )

    assert len(sampled.metadata) == 9
    assert sampled.metadata["label"].value_counts().sort_index().to_dict() == {
        "a": 3,
        "b": 3,
        "c": 3,
    }
    assert sampled.vectors is not None
    assert sampled.vectors.shape == (9, 2)
    assert "view-sample:stratified:9:123" in sampled.source_id


def test_model_diagnostics_merge_fm_jacobian_into_projection_view(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3],
            "label": ["0", "0", "1", "1"],
            "source_index": [10, 11, 12, 13],
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    data_path = tmp_path / "data.npy"
    labels_path = tmp_path / "labels.npy"
    write_parquet(metadata, dataset_path)
    np.save(
        data_path,
        np.asarray(
            [
                [0.0, 0.2, 0.4],
                [0.1, 0.3, 0.5],
                [0.2, 0.4, 0.6],
                [0.3, 0.5, 0.7],
            ],
            dtype=np.float32,
        ),
    )
    np.save(labels_path, np.asarray([0, 0, 1, 1], dtype=np.int64))
    registry.register_dataset_variant(
        variant_id="mnist/tiny_model_diag",
        family="mnist",
        variant="tiny_model_diag",
        base="original",
        split="train",
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=None,
        row_count=4,
        label_counts={"0": 2, "1": 2},
        image_shape=(1, 3),
        value_range=(0.0, 1.0),
    )

    view_dir = tmp_path / "view"
    explorer_path = view_dir / "explorer" / "explorer_data.parquet"
    explorer_path.parent.mkdir(parents=True)
    explorer = metadata.assign(
        pca_3d_x=[0.0, 1.0, 0.0, 1.0],
        pca_3d_y=[0.0, 0.0, 1.0, 1.0],
        pca_3d_z=[0.0, 0.0, 0.0, 0.0],
    )
    write_parquet(explorer, explorer_path)
    registry.register_projection_view(
        view_id="tiny_view",
        variant_id="mnist/tiny_model_diag",
        feature_name="raw_pixels",
        feature_mode="raw",
        explorer_data_path=explorer_path,
        output_dir=view_dir,
        projection_names={"pca_3d": "PCA 3D"},
        renderer="three3d",
        row_count=4,
    )

    config = {
        "source": {"name": "gaussian", "dim": 3},
        "data": {"name": "mnist", "normalize": "zero_one"},
        "coupling": {"name": "independent"},
        "model": {"name": "mlp", "hidden_dim": 8, "depth": 1, "time_embedding_dim": 4},
    }
    model = build_model(config, dim=3)
    run_dir = tmp_path / "runs" / "tiny_fm"
    save_checkpoint(
        run_dir / "checkpoint.pt",
        model=model,
        optimizer=None,
        step=0,
        config=config,
        metrics={},
    )

    result = build_model_diagnostics(
        variant_id="mnist/tiny_model_diag",
        run_dir=run_dir,
        workspace=workspace,
        estimators=("fm_jacobian", "fm_flipd"),
        t_values=(0.8,),
        eps=1e-2,
        num_directions=4,
        threshold=1e-3,
        num_trace_samples=1,
        batch_size=2,
        nfe=1,
        solver="euler",
        max_samples=3,
        sample_seed=0,
        device="cpu",
        rebuild_payload=False,
    )

    merged = read_parquet(explorer_path)
    metric = "fm_jacobian_participation_rank_t0800"
    assert metric in merged
    assert int(merged[metric].notna().sum()) == 3
    assert metric in sample_metric_columns(merged)
    assert metric_label(metric) == "FM Jacobian participation rank (t=0.800)"
    assert "fm_flipd_lid_t0800" in merged
    assert "fm_flipd_lid_t0800" in sample_metric_columns(merged)
    assert metric_label("fm_flipd_lid_t0800") == "FM-FLIPD raw ID estimate (t=0.800)"
    group_path = Path(result["merged_views"][0]["group_id_path"])
    group = pd.read_csv(group_path)
    assert "mean_fm_jacobian_participation_rank_t0800" in group
    assert "std_fm_jacobian_participation_rank_t0800" in group
    assert "mean_fm_flipd_lid_t0800" in group
    assert "std_fm_flipd_lid_t0800" in group
    assert set(group["groupby_column"]) == {"__all__", "label"}
    assert result["rows_computed"] == 3


def test_model_diagnostics_rejects_fm_flipd_for_ot_coupling(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    metadata = pd.DataFrame({"row_id": [0], "label": ["0"], "source_index": [10]})
    dataset_path = tmp_path / "dataset.parquet"
    data_path = tmp_path / "data.npy"
    write_parquet(metadata, dataset_path)
    np.save(data_path, np.asarray([[0.0, 0.2, 0.4]], dtype=np.float32))
    registry.register_dataset_variant(
        variant_id="mnist/tiny_ot_model_diag",
        family="mnist",
        variant="tiny_ot_model_diag",
        base="original",
        split="train",
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=None,
        config_path=None,
        row_count=1,
        label_counts={"0": 1},
        image_shape=(1, 3),
        value_range=(0.0, 1.0),
    )
    explorer_path = tmp_path / "view" / "explorer" / "explorer_data.parquet"
    explorer_path.parent.mkdir(parents=True)
    write_parquet(
        metadata.assign(pca_3d_x=[0.0], pca_3d_y=[0.0], pca_3d_z=[0.0]),
        explorer_path,
    )
    registry.register_projection_view(
        view_id="tiny_ot_view",
        variant_id="mnist/tiny_ot_model_diag",
        feature_name="raw_pixels",
        feature_mode="raw",
        explorer_data_path=explorer_path,
        output_dir=tmp_path / "view",
        projection_names={"pca_3d": "PCA 3D"},
        renderer="three3d",
        row_count=1,
    )

    config = {
        "source": {"name": "gaussian", "dim": 3},
        "data": {"name": "mnist", "normalize": "zero_one"},
        "coupling": {"name": "minibatch_ot"},
        "model": {"name": "mlp", "hidden_dim": 8, "depth": 1, "time_embedding_dim": 4},
    }
    model = build_model(config, dim=3)
    run_dir = tmp_path / "runs" / "tiny_ot_fm"
    save_checkpoint(
        run_dir / "checkpoint.pt",
        model=model,
        optimizer=None,
        step=0,
        config=config,
        metrics={},
    )

    try:
        build_model_diagnostics(
            variant_id="mnist/tiny_ot_model_diag",
            run_dir=run_dir,
            workspace=workspace,
            estimators=("fm_flipd",),
            t_values=(0.8,),
            max_samples=1,
            device="cpu",
            rebuild_payload=False,
        )
    except ConfigError as exc:
        assert "independent Gaussian" in str(exc)
        assert "fm_jacobian" in str(exc)
    else:
        raise AssertionError("Expected fm_flipd to reject minibatch_ot checkpoints.")


def test_model_diagnostics_merge_diffusion_estimators_into_projection_view(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "label": ["0", "1", "1"],
            "source_index": [10, 11, 12],
        }
    )
    dataset_path = tmp_path / "dataset.parquet"
    data_path = tmp_path / "data.npy"
    labels_path = tmp_path / "labels.npy"
    write_parquet(metadata, dataset_path)
    np.save(
        data_path,
        np.asarray(
            [
                [0.0, 0.2, 0.4],
                [0.1, 0.3, 0.5],
                [0.2, 0.4, 0.6],
            ],
            dtype=np.float32,
        ),
    )
    np.save(labels_path, np.asarray([0, 1, 1], dtype=np.int64))
    registry.register_dataset_variant(
        variant_id="mnist/tiny_diffusion_model_diag",
        family="mnist",
        variant="tiny_diffusion_model_diag",
        base="original",
        split="train",
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=None,
        row_count=3,
        label_counts={"0": 1, "1": 2},
        image_shape=(1, 3),
        value_range=(0.0, 1.0),
    )

    view_dir = tmp_path / "view"
    explorer_path = view_dir / "explorer" / "explorer_data.parquet"
    explorer_path.parent.mkdir(parents=True)
    explorer = metadata.assign(
        pca_3d_x=[0.0, 1.0, 0.0],
        pca_3d_y=[0.0, 0.0, 1.0],
        pca_3d_z=[0.0, 0.0, 0.0],
    )
    write_parquet(explorer, explorer_path)
    registry.register_projection_view(
        view_id="tiny_diffusion_view",
        variant_id="mnist/tiny_diffusion_model_diag",
        feature_name="raw_pixels",
        feature_mode="raw",
        explorer_data_path=explorer_path,
        output_dir=view_dir,
        projection_names={"pca_3d": "PCA 3D"},
        renderer="three3d",
        row_count=3,
    )

    config = {
        "source": {"name": "gaussian", "dim": 3},
        "data": {"name": "mnist", "normalize": "zero_one"},
        "path": {"name": "gaussian_diffusion", "schedule": "linear", "sigma_min": 1e-4},
        "objective": {"name": "diffusion", "prediction_type": "score"},
        "model": {"name": "mlp", "hidden_dim": 8, "depth": 1, "time_embedding_dim": 4},
    }
    model = build_model(config, dim=3)
    run_dir = tmp_path / "runs" / "tiny_diffusion"
    save_checkpoint(
        run_dir / "checkpoint.pt",
        model=model,
        optimizer=None,
        step=0,
        config=config,
        metrics={},
    )

    result = build_model_diagnostics(
        variant_id="mnist/tiny_diffusion_model_diag",
        run_dir=run_dir,
        workspace=workspace,
        estimators=("diffusion_normal_bundle", "diffusion_flipd"),
        t_values=(0.8,),
        threshold=1e-3,
        num_trace_samples=1,
        num_perturbations=4,
        batch_size=2,
        max_samples=2,
        sample_seed=0,
        device="cpu",
        rebuild_payload=False,
    )

    merged = read_parquet(explorer_path)
    normal_metric = "diffusion_normal_bundle_lid_t0800"
    flipd_metric = "diffusion_flipd_lid_t0800"
    assert normal_metric in merged
    assert flipd_metric in merged
    assert int(merged[normal_metric].notna().sum()) == 2
    assert normal_metric in sample_metric_columns(merged)
    assert flipd_metric in sample_metric_columns(merged)
    assert (
        metric_label(normal_metric)
        == "Diffusion normal-bundle ID upper bound (t=0.800)"
    )
    assert metric_label(flipd_metric) == "Diffusion FLIPD raw ID estimate (t=0.800)"
    group = pd.read_csv(Path(result["merged_views"][0]["group_id_path"]))
    assert "mean_diffusion_normal_bundle_lid_t0800" in group
    assert "std_diffusion_normal_bundle_lid_t0800" in group
    assert "mean_diffusion_flipd_lid_t0800" in group
    assert "std_diffusion_flipd_lid_t0800" in group


def test_build_projection_view_and_unified_dataset_payload(tmp_path: Path) -> None:
    data_root = tmp_path / "mnist"
    _write_fake_mnist(data_root, split="train", count=24)
    workspace = tmp_path / "workspace"
    build_dataset_variant(
        DatasetVariantConfig(
            family="mnist",
            variant="small",
            split="train",
            input={"dataset_root": str(data_root), "split": "train"},
            selection={"per_class_counts": {"0": 2, "1": 2, "2": 2}},
        ),
        workspace=workspace,
    )
    config_path = tmp_path / "view.yaml"
    config_path.write_text(
        """
explorer_name: small_view
input:
  type: numpy
  data_path: unused.npy
features:
  mode: raw
  name: raw_pixels
  skip_existing: false
projection:
  variants:
    - name: PCA 3D
      key: pca_3d
      method: pca
      n_components: 3
diagnostics:
  enabled: false
output:
  root_dir: unused
explorer:
  renderer: three3d
  compute_projection_diagnostics: false
id_estimation:
  enabled: false
""",
        encoding="utf-8",
    )

    result = build_projection_view(
        variant_id="mnist/small",
        config_path=config_path,
        workspace=workspace,
    )
    registry = GeometryRegistry(workspace)
    indexed = registry.projection_payload(result["view_id"])
    assert indexed is not None
    assert len(indexed["points"]) == 6
    assert registry.projection_label_counts(result["view_id"]) == {
        "0": 2,
        "1": 2,
        "2": 2,
    }
    Path(result["explorer_data"]).unlink()
    payload = load_projection_payload(result["view_id"], workspace=workspace)
    html = build_geometry_html(payload, three_source="window.THREE = {};")

    assert payload["mode"] == "dataset"
    assert payload["projectionDimensions"] == {"PCA 3D": 3}
    assert len(payload["points"]) == 6
    assert payload["atlasSize"] >= payload["tileSize"]
    assert "THREE.PerspectiveCamera" in html
    assert "texture2D(textureAtlas" in html
    assert "Diagnostics · ${projection}" in html
    assert "ResizeObserver" in html
    assert "overflow-y: auto" in html
    assert 'id="diagnostics-dock"' in html
    assert 'id="sidebar-splitter"' in html
    assert 'id="dock-splitter"' in html
    assert "--sidebar-width: 320px" in html
    assert "grid-template-rows: minmax(0, 1fr) 7px var(--dock-height)" in html
    assert "startLayoutDrag" in html
    assert ".legend-item" in html
    assert "addHoverAtlasThumbnail" in html
    assert "appendGeometryDiagnostics" in html
    assert "Core estimates" in html
    assert "Aggregate local estimators" in html
    assert "Selected sample" in html
    assert "More geometry diagnostics" in html
    assert "Representative model estimates" in html
    assert "coloredPointCloud" in html
    assert "colorizeGrayscaleThumbnail" in html
    assert "usesLabelTintedThumbnail(point)" in html
    assert 'id="show-thumbnails"' in html
    assert 'id="point-display-mode"' in html
    assert '<option value="class_color">Class color</option>' in html
    assert 'pointDisplayMode.addEventListener("change"' in html
    assert 'id="class-filter"' in html


def test_build_projection_view_registers_id_merged_explorer_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "mnist"
    _write_fake_mnist(data_root, split="train", count=24)
    workspace = tmp_path / "workspace"
    build_dataset_variant(
        DatasetVariantConfig(
            family="mnist",
            variant="small_id",
            split="train",
            input={"dataset_root": str(data_root), "split": "train"},
            selection={"per_class_counts": {"0": 2, "1": 2, "2": 2}},
        ),
        workspace=workspace,
    )
    config_path = tmp_path / "view_id.yaml"
    config_path.write_text(
        """
explorer_name: small_id_view
input:
  type: numpy
  data_path: unused.npy
features:
  mode: raw
  name: raw_pixels
  skip_existing: false
projection:
  variants:
    - name: PCA 3D
      key: pca_3d
      method: pca
      n_components: 3
diagnostics:
  enabled: false
output:
  root_dir: unused
  save_features: true
explorer:
  renderer: three3d
  compute_projection_diagnostics: false
id_estimation:
  enabled: true
""",
        encoding="utf-8",
    )

    def fake_run_id_estimation(config, *, project_root=None):
        del project_root
        explorer_path = Path(config.input.diagnostics_dir) / config.input.explorer_data_path
        frame = pd.read_parquet(explorer_path)
        frame["mle_lid_k15"] = np.linspace(1.0, 2.0, len(frame))
        merged_path = explorer_path.with_name("explorer_data_with_raw_pixels_id.parquet")
        frame.to_parquet(merged_path, index=False)
        group_path = (
            config.output_dir
            / "intrinsic_dimension"
            / "group_id_raw_pixels_pca50.csv"
        )
        group_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "groupby_column": "__all__",
                    "group_value": "__all__",
                    "n_samples": 6,
                    "feature_space": "raw_pixels_pca50",
                    "global_mle_lid_k20": 4.5,
                    "global_two_nn_lid": 5.5,
                    "global_pca_dim_95": 3,
                    "mean_local_mle_lid_k15": 1.6,
                    "median_local_mle_lid_k15": 1.5,
                    "custom_numeric_diagnostic": 9.0,
                },
                {
                    "groupby_column": "label",
                    "group_value": "0",
                    "n_samples": 2,
                    "feature_space": "raw_pixels_pca50",
                    "global_mle_lid_k20": 3.2,
                    "global_two_nn_lid": 4.2,
                    "global_pca_dim_95": 2,
                    "mean_local_mle_lid_k15": 1.2,
                    "median_local_mle_lid_k15": 1.1,
                    "custom_numeric_diagnostic": 8.0,
                },
            ]
        ).to_csv(group_path, index=False)
        model_group_path = (
            config.output_dir.parent
            / "model_diagnostics_run"
            / "intrinsic_dimension"
            / "group_id_model_diagnostics_run.csv"
        )
        model_group_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "groupby_column": "__all__",
                    "group_value": "__all__",
                    "n_samples": 4,
                    "feature_space": "model_diagnostics:run",
                    "mean_fm_flipd_lid_t0800": 7.0,
                    "median_fm_flipd_lid_t0800": 6.5,
                },
                {
                    "groupby_column": "label",
                    "group_value": "0",
                    "n_samples": 2,
                    "feature_space": "model_diagnostics:run",
                    "mean_fm_flipd_lid_t0800": 3.0,
                    "median_fm_flipd_lid_t0800": 2.5,
                },
            ]
        ).to_csv(model_group_path, index=False)
        return {"merged_explorer_path": str(merged_path), "group_id_path": str(group_path)}

    monkeypatch.setattr(
        "fm_lab.geometry_explorer.views.run_id_estimation",
        fake_run_id_estimation,
    )

    result = build_projection_view(
        variant_id="mnist/small_id",
        config_path=config_path,
        workspace=workspace,
    )
    payload = load_projection_payload(result["view_id"], workspace=workspace)
    html = build_geometry_html(payload, three_source="window.THREE = {};")

    assert Path(result["explorer_data"]).name == "explorer_data_with_raw_pixels_id.parquet"
    assert "mle_lid_k15" in payload["points"][0]["details"]
    assert payload["metricLabels"]["mle_lid_k15"] == "MLE intrinsic dimension (k=15)"
    assert payload["groupDiagnostics"]["overall"]["global_mle_lid_k20"] == 4.5
    assert payload["groupDiagnostics"]["groups"]["0"]["global_mle_lid_k20"] == 3.2
    assert payload["groupDiagnostics"]["groups"]["0"]["class_share"] == 2 / 6
    assert "global_two_nn_lid" in payload["groupDiagnostics"]["metrics"]
    assert "custom_numeric_diagnostic" in payload["groupDiagnostics"]["metrics"]
    assert "mean_fm_flipd_lid_t0800" in payload["groupDiagnostics"]["metrics"]
    assert "mean_fm_flipd_lid_t0800" in payload["groupDiagnostics"]["modelMetrics"]
    assert payload["groupDiagnostics"]["overall"]["mean_fm_flipd_lid_t0800"] == 7.0
    assert payload["groupDiagnostics"]["groups"]["0"]["mean_fm_flipd_lid_t0800"] == 3.0
    assert (
        payload["metricLabels"]["global_mle_lid_k20"]
        == "Global MLE intrinsic dimension (k=20)"
    )
    assert (
        payload["metricLabels"]["mean_fm_flipd_lid_t0800"]
        == "Mean FM-FLIPD raw ID estimate (t=0.800)"
    )
    assert "showGroupDiagnostics" in html
    assert "Class ID ·" in html
    assert "Model ID ·" in html
    assert "Global ID · All classes" in html
    assert "Class comparison ·" in html
    assert "formatMetricDisplay" in html


def test_unified_trajectory_payload_and_html(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    dataset_path = tmp_path / "dataset.parquet"
    write_parquet(pd.DataFrame({"row_id": [0, 1], "label": ["0", "1"]}), dataset_path)
    registry.register_dataset_variant(
        variant_id="mnist/small",
        family="mnist",
        variant="small",
        base="original",
        split="train",
        dataset_path=dataset_path,
        data_path=None,
        labels_path=None,
        config_path=None,
        row_count=2,
        label_counts={"0": 1, "1": 1},
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
    )
    run_dir = tmp_path / "run"
    (run_dir / "samples").mkdir(parents=True)
    (run_dir / "trajectories").mkdir()
    target = np.zeros((2, 784), dtype=np.float32)
    generated = np.ones((2, 784), dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.int64)
    raw_trajectory = np.stack([target, generated], axis=0)
    np.save(run_dir / "samples" / "target_reference.npy", target)
    np.save(run_dir / "samples" / "target_reference_labels.npy", labels)
    np.save(run_dir / "samples" / "euler_nfe1.npy", generated)
    np.save(run_dir / "trajectories" / "euler_nfe1.npy", raw_trajectory)
    coordinates_path = tmp_path / "coords.npz"
    np.savez(
        coordinates_path,
        target=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        generated=np.asarray([[0, 1, 0], [1, 1, 0]], dtype=np.float32),
        trajectory=np.asarray(
            [
                [[0, 0, 0], [1, 0, 0]],
                [[0, 1, 0], [1, 1, 0]],
            ],
            dtype=np.float32,
        ),
    )
    view_output = tmp_path / "view_output"
    registry.register_projection_view(
        view_id="view",
        variant_id="mnist/small",
        feature_name="raw_pixels",
        feature_mode="raw",
        explorer_data_path=dataset_path,
        output_dir=view_output,
        projection_names={"pca_3d": "PCA 3D"},
        renderer="three3d",
        row_count=2,
    )
    trajectory_group_path = (
        view_output
        / "id_estimation"
        / "model_diagnostics_run"
        / "intrinsic_dimension"
        / "group_id_model_diagnostics_run.csv"
    )
    trajectory_group_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "groupby_column": "__all__",
                "group_value": "__all__",
                "n_samples": 2,
                "feature_space": "model_diagnostics:run",
                "mean_fm_flipd_lid_t0800": 4.0,
            },
            {
                "groupby_column": "label",
                "group_value": "1",
                "n_samples": 1,
                "feature_space": "model_diagnostics:run",
                "mean_fm_flipd_lid_t0800": 5.0,
            },
        ]
    ).to_csv(trajectory_group_path, index=False)
    registry.register_model_run(
        run_id="run",
        run_dir=run_dir,
        variant_id="mnist/small",
        family="mnist",
        variant="small",
        config_path=None,
        metrics_path=None,
    )
    registry.register_trajectory_view(
        view_id="traj",
        run_id="run",
        variant_id="mnist/small",
        solver="euler",
        nfe=1,
        coordinates_path=coordinates_path,
        trajectory_path=run_dir / "trajectories" / "euler_nfe1.npy",
        generated_path=run_dir / "samples" / "euler_nfe1.npy",
        target_path=run_dir / "samples" / "target_reference.npy",
        labels_path=run_dir / "samples" / "target_reference_labels.npy",
        output_dir=tmp_path / "trajectory_output",
        interactive_path=None,
        n_steps=2,
        n_trajectories=2,
    )

    payload = load_trajectory_payload("traj", workspace=workspace)
    html = build_geometry_html(payload, three_source="window.THREE = {};")

    assert payload["mode"] == "trajectory"
    assert payload["counts"]["trajectories"] == 2
    assert len(payload["trajectoryPreviews"]) == 2
    assert payload["atlasSize"] >= payload["tileSize"]
    assert payload["groupDiagnostics"]["overall"]["mean_fm_flipd_lid_t0800"] == 4.0
    assert "mean_fm_flipd_lid_t0800" in payload["groupDiagnostics"]["modelMetrics"]
    assert "texture2D(textureAtlas" in html
    assert 'id="time"' in html
    assert 'id="show-trajectory"' in html


def test_geometry_viewer_defaults_to_umap_projection() -> None:
    payload = {
        "mode": "dataset",
        "points": [],
        "trajectory": [],
        "trajectoryLabels": [],
        "trajectoryPreviews": [],
        "atlases": [],
        "palette": {},
        "projections": ["PCA 3D", "UMAP 3D (k=15)"],
        "projectionDimensions": {"PCA 3D": 3, "UMAP 3D (k=15)": 3},
        "projectionDiagnostics": {},
        "groupDiagnostics": {},
        "metricLabels": {},
        "tileSize": 28,
        "atlasSize": 2048,
        "atlasColumns": 73,
        "options": {"drawThumbnailsDefault": True},
        "counts": {"points": 0, "trajectorySteps": 0, "trajectories": 0},
    }

    html = build_geometry_html(payload, three_source="window.THREE = {};")

    assert "defaultProjectionName" in html
    assert 'includes("umap")' in html
    assert "projectionSelect.value = projection" in html


def test_explorer_cli_launch_dry_run(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "fm-lab-explorer",
            "--workspace",
            str(tmp_path / "workspace"),
            "launch",
            "--dry-run",
        ],
    )

    explorer_cli_main()

    output = capsys.readouterr().out
    assert "streamlit run" in output
    assert "geometry_explorer_app.py" in output


def test_explorer_cli_summarize_group_diagnostics(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    dataset_path = tmp_path / "dataset.parquet"
    write_parquet(pd.DataFrame({"row_id": [0, 1], "label": ["0", "1"]}), dataset_path)
    output_dir = tmp_path / "view_output"
    group_path = (
        output_dir
        / "id_estimation"
        / "raw_geometry_view_raw_pixels_id"
        / "intrinsic_dimension"
        / "group_id_raw_pixels_pca50.csv"
    )
    group_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "groupby_column": "__all__",
                "group_value": "__all__",
                "n_samples": 2,
                "feature_space": "raw_pixels_pca50",
                "global_mle_lid_k20": 4.25,
            },
            {
                "groupby_column": "label",
                "group_value": "0",
                "n_samples": 1,
                "feature_space": "raw_pixels_pca50",
                "global_mle_lid_k20": 3.25,
            },
        ]
    ).to_csv(group_path, index=False)
    registry.register_dataset_variant(
        variant_id="mnist/small",
        family="mnist",
        variant="small",
        base="original",
        split="train",
        dataset_path=dataset_path,
        data_path=None,
        labels_path=None,
        config_path=None,
        row_count=2,
        label_counts={"0": 1, "1": 1},
        image_shape=(28, 28),
        value_range=(0.0, 1.0),
    )
    registry.register_projection_view(
        view_id="mnist__small__raw_pixels__raw_geometry_view",
        variant_id="mnist/small",
        feature_name="raw_pixels",
        feature_mode="raw",
        explorer_data_path=dataset_path,
        output_dir=output_dir,
        projection_names={"umap_3d": "UMAP 3D"},
        renderer="three3d",
        row_count=2,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "fm-lab-explorer",
            "--workspace",
            str(workspace),
            "summarize",
            "--dataset",
            "mnist/small",
            "--include-classes",
        ],
    )

    explorer_cli_main()

    output = capsys.readouterr().out
    assert "mnist/small | raw_pixels" in output
    assert "Global MLE intrinsic dimension (k=20): 4.250" in output
    assert "0: Global MLE intrinsic dimension (k=20)=3.250" in output
    assert "share=50.0%" in output


def test_explorer_cli_build_all_dry_run_discovers_dataset_configs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config_dir = tmp_path / "configs"
    dataset_config = config_dir / "datasets" / "mnist" / "original" / "dataset.yaml"
    view_config = config_dir / "views" / "raw_pixels.yaml"
    dataset_config.parent.mkdir(parents=True)
    view_config.parent.mkdir(parents=True)
    dataset_config.write_text(
        """
family: mnist
variant: original
input:
  dataset_root: data/mnist
""",
        encoding="utf-8",
    )
    view_config.write_text(
        """
explorer_name: raw_geometry_view
features:
  name: raw_pixels
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "fm-lab-explorer",
            "--workspace",
            str(tmp_path / "workspace"),
            "build-all",
            "--config-dir",
            str(config_dir / "datasets"),
            "--view-config",
            str(view_config),
            "--dry-run",
        ],
    )

    explorer_cli_main()

    output = capsys.readouterr().out
    assert "Geometry explorer build plan" in output
    assert "mnist/original" in output
    assert "raw_pixels.yaml" in output


def test_explorer_cli_build_all_runs_dataset_then_view(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_dir = tmp_path / "configs"
    dataset_config = config_dir / "datasets" / "mnist" / "original" / "dataset.yaml"
    view_config = config_dir / "views" / "raw_pixels.yaml"
    dataset_config.parent.mkdir(parents=True)
    view_config.parent.mkdir(parents=True)
    dataset_config.write_text(
        """
family: mnist
variant: original
input:
  dataset_root: data/mnist
""",
        encoding="utf-8",
    )
    view_config.write_text(
        """
explorer_name: raw_geometry_view
features:
  name: raw_pixels
""",
        encoding="utf-8",
    )
    calls = []

    def fake_build_dataset_variant(config, **kwargs):
        calls.append(("dataset", f"{config.family}/{config.variant}", kwargs))
        return {
            "variant_id": f"{config.family}/{config.variant}",
            "dataset_path": tmp_path / "dataset.parquet",
        }

    def fake_build_projection_view(**kwargs):
        calls.append(("view", kwargs["variant_id"], kwargs))
        return {
            "view_id": "mnist__original__raw_pixels__raw_geometry_view",
            "explorer_data": tmp_path / "explorer.parquet",
        }

    monkeypatch.setattr(
        "fm_lab.experiments.run_explorer.build_dataset_variant",
        fake_build_dataset_variant,
    )
    monkeypatch.setattr(
        "fm_lab.experiments.run_explorer.build_projection_view",
        fake_build_projection_view,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "fm-lab-explorer",
            "--workspace",
            str(tmp_path / "workspace"),
            "build-all",
            "--config-dir",
            str(config_dir / "datasets"),
            "--view-config",
            str(view_config),
            "--dataset",
            "mnist/original",
        ],
    )

    explorer_cli_main()

    assert [call[:2] for call in calls] == [
        ("dataset", "mnist/original"),
        ("view", "mnist/original"),
    ]
    assert calls[1][2]["config_path"] == view_config


def test_explorer_cli_build_registered_views_dry_run_uses_registry(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    dataset_path = tmp_path / "dataset.parquet"
    write_parquet(pd.DataFrame({"row_id": [0], "label": ["0"]}), dataset_path)
    for variant_id, family, variant in (
        ("mnist/original", "mnist", "original"),
        ("mnist/generated", "mnist", "generated"),
    ):
        registry.register_dataset_variant(
            variant_id=variant_id,
            family=family,
            variant=variant,
            base="generated" if variant == "generated" else "original",
            split="all",
            dataset_path=dataset_path,
            data_path=None,
            labels_path=None,
            config_path=None,
            row_count=1,
            label_counts={"0": 1},
            image_shape=[28, 28],
            value_range=[0.0, 1.0],
        )
    monkeypatch.setattr(
        "sys.argv",
        [
            "fm-lab-explorer",
            "--workspace",
            str(workspace),
            "build-registered-views",
            "--dry-run",
        ],
    )

    explorer_cli_main()

    output = capsys.readouterr().out
    assert "Registered dataset view build plan" in output
    assert "mnist/original (1 rows)" in output
    assert "mnist/generated (1 rows)" in output


def test_explorer_cli_build_registered_views_runs_for_selected_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    registry = GeometryRegistry(workspace)
    dataset_path = tmp_path / "dataset.parquet"
    view_config = tmp_path / "raw_pixels.yaml"
    write_parquet(pd.DataFrame({"row_id": [0], "label": ["0"]}), dataset_path)
    view_config.write_text("explorer_name: raw_geometry_view\n", encoding="utf-8")
    for variant_id, variant in (
        ("mnist/original", "original"),
        ("mnist/generated", "generated"),
    ):
        registry.register_dataset_variant(
            variant_id=variant_id,
            family="mnist",
            variant=variant,
            base=variant,
            split="all",
            dataset_path=dataset_path,
            data_path=None,
            labels_path=None,
            config_path=None,
            row_count=1,
            label_counts={"0": 1},
            image_shape=[28, 28],
            value_range=[0.0, 1.0],
        )
    calls = []

    def fake_build_projection_view(**kwargs):
        calls.append(kwargs)
        return {
            "view_id": f"{kwargs['variant_id'].replace('/', '__')}__view",
            "explorer_data": tmp_path / "explorer.parquet",
        }

    monkeypatch.setattr(
        "fm_lab.experiments.run_explorer.build_projection_view",
        fake_build_projection_view,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "fm-lab-explorer",
            "--workspace",
            str(workspace),
            "build-registered-views",
            "--view-config",
            str(view_config),
            "--dataset",
            "mnist/generated",
        ],
    )

    explorer_cli_main()

    assert [call["variant_id"] for call in calls] == ["mnist/generated"]
    assert calls[0]["config_path"] == view_config


def _write_fake_mnist(root: Path, *, split: str, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    prefix = "train" if split == "train" else "t10k"
    _write_fake_idx_dataset(root, split=split, count=count, prefix=prefix)


def _write_fake_idx_dataset(
    root: Path,
    *,
    split: str,
    count: int,
    prefix: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    prefix = prefix or ("train" if split == "train" else "t10k")
    images_path = root / f"{prefix}-images-idx3-ubyte.gz"
    labels_path = root / f"{prefix}-labels-idx1-ubyte.gz"
    images = np.arange(count * 28 * 28, dtype=np.uint8)
    labels = np.arange(count, dtype=np.uint8) % 10
    with gzip.open(images_path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, count, 28, 28))
        handle.write(images.tobytes())
    with gzip.open(labels_path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, count))
        handle.write(labels.tobytes())


def _write_fake_cifar10(root: Path, *, count: int) -> None:
    data_dir = root / "cifar-10-batches-bin"
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = np.arange(count, dtype=np.uint8) % 10
    images = np.arange(count * 32 * 32 * 3, dtype=np.uint8).reshape(count, -1)
    records = np.column_stack([labels, images]).astype(np.uint8)
    for index in range(1, 6):
        batch = records if index == 1 else np.empty((0, 3073), dtype=np.uint8)
        batch.tofile(data_dir / f"data_batch_{index}.bin")


def _write_fake_cifar100(root: Path, *, count: int) -> None:
    data_dir = root / "cifar-100-binary"
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = np.arange(count, dtype=np.uint8) % 2
    coarse_labels = labels // 5
    images = np.arange(count * 32 * 32 * 3, dtype=np.uint8).reshape(count, -1)
    records = np.column_stack([coarse_labels, labels, images]).astype(np.uint8)
    records.tofile(data_dir / "train.bin")
    np.empty((0, 3074), dtype=np.uint8).tofile(data_dir / "test.bin")
    names = ["apple", "aquarium_fish", *(f"class_{index}" for index in range(2, 100))]
    (data_dir / "fine_label_names.txt").write_text("\n".join(names), encoding="utf-8")
    (data_dir / "coarse_label_names.txt").write_text(
        "\n".join(f"coarse_{index}" for index in range(20)),
        encoding="utf-8",
    )


def _write_fake_cinic10_tar(root: Path, *, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    labels = ("airplane", "automobile")
    archive_path = root / "CINIC-10.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for index in range(count):
            label = labels[index % len(labels)]
            image = np.full((32, 32, 3), index % 256, dtype=np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
            payload = buffer.getvalue()
            info = tarfile.TarInfo(f"train/{label}/{index:05d}.png")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _patch_fashion_mnist_checksums(monkeypatch, root: Path) -> None:
    import hashlib

    import fm_lab.image_diagnostics.dataset_loader as dataset_loader

    replacements = {}
    for path in root.glob("*-idx*-ubyte.gz"):
        replacements[path.name] = hashlib.md5(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        dataset_loader,
        "FASHION_MNIST_FILES",
        dataset_loader.FASHION_MNIST_FILES | replacements,
    )
