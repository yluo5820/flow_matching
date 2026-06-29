from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from fm_lab.experiments.factory import build_target
from fm_lab.experiments.run_explorer import main as explorer_cli_main
from fm_lab.geometry_explorer.bundles import (
    load_projection_payload,
    load_trajectory_payload,
)
from fm_lab.geometry_explorer.registry import GeometryRegistry
from fm_lab.geometry_explorer.variants import (
    DatasetVariantConfig,
    build_dataset_variant,
)
from fm_lab.geometry_explorer.viewer import build_geometry_html
from fm_lab.geometry_explorer.views import build_projection_view
from fm_lab.image_diagnostics.save_utils import write_parquet


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
    assert registry.trajectory_views(variant_id="mnist/test")[0].view_id == "traj"


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
    payload = load_projection_payload(result["view_id"], workspace=workspace)
    html = build_geometry_html(payload, three_source="window.THREE = {};")

    assert payload["mode"] == "dataset"
    assert payload["projectionDimensions"] == {"PCA 3D": 3}
    assert len(payload["points"]) == 6
    assert "THREE.PerspectiveCamera" in html
    assert 'id="show-thumbnails"' in html
    assert 'id="class-filter"' in html


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
    assert 'id="time"' in html
    assert 'id="show-trajectory"' in html


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


def _write_fake_mnist(root: Path, *, split: str, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    prefix = "train" if split == "train" else "t10k"
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
