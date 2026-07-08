from __future__ import annotations

from pathlib import Path

import numpy as np

from fm_lab.geometry_explorer.registry import GeometryRegistry
from fm_lab.geometry_explorer.synthetic_objects import (
    SyntheticObjectSpec,
    SyntheticRenderConfig,
    build_synthetic_object_dataset,
    render_marked_cube,
    synthetic_object_config_from_dict,
)
from fm_lab.geometry_explorer.variants import DatasetVariantConfig, build_dataset_variant
from fm_lab.image_diagnostics.explorer_payload import sample_metric_columns
from fm_lab.image_diagnostics.save_utils import read_parquet


def test_render_marked_cube_is_deterministic_and_pose_dependent() -> None:
    render = SyntheticRenderConfig(
        image_size=32,
        azimuth_steps=3,
        supersample=1,
        focal_length=36,
    )

    first = render_marked_cube(render=render, azimuth_deg=0.0)
    second = render_marked_cube(render=render, azimuth_deg=0.0)
    rotated = render_marked_cube(render=render, azimuth_deg=60.0)

    assert first.shape == (32, 32, 3)
    assert np.allclose(first, second)
    assert not np.allclose(first, rotated)
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0


def test_asymmetric_object_kinds_render_distinct_images() -> None:
    render = SyntheticRenderConfig(
        image_size=32,
        azimuth_steps=3,
        supersample=1,
        focal_length=36,
    )

    statue = render_marked_cube(
        object_spec=SyntheticObjectSpec(kind="abstract_statue", marker=False),
        render=render,
        azimuth_deg=30.0,
    )
    monument = render_marked_cube(
        object_spec=SyntheticObjectSpec(kind="offset_monument", marker=False),
        render=render,
        azimuth_deg=30.0,
    )
    statue_again = render_marked_cube(
        object_spec=SyntheticObjectSpec(kind="abstract_statue", marker=False),
        render=render,
        azimuth_deg=30.0,
    )

    assert np.allclose(statue, statue_again)
    assert not np.allclose(statue, monument)
    assert statue.shape == (32, 32, 3)
    assert monument.shape == (32, 32, 3)


def test_build_synthetic_object_dataset_registers_variant(tmp_path: Path) -> None:
    config = synthetic_object_config_from_dict(
        {
            "family": "synthetic_object",
            "variant": "tiny_pose",
            "base": "analytic",
            "split": "pose_sweep",
            "object": {"kind": "marked_cube"},
            "render": {
                "image_size": 32,
                "azimuth_start": 0,
                "azimuth_stop": 40,
                "azimuth_steps": 5,
                "azimuth_bins": 5,
                "supersample": 1,
                "focal_length": 36,
            },
            "output": {"save_pngs": True},
        }
    )

    result = build_synthetic_object_dataset(config, workspace=tmp_path / "workspace")

    data = np.load(result["data_path"])
    labels = np.load(result["labels_path"])
    metadata = read_parquet(result["dataset_path"])
    registered = GeometryRegistry(tmp_path / "workspace").dataset_variants()

    assert result["variant_id"] == "synthetic_object/tiny_pose"
    assert data.shape == (5, 32 * 32 * 3)
    assert labels.tolist() == [0, 1, 2, 3, 4]
    assert metadata["azimuth_deg"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert metadata["image_path"].map(lambda value: Path(value).is_file()).all()
    assert registered[0].variant_id == "synthetic_object/tiny_pose"
    assert registered[0].row_count == 5
    assert "azimuth_deg" in sample_metric_columns(metadata)
    assert "light_energy" in sample_metric_columns(metadata)


def test_build_synthetic_object_dataset_supports_so3_sampling(tmp_path: Path) -> None:
    config = synthetic_object_config_from_dict(
        {
            "family": "synthetic_object",
            "variant": "tiny_so3",
            "base": "analytic",
            "split": "so3",
            "seed": 11,
            "object": {"kind": "abstract_statue", "marker": False},
            "render": {
                "image_size": 24,
                "supersample": 1,
                "focal_length": 28,
            },
            "pose": {
                "mode": "so3",
                "samples": 7,
                "orientation_bins": 4,
            },
            "output": {"save_pngs": False},
        }
    )

    result = build_synthetic_object_dataset(config, workspace=tmp_path / "workspace")

    data = np.load(result["data_path"])
    metadata = read_parquet(result["dataset_path"])

    assert data.shape == (7, 24 * 24 * 3)
    assert metadata["pose_mode"].tolist() == ["so3"] * 7
    assert metadata["object_kind"].tolist() == ["abstract_statue"] * 7
    assert set(metadata["label_id"]) <= {0, 1, 2, 3}
    assert "rotation_angle_deg" in metadata
    assert "rotation_r00" in metadata
    assert "quat_w" in metadata
    assert "rotation_angle_deg" in sample_metric_columns(metadata)
    assert "quat_w" in sample_metric_columns(metadata)


def test_build_synthetic_object_dataset_supports_sphere_camera_sampling(
    tmp_path: Path,
) -> None:
    config = synthetic_object_config_from_dict(
        {
            "family": "synthetic_object",
            "variant": "tiny_sphere",
            "base": "analytic",
            "split": "sphere",
            "seed": 13,
            "object": {"kind": "offset_monument", "marker": False},
            "render": {
                "image_size": 24,
                "supersample": 1,
                "focal_length": 28,
            },
            "pose": {
                "mode": "sphere",
                "samples": 9,
                "orientation_bins": 3,
            },
            "output": {"save_pngs": False},
        }
    )

    result = build_synthetic_object_dataset(config, workspace=tmp_path / "workspace")

    data = np.load(result["data_path"])
    metadata = read_parquet(result["dataset_path"])
    directions = metadata[
        ["camera_direction_x", "camera_direction_y", "camera_direction_z"]
    ].to_numpy(dtype=np.float32)

    assert data.shape == (9, 24 * 24 * 3)
    assert metadata["pose_mode"].tolist() == ["sphere"] * 9
    assert metadata["object_kind"].tolist() == ["offset_monument"] * 9
    assert set(metadata["label_id"]) <= {0, 1, 2}
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert "camera_direction_z" in sample_metric_columns(metadata)
    assert "sphere_z_bin" in sample_metric_columns(metadata)


def test_build_synthetic_object_dataset_supports_translation_sampling(
    tmp_path: Path,
) -> None:
    config = synthetic_object_config_from_dict(
        {
            "family": "synthetic_object",
            "variant": "tiny_translation_xy",
            "base": "analytic",
            "split": "translation_xy",
            "seed": 17,
            "object": {"kind": "abstract_statue", "marker": False},
            "render": {
                "image_size": 24,
                "camera_distance": 4,
                "supersample": 1,
                "focal_length": 28,
            },
            "pose": {
                "mode": "translation_xy",
                "samples": 11,
                "translation_bins": 4,
                "translation_x_range": [-0.5, 0.5],
                "translation_y_range": [-0.4, 0.4],
            },
            "output": {"save_pngs": False},
        }
    )

    result = build_synthetic_object_dataset(config, workspace=tmp_path / "workspace")

    data = np.load(result["data_path"])
    metadata = read_parquet(result["dataset_path"])

    assert data.shape == (11, 24 * 24 * 3)
    assert metadata["pose_mode"].tolist() == ["translation_xy"] * 11
    assert metadata["object_kind"].tolist() == ["abstract_statue"] * 11
    assert set(metadata["label_id"]) <= {0, 1, 2, 3}
    assert metadata["camera_depth"].nunique() == 1
    assert metadata["object_center_u"].nunique() > 1
    assert metadata["object_center_v"].nunique() > 1
    assert "translation_x" in sample_metric_columns(metadata)
    assert "object_center_u" in sample_metric_columns(metadata)


def test_build_dataset_variant_dispatches_synthetic_family(tmp_path: Path) -> None:
    config = DatasetVariantConfig(
        family="synthetic_object",
        variant="dispatch_pose",
        base="analytic",
        split="pose_sweep",
        render={
            "image_size": 24,
            "azimuth_start": 0,
            "azimuth_stop": 20,
            "azimuth_steps": 3,
            "azimuth_bins": 3,
            "supersample": 1,
            "focal_length": 28,
        },
    )

    result = build_dataset_variant(config, workspace=tmp_path / "workspace")

    assert result["variant_id"] == "synthetic_object/dispatch_pose"
    assert np.load(result["data_path"]).shape == (3, 24 * 24 * 3)
