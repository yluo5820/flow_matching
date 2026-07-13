from __future__ import annotations

from pathlib import Path

import numpy as np

from fm_lab.geometry_explorer.registry import GeometryRegistry
from fm_lab.geometry_explorer.sampling_timesteps import (
    build_sampling_timestep_dataset_from_trajectory,
    select_timestep_classes,
)
from fm_lab.image_diagnostics.save_utils import read_parquet


def test_select_timestep_classes_uses_nearest_unique_grid_points() -> None:
    classes = select_timestep_classes(
        np.linspace(0.0, 1.0, 6),
        num_classes=3,
        time_start=0.2,
        time_stop=0.8,
    )

    assert [item.step_index for item in classes] == [1, 2, 4]
    assert [item.label for item in classes] == [
        "timestep_00_t0p200",
        "timestep_01_t0p400",
        "timestep_02_t0p800",
    ]


def test_build_sampling_timestep_dataset_from_trajectory_registers_variant(
    tmp_path: Path,
) -> None:
    trajectory = np.arange(6 * 4 * 9, dtype=np.float32).reshape(6, 4, 9)
    trajectory_path = tmp_path / "trajectory.npy"
    np.save(trajectory_path, trajectory)
    workspace = tmp_path / "workspace"

    result = build_sampling_timestep_dataset_from_trajectory(
        trajectory_path=trajectory_path,
        variant_id="cifar10/test_sampling_timesteps",
        workspace=workspace,
        num_classes=3,
        total_rows=12,
        time_start=0.2,
        time_stop=0.8,
        image_shape=(3, 3),
        value_range=(0.0, 1.0),
        atlas_tile_size=4,
        atlas_size=16,
    )

    assert result.rows == 12
    assert result.paths_per_class == 4
    data = np.load(result.data_path)
    labels = np.load(result.labels_path)
    assert data.shape == (12, 9)
    assert labels.tolist()[:3] == [
        "timestep_00_t0p200",
        "timestep_01_t0p400",
        "timestep_02_t0p800",
    ]
    assert np.allclose(data[0], trajectory[1, 0])
    assert np.allclose(data[1], trajectory[2, 0])
    assert np.allclose(data[2], trajectory[4, 0])

    metadata = read_parquet(result.dataset_path)
    assert metadata["trajectory_index"].tolist()[:6] == [0, 0, 0, 1, 1, 1]
    assert metadata["timestep_step_index"].tolist()[:3] == [1, 2, 4]
    assert metadata["is_last_selected_timestep"].tolist()[:3] == [False, False, True]
    assert metadata["is_solver_final_timestep"].tolist()[:3] == [False, False, False]
    assert Path(metadata["sprite_atlas_path"].iloc[0]).exists()

    row = dict(GeometryRegistry(workspace).get_dataset_variant(result.variant_id))
    assert row["row_count"] == 12
    assert row["data_path"].endswith("data.npy")
