from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fm_lab.image_diagnostics.id_config import id_config_from_dict
from fm_lab.image_diagnostics.id_estimators import (
    compute_global_id,
    compute_local_id,
    compute_neighbor_graph,
)
from fm_lab.image_diagnostics.id_feature_loader import load_id_features
from fm_lab.image_diagnostics.id_runner import run_id_estimation


def test_local_id_recovers_line_and_plane_dimensions() -> None:
    rng = np.random.default_rng(4)
    line_coordinate = rng.uniform(-1, 1, size=500)
    line = np.column_stack(
        [line_coordinate, 2 * line_coordinate, -0.5 * line_coordinate]
    ).astype(np.float32)
    plane_uv = rng.uniform(-1, 1, size=(500, 2))
    plane = np.column_stack(
        [plane_uv[:, 0], plane_uv[:, 1], plane_uv.sum(axis=1)]
    ).astype(np.float32)
    metadata = pd.DataFrame({"row_id": np.arange(500)})
    config = _id_config("unused", "unused.npy", metric="euclidean")

    line_id = compute_local_id(line, metadata, config, feature_space="line")
    plane_id = compute_local_id(plane, metadata, config, feature_space="plane")

    assert np.nanmedian(line_id["participation_ratio_k15"]) == pytest_approx(
        1.0, abs=0.05
    )
    assert np.nanmedian(line_id["pca_dim_95_k15"]) == 1.0
    assert 0.7 < np.nanmedian(line_id["ball_scaling_dim_k15"]) < 1.4
    assert 1.4 < np.nanmedian(plane_id["participation_ratio_k15"]) <= 2.1
    assert np.nanmedian(plane_id["pca_dim_95_k15"]) == 2.0


def test_duplicate_points_produce_nan_instead_of_crashing() -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    metadata = pd.DataFrame({"row_id": range(4)})
    config = _id_config("unused", "unused.npy", metric="euclidean")

    graph = compute_neighbor_graph(features, max_neighbors=3, metric="euclidean")
    local = compute_local_id(features, metadata, config, feature_space="duplicates")
    global_result, scaling = compute_global_id(
        features,
        config,
        feature_space="duplicates",
    )

    assert graph.distances.shape == (4, 3)
    assert local["two_nn_lid_local"].isna().any()
    assert "global_two_nn_lid" in global_result
    assert scaling is not None


def test_feature_loader_aligns_embedding_metadata_to_explorer(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    explorer_path = diagnostics / "explorer" / "explorer_data.parquet"
    explorer_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "row_id": [10, 20, 30],
            "label": ["a", "b", "c"],
            "image_path": ["a.png", "b.png", "c.png"],
        }
    ).to_parquet(explorer_path, index=False)
    features = np.asarray([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    feature_path = diagnostics / "features" / "test_features.npy"
    feature_path.parent.mkdir()
    np.save(feature_path, features)
    pd.DataFrame({"row_id": [30, 10, 20]}).to_parquet(
        diagnostics / "features" / "test_metadata.parquet",
        index=False,
    )
    config = id_config_from_dict(
        {
            "id_estimation_name": "align",
            "input": {
                "diagnostics_dir": str(diagnostics),
                "embedding_source": "features/test_features.npy",
                "embedding_metadata": "features/test_metadata.parquet",
                "feature_space_name": "test",
            },
            "features": {"normalize": False},
        }
    )

    bundle = load_id_features(config, project_root=tmp_path)

    assert bundle.metadata["row_id"].tolist() == [30, 10, 20]
    assert bundle.metadata["label"].tolist() == ["c", "a", "b"]
    assert np.array_equal(bundle.features, features)


def test_raw_pixel_loader_and_pca_preprocess(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    explorer_path = diagnostics / "explorer" / "explorer_data.parquet"
    explorer_path.parent.mkdir(parents=True)
    image_paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.fromarray(
            np.full((6, 5), index * 50, dtype=np.uint8),
            mode="L",
        ).save(path)
        image_paths.append(str(path))
    pd.DataFrame(
        {
            "row_id": range(4),
            "image_path": image_paths,
        }
    ).to_parquet(explorer_path, index=False)
    config = id_config_from_dict(
        {
            "id_estimation_name": "raw",
            "input": {
                "diagnostics_dir": str(diagnostics),
                "source_type": "raw_pixels",
                "feature_space_name": "raw",
                "raw_grayscale": True,
            },
            "features": {
                "normalize": False,
                "pca_preprocess": {"enabled": True, "n_components": 2},
            },
        }
    )

    bundle = load_id_features(config, project_root=tmp_path)

    assert bundle.features.shape == (4, 2)
    assert bundle.feature_space == "raw_pca2"


def test_id_runner_saves_groups_curves_and_merged_explorer(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    explorer_path = diagnostics / "explorer" / "explorer_data.parquet"
    explorer_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(7)
    features = rng.normal(size=(80, 6)).astype(np.float32)
    metadata = pd.DataFrame(
        {
            "row_id": np.arange(80),
            "family": ["left"] * 40 + ["right"] * 40,
            "manual_label": ["unlabeled"] * 80,
        }
    )
    metadata.to_parquet(explorer_path, index=False)
    feature_path = diagnostics / "features" / "test_features.npy"
    feature_path.parent.mkdir()
    np.save(feature_path, features)
    metadata[["row_id"]].to_parquet(
        diagnostics / "features" / "test_metadata.parquet",
        index=False,
    )
    config = id_config_from_dict(
        {
            "id_estimation_name": "runner",
            "input": {
                "diagnostics_dir": str(diagnostics),
                "embedding_source": "features/test_features.npy",
                "embedding_metadata": "features/test_metadata.parquet",
                "feature_space_name": "test",
            },
            "features": {"normalize": False},
            "groups": {"groupby_columns": ["family", "missing_column"]},
            "local_id": {
                "k_values": [5, 10],
                "covariance_eigenvalues": 4,
            },
            "global_id": {
                "min_group_size": 20,
                "mle_k_values": [5, 10],
                "scaling_max_points": 80,
            },
            "output": {
                "root_dir": str(tmp_path / "outputs"),
                "merge_into_explorer_data": True,
            },
        }
    )

    result = run_id_estimation(config, project_root=tmp_path)

    local = pd.read_parquet(result["local_id_path"])
    groups = pd.read_csv(result["group_id_path"])
    merged = pd.read_parquet(result["merged_explorer_path"])
    curve_paths = list(
        (tmp_path / "outputs" / "runner" / "intrinsic_dimension" / "id_curves").glob(
            "*.csv"
        )
    )
    assert len(local) == 80
    assert set(groups["groupby_column"]) == {"__all__", "family"}
    assert "mle_lid_k10" in merged
    assert merged["id_feature_space"].eq("test").all()
    assert curve_paths
    assert explorer_path.exists()
    assert Path(result["merged_explorer_path"]).name == "explorer_data_with_id.parquet"


def _id_config(
    diagnostics_dir: str,
    embedding_source: str,
    *,
    metric: str,
):
    return id_config_from_dict(
        {
            "id_estimation_name": "test",
            "input": {
                "diagnostics_dir": diagnostics_dir,
                "embedding_source": embedding_source,
                "feature_space_name": "test",
            },
            "features": {"normalize": False},
            "local_id": {
                "k_values": [5, 15],
                "covariance_eigenvalues": 4,
            },
            "global_id": {
                "mle_k_values": [5, 15],
                "min_group_size": 5,
                "scaling_max_points": 500,
            },
            "distance": {"metric": metric},
            "output": {"merge_into_explorer_data": False},
        }
    )


def pytest_approx(value: float, *, abs: float):
    import pytest

    return pytest.approx(value, abs=abs)
