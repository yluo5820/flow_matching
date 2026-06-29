import sys
from types import SimpleNamespace

import numpy as np
import torch

from fm_lab.diagnostics.trajectory_umap import (
    TrajectoryUMAPConfig,
    project_saved_trajectories,
)
from fm_lab.plotting.diagnostics import plot_training_history
from fm_lab.plotting.trajectories import (
    plot_generated_samples,
    plot_trajectories,
    plot_umap_projected_trajectories,
)


def test_plot_generated_samples_supports_3d(tmp_path) -> None:
    output_path = tmp_path / "plots" / "generated_3d.png"

    plot_generated_samples(
        target_samples=torch.randn(32, 3),
        generated={"euler": torch.randn(32, 3), "rk4": torch.randn(32, 3)},
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_trajectories_supports_3d(tmp_path) -> None:
    output_path = tmp_path / "plots" / "trajectories_3d.png"
    increments = torch.randn(5, 8, 3) * 0.1
    trajectory = increments.cumsum(dim=0)

    plot_trajectories(
        trajectory=trajectory,
        target_samples=torch.randn(32, 3),
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_generated_samples_supports_image_grid(tmp_path) -> None:
    output_path = tmp_path / "plots" / "generated_images.png"

    plot_generated_samples(
        target_samples=torch.rand(16, 28 * 28),
        generated={"euler": torch.rand(16, 28 * 28)},
        output_path=output_path,
        image_shape=[28, 28],
        max_points=16,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_trajectories_supports_image_snapshots(tmp_path) -> None:
    output_path = tmp_path / "plots" / "trajectory_images.png"
    trajectory = torch.rand(5, 4, 28 * 28)

    plot_trajectories(
        trajectory=trajectory,
        output_path=output_path,
        image_shape=[28, 28],
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_umap_projected_trajectories_supports_high_dimensional_paths(
    tmp_path,
    monkeypatch,
) -> None:
    observed = {}

    class FakeUMAP:
        def __init__(self, **kwargs) -> None:
            observed["kwargs"] = kwargs

        def fit_transform(self, values: np.ndarray) -> np.ndarray:
            observed["input_shape"] = values.shape
            return values[:, :3]

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    output_path = tmp_path / "plots" / "trajectory_umap.png"
    coordinates_path = tmp_path / "trajectories" / "trajectory_umap.npz"
    interactive_path = tmp_path / "plots" / "trajectory_umap.html"

    result = plot_umap_projected_trajectories(
        trajectory=torch.randn(5, 4, 8),
        target_samples=torch.randn(10, 8),
        output_path=output_path,
        max_target_points=6,
        n_neighbors=5,
        coordinates_path=coordinates_path,
        interactive_path=interactive_path,
    )

    assert output_path.exists()
    assert coordinates_path.exists()
    assert interactive_path.exists()
    html = interactive_path.read_text(encoding="utf-8")
    assert 'id="time"' in html
    assert "cumulative paths" in html
    assert result["interactive_path"] == str(interactive_path)
    assert observed["input_shape"] == (5 * 4 + 6, 8)
    assert observed["kwargs"]["n_components"] == 3
    assert observed["kwargs"]["n_neighbors"] == 5
    coordinates = np.load(coordinates_path)
    assert coordinates["trajectory"].shape == (5, 4, 3)
    assert coordinates["target"].shape == (6, 3)
    assert result["trajectory_points"] == 20


def test_plot_umap_projected_trajectories_writes_endpoint_explorer(
    tmp_path,
    monkeypatch,
) -> None:
    observed = {}

    class FakeUMAP:
        def __init__(self, **kwargs) -> None:
            observed["kwargs"] = kwargs

        def fit_transform(self, values: np.ndarray) -> np.ndarray:
            observed["input_shape"] = values.shape
            return values[:, :3]

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    output_path = tmp_path / "plots" / "trajectory_umap.png"
    coordinates_path = tmp_path / "trajectories" / "trajectory_umap.npz"
    interactive_path = tmp_path / "plots" / "trajectory_umap.html"

    result = plot_umap_projected_trajectories(
        trajectory=torch.rand(4, 3, 8),
        target_samples=torch.rand(6, 8),
        generated_samples=torch.rand(5, 8),
        target_labels=torch.tensor([0, 1, 2, 3, 4, 5]),
        output_path=output_path,
        n_neighbors=4,
        coordinates_path=coordinates_path,
        interactive_path=interactive_path,
        image_shape=[2, 4],
        dataset_name="mnist",
    )

    assert output_path.exists()
    assert interactive_path.exists()
    html = interactive_path.read_text(encoding="utf-8")
    assert 'id="preview"' in html
    assert "<title>Geometry Explorer</title>" in html
    assert "trajectoryLabels" in html
    assert "trajectoryPreviews" in html
    assert 'id="show-thumbnails"' in html
    assert '"drawThumbnailsDefault":false' in html
    assert "THREE.PerspectiveCamera" in html
    assert 'id="show-trajectory"' in html
    assert 'id="time"' in html
    assert "function initializeClassFilter(points, onChange)" in html
    assert "function drawPreviewTile(point, x, y, size)" in html
    assert result["generated_points"] == 5
    assert result["explorer"]["endpoint_points"] == 11
    assert result["explorer"]["trajectory_preview_points"] == 3
    assert result["explorer"]["atlas_points"] == 14
    assert observed["input_shape"] == (6 + 5 + 4 * 3, 8)
    coordinates = np.load(coordinates_path)
    assert coordinates["generated"].shape == (5, 3)


def test_project_saved_trajectories_writes_summary(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "samples").mkdir(parents=True)
    (run_dir / "trajectories").mkdir()
    np.save(run_dir / "samples" / "target_reference.npy", np.zeros((7, 4), dtype=np.float32))
    np.save(run_dir / "samples" / "target_reference_labels.npy", np.arange(7, dtype=np.int64))
    np.save(run_dir / "samples" / "euler_nfe9.npy", np.ones((5, 4), dtype=np.float32))
    np.save(run_dir / "trajectories" / "euler_nfe9.npy", np.zeros((3, 2, 4), dtype=np.float32))
    np.save(
        run_dir / "trajectories" / "source_reference_nfe9.npy",
        np.zeros((2, 4), dtype=np.float32),
    )

    def fake_plot(trajectory, output_path, **kwargs):
        assert kwargs["generated_samples"].shape == (5, 4)
        assert kwargs["target_labels"].shape == (7,)
        assert kwargs["image_shape"] == [2, 2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")
        kwargs["interactive_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["interactive_path"].write_text("html", encoding="utf-8")
        return {
            "plot_path": str(output_path),
            "coordinates_path": None,
            "interactive_path": str(kwargs["interactive_path"]),
            "n_steps": int(trajectory.shape[0]),
            "n_trajectories": int(trajectory.shape[1]),
        }

    monkeypatch.setattr(
        "fm_lab.diagnostics.trajectory_umap.plot_umap_projected_trajectories",
        fake_plot,
    )

    result = project_saved_trajectories(
        TrajectoryUMAPConfig(run_dir=run_dir, nfe=9, save_coordinates=False)
    )

    assert "euler" in result["results"]
    assert (run_dir / "plots" / "trajectory_umap3d_euler_nfe9.png").exists()
    assert (run_dir / "plots" / "trajectory_umap3d_euler_nfe9.html").exists()
    assert (run_dir / "diagnostics" / "trajectory_umap_nfe9.json").exists()


def test_plot_training_history_writes_loss_curve(tmp_path) -> None:
    output_path = tmp_path / "plots" / "training_loss.png"
    history = [
        {"step": 1, "loss": 1.0, "flow_matching_loss": 1.0},
        {
            "step": 10,
            "loss": 0.5,
            "flow_matching_loss": 0.45,
            "straightness_loss": 5.0,
            "straightness_weighted": 0.05,
        },
    ]

    plot_training_history(history, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
