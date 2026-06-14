import torch

from fm_lab.plotting.diagnostics import plot_training_history
from fm_lab.plotting.trajectories import plot_generated_samples, plot_trajectories


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
