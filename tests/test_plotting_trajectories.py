import torch

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
