import torch

from fm_lab.diagnostics import grid_ambiguity, knn_ambiguity, sliced_wasserstein, squared_mmd


def test_grid_ambiguity_zero_for_constant_local_velocity() -> None:
    xt = torch.tensor([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0], [2.1, 2.1]])
    velocities = torch.ones_like(xt)

    result = grid_ambiguity(xt, velocities, bins=2, min_count=2)

    assert result.ambiguity == 0.0


def test_knn_ambiguity_positive_for_conflicting_velocities() -> None:
    xt = torch.zeros(4, 2)
    velocities = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])

    result = knn_ambiguity(xt, velocities, k=4)

    assert result["ambiguity"] > 0.0


def test_distribution_metrics_are_small_for_identical_samples() -> None:
    samples = torch.randn(32, 2)

    assert squared_mmd(samples, samples) < 1e-6
    assert sliced_wasserstein(samples, samples) < 1e-6
