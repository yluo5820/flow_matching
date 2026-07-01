import torch
from torch import nn

from fm_lab.diagnostics.diffusion_lid import flipd_dimension, normal_bundle_dimension


class PlaneNormalScore(nn.Module):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        score = torch.zeros_like(x)
        score[:, 2] = -x[:, 2] / (self.sigma * self.sigma)
        return score


class LineNormalScore(nn.Module):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        score = torch.zeros_like(x)
        score[:, 1:] = -x[:, 1:] / (self.sigma * self.sigma)
        return score


class ConstantScore(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.ones_like(x)


def test_normal_bundle_dimension_recovers_plane_lid() -> None:
    sigma = 0.05
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])
    with torch.random.fork_rng():
        torch.manual_seed(0)
        estimate = normal_bundle_dimension(
            PlaneNormalScore(sigma),
            x,
            sigma=sigma,
            t=0.9,
            n_perturbations=16,
            rank_threshold=1e-5,
        )

    assert estimate.ambient_dimension == 3
    assert torch.equal(estimate.normal_dimension, torch.tensor([1, 1]))
    assert torch.allclose(estimate.intrinsic_dimension, torch.tensor([2.0, 2.0]))


def test_flipd_dimension_recovers_plane_lid_with_exact_divergence() -> None:
    sigma = 0.1
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.25]])
    estimate = flipd_dimension(
        PlaneNormalScore(sigma),
        x,
        sigma=sigma,
        t=0.9,
        hutchinson_samples=None,
    )

    assert torch.allclose(estimate.divergence, torch.full((2,), -1.0 / sigma**2))
    assert torch.allclose(estimate.intrinsic_dimension, torch.tensor([2.0, 2.0]))


def test_flipd_dimension_recovers_line_lid_with_rademacher_trace() -> None:
    sigma = 0.2
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.25]])
    with torch.random.fork_rng():
        torch.manual_seed(0)
        estimate = flipd_dimension(
            LineNormalScore(sigma),
            x,
            sigma=sigma,
            t=0.9,
            hutchinson_samples=4,
        )

    assert torch.allclose(estimate.divergence, torch.full((2,), -2.0 / sigma**2))
    assert torch.allclose(estimate.intrinsic_dimension, torch.tensor([1.0, 1.0]))


def test_flipd_dimension_handles_constant_score_field() -> None:
    x = torch.zeros(2, 3)

    estimate = flipd_dimension(ConstantScore(), x, sigma=0.1, t=0.9, hutchinson_samples=None)

    assert torch.allclose(estimate.divergence, torch.zeros(2))
    assert torch.allclose(estimate.intrinsic_dimension, torch.full((2,), 3.0))
