import torch
from torch import nn

from fm_lab.diagnostics.fm_lid import (
    FMFLIPDEstimator,
    FMJacobianSpectrumEstimator,
    GaussianFMSchedule,
    entropy_rank,
    participation_rank,
    summarize_lid_values,
    threshold_rank,
)


class ZeroVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(x)


class ProjectionFlowSolver:
    name = "projection_flow"

    def __init__(self, tangent_dim: int) -> None:
        self.tangent_dim = tangent_dim

    def solve(
        self,
        v_fn,
        x0: torch.Tensor,
        t_grid: torch.Tensor,
        return_trajectory: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del v_fn, kwargs
        final = x0.clone()
        if float(t_grid[-1]) > float(t_grid[0]):
            final[:, self.tangent_dim :] = 0.0
        if return_trajectory:
            return torch.stack([x0, final], dim=0)
        return final


class FlatSubspaceGaussianFMVelocity(nn.Module):
    def __init__(self, dim: int, tangent_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.tangent_dim = tangent_dim

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha = t[:, None].clamp_min(1e-4)
        sigma = (1.0 - t[:, None]).clamp_min(1e-4)
        velocity = torch.zeros_like(y)
        velocity[:, : self.tangent_dim] = y[:, : self.tangent_dim] / alpha
        velocity[:, self.tangent_dim :] = -y[:, self.tangent_dim :] / sigma
        return velocity


def test_effective_rank_utilities() -> None:
    singular_values = torch.tensor([2.0, 2.0, 0.0])

    assert torch.allclose(participation_rank(singular_values), torch.tensor(2.0))
    assert torch.allclose(entropy_rank(singular_values), torch.tensor(2.0))
    assert torch.equal(threshold_rank(singular_values, threshold=1e-2), torch.tensor(2))


def test_fm_jacobian_spectrum_recovers_projection_line_rank() -> None:
    estimator = FMJacobianSpectrumEstimator(
        model=ZeroVelocity(),
        ode_solver=ProjectionFlowSolver(tangent_dim=1),
        t_values=[0.5, 0.9],
        eps=1e-2,
        num_directions=16,
        threshold=1e-5,
        device="cpu",
        nfe=4,
    )
    x = torch.tensor([1.0, 0.0, 0.0])

    estimate = estimator.estimate_point(x)

    assert torch.equal(estimate.threshold_rank, torch.tensor([1, 1]))
    assert torch.allclose(estimate.participation_rank, torch.ones(2))
    assert estimate.as_dict()["fm_jacobian_threshold_rank"] == [1, 1]


def test_fm_jacobian_spectrum_recovers_projection_plane_rank() -> None:
    estimator = FMJacobianSpectrumEstimator(
        model=ZeroVelocity(),
        ode_solver=ProjectionFlowSolver(tangent_dim=2),
        t_values=[0.5],
        eps=1e-2,
        num_directions=32,
        threshold=1e-5,
        device="cpu",
        nfe=4,
    )
    x = torch.tensor([1.0, -1.0, 0.0])

    estimate = estimator.estimate_point(x)

    assert torch.equal(estimate.threshold_rank, torch.tensor([2]))
    assert estimate.participation_rank[0] > 1.4
    assert estimate.participation_rank[0] <= 2.0


def test_fm_flipd_recovers_flat_subspace_dimensions_with_exact_divergence() -> None:
    x = torch.tensor([[1.0, -1.0, 0.0], [0.5, 2.0, 0.0]])
    estimator = FMFLIPDEstimator(
        FlatSubspaceGaussianFMVelocity(dim=3, tangent_dim=2),
        GaussianFMSchedule("linear"),
        t_values=[0.7, 0.9],
        num_trace_samples=None,
        device="cpu",
    )

    estimate = estimator.estimate_batch(x)

    assert estimate.lid.shape == (2, 2)
    assert torch.allclose(estimate.lid, torch.full((2, 2), 2.0), atol=1e-5)
    assert estimate.ambient_dimension == 3


def test_fm_flipd_recovers_line_dimension_with_hutchinson_trace() -> None:
    x = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    estimator = FMFLIPDEstimator(
        FlatSubspaceGaussianFMVelocity(dim=3, tangent_dim=1),
        "linear",
        t_values=[0.8],
        num_trace_samples=4,
        device="cpu",
    )

    with torch.random.fork_rng():
        torch.manual_seed(0)
        estimate = estimator.estimate_batch(x)

    assert torch.allclose(estimate.lid, torch.full((1, 2), 1.0), atol=1e-5)


def test_lid_summary_reports_expected_statistics() -> None:
    summary = summarize_lid_values(torch.tensor([1.0, 2.0, 3.0]))

    assert summary["mean_lid"] == 2.0
    assert summary["median_lid"] == 2.0
    assert len(summary["quantiles"]) == 5
