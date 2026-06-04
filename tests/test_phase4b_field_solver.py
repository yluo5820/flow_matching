import torch
from torch import nn

from fm_lab.diagnostics import (
    curvature_stats,
    generate_solver_samples,
    jacobian_stats,
    pairwise_solver_distances,
)
from fm_lab.solvers import EulerSolver, HeunSolver
from fm_lab.sources import GaussianSource


class IdentityVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        return x


class ConstantVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        return torch.ones_like(x)


def test_jacobian_stats_for_identity_field() -> None:
    model = IdentityVelocity()
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)

    stats = jacobian_stats(model, x, t)

    assert stats["spectral_mean"] == 1.0
    assert stats["divergence_mean"] == 2.0


def test_curvature_stats_for_identity_field() -> None:
    model = IdentityVelocity()
    x = torch.randn(8, 2)
    t = torch.full((8,), 0.5)

    stats = curvature_stats(model, x, t)

    assert torch.allclose(stats["acceleration"], x)


def test_solver_sensitivity_zero_for_constant_velocity() -> None:
    samples = generate_solver_samples(
        model=ConstantVelocity(),
        source=GaussianSource(dim=2),
        solvers=[EulerSolver(), HeunSolver()],
        n_samples=16,
        nfe=4,
        device=torch.device("cpu"),
    )

    rows = pairwise_solver_distances(samples, metrics=("mmd", "sliced_wasserstein"))

    assert rows[0]["mmd"] < 1e-6
    assert rows[0]["sliced_wasserstein"] < 1e-6
