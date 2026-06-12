import torch
from torch import nn

from fm_lab.diagnostics import (
    curvature_stats,
    generate_solver_samples,
    jacobian_stats,
    pairwise_solver_distances,
)
from fm_lab.experiments.run_field_diagnostics import run_field_diagnostics
from fm_lab.solvers import EulerSolver, HeunSolver
from fm_lab.sources import GaussianSource
from fm_lab.utils.config import ConfigError, load_config


class IdentityVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        return x


class ConstantVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        return torch.ones_like(x)


class UnitXDirectionSpeed(nn.Module):
    requires_source_label = True

    def direction(self, source_label: torch.Tensor) -> torch.Tensor:
        direction = torch.zeros_like(source_label)
        direction[:, 0] = 1.0
        return direction

    def speed(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        source_label: torch.Tensor,
    ) -> torch.Tensor:
        del x, t
        return torch.ones(source_label.shape[0], device=source_label.device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = context["source_label"]
        return self.speed(x, t, source_label)[:, None] * self.direction(source_label)


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


def test_solver_sensitivity_supports_source_label_conditioned_models() -> None:
    samples = generate_solver_samples(
        model=UnitXDirectionSpeed(),
        source=GaussianSource(dim=3),
        solvers=[EulerSolver(), HeunSolver()],
        n_samples=16,
        nfe=4,
        device=torch.device("cpu"),
    )

    rows = pairwise_solver_distances(samples, metrics=("mmd", "sliced_wasserstein"))

    assert samples["euler"].shape == (16, 3)
    assert rows[0]["mmd"] < 1e-6
    assert rows[0]["sliced_wasserstein"] < 1e-6


def test_field_diagnostics_reject_label_conditioned_models() -> None:
    config = load_config("configs/toy/gaussian_to_gaussian_mixture_linear_3d_direction_only.yaml")

    try:
        run_field_diagnostics(
            payload={"model_state_dict": {}},
            config=config,
            device=torch.device("cpu"),
            n_samples=4,
        )
    except ConfigError as exc:
        assert "Eulerian" in str(exc)
    else:
        raise AssertionError("Expected label-conditioned field diagnostics to be rejected.")
