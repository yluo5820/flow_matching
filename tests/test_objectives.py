import torch
from torch import nn

from fm_lab.paths import LearnedAccelerationPath, LinearPath
from fm_lab.training.losses import (
    KernelVStarConfig,
    build_objective,
    flow_matching_loss,
    kernel_vstar_estimate,
    learned_flow_straightness_loss,
)


class ConstantVelocity(nn.Module):
    def __init__(self, dim: int, value: float = 1.0) -> None:
        super().__init__()
        self.velocity = nn.Parameter(torch.full((dim,), value))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return self.velocity.expand_as(x)


class TimeScaledVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t[:, None] * x


class FixedDirectionSpeed(nn.Module):
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
        return torch.zeros(source_label.shape[0], device=source_label.device)


def test_default_objective_matches_flow_matching_loss() -> None:
    model = ConstantVelocity(dim=3, value=1.0)
    path = LinearPath()
    x0 = torch.zeros(8, 3)
    x1 = torch.ones(8, 3)
    t = torch.full((8,), 0.5)

    direct_loss, direct_metrics = flow_matching_loss(model, path, x0, x1, t)
    objective = build_objective({})
    objective_loss, objective_metrics = objective(model=model, path=path, x0=x0, x1=x1, t=t)

    assert torch.allclose(direct_loss, objective_loss)
    assert direct_metrics["loss"] == objective_metrics["loss"]
    assert objective_metrics["flow_matching_loss"] == 0.0


def test_learned_flow_straightness_is_zero_for_constant_field() -> None:
    model = ConstantVelocity(dim=3, value=0.5)
    x = torch.randn(8, 3)
    t = torch.linspace(0.1, 0.9, 8)

    loss = learned_flow_straightness_loss(model=model, x=x, t=t)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_learned_flow_straightness_is_positive_for_curved_field() -> None:
    model = TimeScaledVelocity()
    x = torch.randn(8, 3)
    t = torch.linspace(0.1, 0.9, 8)

    loss = learned_flow_straightness_loss(model=model, x=x, t=t)

    assert loss > 0


def test_learned_flow_straightness_uses_supplied_advective_velocity() -> None:
    x = torch.randn(8, 3)
    t = torch.linspace(0.1, 0.9, 8)
    advective = torch.randn_like(x)

    constant_loss = learned_flow_straightness_loss(
        model=ConstantVelocity(dim=3, value=0.5),
        x=x,
        t=t,
        advective_velocity=advective,
    )
    curved_loss = learned_flow_straightness_loss(
        model=TimeScaledVelocity(),
        x=x,
        t=t,
        advective_velocity=torch.ones_like(x),
    )

    assert torch.allclose(constant_loss, torch.tensor(0.0))
    assert curved_loss > 0


def test_kernel_vstar_estimator_recovers_constant_target_velocity() -> None:
    path = LinearPath()
    shift = torch.tensor([0.25, -1.0, 0.5])
    x0 = torch.randn(12, 3)
    x1 = x0 + shift
    t = torch.linspace(0.1, 0.9, 12)
    config = KernelVStarConfig(estimator_size=12, query_size=12, bandwidth=5.0)

    estimate = kernel_vstar_estimate(
        path=path,
        x0=x0,
        x1=x1,
        t=t,
        config=config,
    )

    assert estimate.vstar.shape == (12, 3)
    assert torch.allclose(estimate.vstar, shift.expand_as(estimate.vstar), atol=1.0e-6)
    assert estimate.metrics["kernel_vstar_effective_sample_size_mean"] > 1.0


def test_kernel_vstar_estimator_is_finite_for_2d_and_3d_batches() -> None:
    for dim in (2, 3):
        path = LearnedAccelerationPath(dim=dim, hidden_dim=8, depth=1)
        x0 = torch.randn(10, dim)
        x1 = torch.randn(10, dim)
        t = torch.linspace(0.1, 0.9, 10)
        config = KernelVStarConfig(estimator_size=8, query_size=4)

        estimate = kernel_vstar_estimate(
            path=path,
            x0=x0,
            x1=x1,
            t=t,
            config=config,
        )

        assert estimate.query_x.shape == (4, dim)
        assert torch.isfinite(estimate.vstar).all()
        assert all(value == value for value in estimate.metrics.values())


def test_objective_adds_straightness_metrics() -> None:
    objective = build_objective(
        {"straightness": {"weight": 0.25, "sample_size": 4}},
    )
    model = TimeScaledVelocity()
    path = LinearPath()
    x0 = torch.zeros(8, 3)
    x1 = torch.ones(8, 3)
    t = torch.linspace(0.1, 0.9, 8)

    loss, metrics = objective(model=model, path=path, x0=x0, x1=x1, t=t)

    assert loss > 0
    assert metrics["straightness_loss"] > 0
    assert metrics["straightness_weighted"] > 0


def test_objective_rejects_invalid_straightness_config() -> None:
    try:
        build_objective({"straightness": {"weight": -1.0}})
    except ValueError as exc:
        assert "weight" in str(exc)
    else:
        raise AssertionError("Expected invalid straightness weight to raise.")


def test_objective_adds_interpolant_acceleration_metrics() -> None:
    objective = build_objective({"interpolant_acceleration": {"weight": 0.5}})
    model = ConstantVelocity(dim=2, value=0.0)
    path = LearnedAccelerationPath(dim=2, hidden_dim=8, depth=1)
    output = path.net[-1]
    assert isinstance(output, nn.Linear)
    with torch.no_grad():
        output.bias.fill_(2.0)
    x0 = torch.zeros(4, 2)
    x1 = torch.ones(4, 2)
    t = torch.full((4,), 0.5)

    loss, metrics = objective(model=model, path=path, x0=x0, x1=x1, t=t)

    assert loss > 0
    assert metrics["interpolant_acceleration_loss"] == 8.0
    assert metrics["interpolant_acceleration_weighted"] == 4.0
    assert metrics["interpolant_acceleration_norm_mean"] > 0


def test_direction_only_objective_computes_decomposed_losses() -> None:
    objective = build_objective(
        {
            "name": "direction_only_straight",
            "direction_weight": 2.0,
            "speed_weight": 3.0,
        }
    )
    model = FixedDirectionSpeed()
    path = LinearPath()
    x0 = torch.zeros(2, 2)
    x1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t = torch.full((2,), 0.5)

    loss, metrics = objective(model=model, path=path, x0=x0, x1=x1, t=t)

    assert torch.allclose(loss, torch.tensor(2.5))
    assert metrics["direction_loss"] == 0.5
    assert metrics["speed_loss"] == 0.5
    assert metrics["direction_weighted"] == 1.0
    assert metrics["speed_weighted"] == 1.5
    assert metrics["direction_speed_vector_mse"] == 0.5
    assert metrics["perpendicular_residual_mean"] == 0.5
