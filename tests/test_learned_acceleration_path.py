import torch
from torch import nn

from fm_lab.paths import LearnedAccelerationPath, LinearPath
from fm_lab.training.losses import build_objective


class TimeScaledVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t[:, None] * x


def test_zero_initialized_learned_acceleration_matches_linear_path() -> None:
    path = LearnedAccelerationPath(dim=2, hidden_dim=8, depth=1)
    linear = LinearPath()
    x0 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    x1 = torch.tensor([[2.0, 5.0], [4.0, 7.0]])
    t = torch.tensor([0.25, 0.75])

    assert torch.allclose(path.sample_xt(x0, x1, t), linear.sample_xt(x0, x1, t))
    assert torch.allclose(path.target_velocity(x0, x1, t), linear.target_velocity(x0, x1, t))


def test_quadratic_learned_acceleration_formula_and_endpoints() -> None:
    path = LearnedAccelerationPath(dim=2, basis="quadratic", hidden_dim=8, depth=1)
    _set_constant_acceleration(path, torch.tensor([1.0, -2.0]))
    x0 = torch.tensor([[0.0, 0.0]])
    x1 = torch.tensor([[2.0, 4.0]])
    t = torch.tensor([0.25])

    assert torch.allclose(path.sample_xt(x0, x1, torch.tensor([0.0])), x0)
    assert torch.allclose(path.sample_xt(x0, x1, torch.tensor([1.0])), x1)
    assert torch.allclose(path.sample_xt(x0, x1, t), torch.tensor([[0.6875, 0.6250]]))
    assert torch.allclose(path.target_velocity(x0, x1, t), torch.tensor([[2.5, 3.0]]))
    assert torch.allclose(path.conditional_acceleration(x0, x1, t), torch.tensor([[-2.0, 4.0]]))


def test_endpoint_bump_formula_preserves_endpoint_velocities() -> None:
    path = LearnedAccelerationPath(dim=2, basis="endpoint_bump", hidden_dim=8, depth=1)
    _set_constant_acceleration(path, torch.tensor([1.0, -2.0]))
    x0 = torch.tensor([[0.0, 0.0]])
    x1 = torch.tensor([[2.0, 4.0]])
    t = torch.tensor([0.25])

    assert torch.allclose(path.sample_xt(x0, x1, torch.tensor([0.0])), x0)
    assert torch.allclose(path.sample_xt(x0, x1, torch.tensor([1.0])), x1)
    assert torch.allclose(path.target_velocity(x0, x1, torch.tensor([0.0])), x1 - x0)
    assert torch.allclose(path.target_velocity(x0, x1, torch.tensor([1.0])), x1 - x0)
    assert torch.allclose(path.sample_xt(x0, x1, t), torch.tensor([[0.53515625, 0.9296875]]))
    assert torch.allclose(path.target_velocity(x0, x1, t), torch.tensor([[2.1875, 3.6250]]))
    assert torch.allclose(
        path.conditional_acceleration(x0, x1, t),
        torch.tensor([[-0.25, 0.5]]),
    )


def test_learned_acceleration_penalty_and_diagnostics_are_finite() -> None:
    path = LearnedAccelerationPath(dim=3, hidden_dim=8, depth=1)
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    t = torch.full((16,), 0.5)

    penalty = path.acceleration_penalty(x0, x1)
    diagnostics = path.diagnostics(x0, x1, t)

    assert torch.isfinite(penalty)
    assert diagnostics["interpolant_acceleration_norm_mean"] == 0.0
    assert diagnostics["interpolant_endpoint_error_max"] < 1.0e-6
    assert all(value == value for value in diagnostics.values())


def test_psi_update_straightness_loss_backpropagates_to_path() -> None:
    path = LearnedAccelerationPath(dim=2, hidden_dim=8, depth=1)
    objective = build_objective({"straightness": {"weight": 1.0}})
    x0 = torch.tensor([[0.0, 1.0], [1.0, -1.0]])
    x1 = torch.tensor([[2.0, 0.0], [-1.0, 2.0]])
    t = torch.full((2,), 0.5)

    loss, _ = objective.psi_update_loss(
        model=TimeScaledVelocity(),
        path=path,
        x0=x0,
        x1=x1,
        t=t,
    )
    loss.backward()

    grad_norm = sum(
        parameter.grad.abs().sum()
        for parameter in path.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0


def test_kernel_vstar_psi_update_backpropagates_to_path() -> None:
    path = LearnedAccelerationPath(dim=2, hidden_dim=8, depth=1)
    objective = build_objective(
        {
            "straightness": {"weight": 1.0},
            "learned_interpolant": {
                "mode": "kernel_vstar",
                "estimator_size": 4,
                "query_size": 2,
                "bandwidth": 10.0,
            },
        }
    )
    x0 = torch.tensor(
        [[0.0, 1.0], [1.0, -1.0], [0.5, 0.5], [-0.5, 1.5]],
    )
    x1 = torch.tensor(
        [[2.0, 0.0], [-1.0, 2.0], [1.5, -0.5], [0.25, -1.0]],
    )
    t = torch.full((4,), 0.5)

    loss, metrics = objective.psi_update_loss(
        model=TimeScaledVelocity(),
        path=path,
        x0=x0,
        x1=x1,
        t=t,
    )
    loss.backward()

    grad_norm = sum(
        parameter.grad.abs().sum()
        for parameter in path.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0
    assert metrics["kernel_vstar_effective_sample_size_mean"] > 0


def _set_constant_acceleration(path: LearnedAccelerationPath, value: torch.Tensor) -> None:
    output = path.net[-1]
    assert isinstance(output, nn.Linear)
    with torch.no_grad():
        output.weight.zero_()
        output.bias.copy_(value)
