import torch
from torch import nn

from fm_lab.paths import LinearPath
from fm_lab.training.losses import (
    build_objective,
    flow_matching_loss,
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
