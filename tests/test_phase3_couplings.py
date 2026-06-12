import pytest
import torch
from torch import nn

from fm_lab.couplings import (
    MinibatchOTCoupling,
    ModelGeneratedCoupling,
    ReflowCouplingPlaceholder,
)
from fm_lab.experiments.factory import build_coupling
from fm_lab.models import MLPVelocity
from fm_lab.solvers import EulerSolver
from fm_lab.utils.checkpoints import save_checkpoint


class ConstantVelocity(nn.Module):
    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return self.velocity.expand_as(x)


def test_minibatch_ot_pairs_by_minimum_assignment() -> None:
    coupling = MinibatchOTCoupling()
    x0 = torch.tensor([[0.0], [10.0], [20.0]])
    x1 = torch.tensor([[19.0], [1.0], [11.0]])

    paired_x0, paired_x1 = coupling.pair(x0, x1)

    assert torch.allclose(paired_x0, x0)
    assert torch.allclose(paired_x1, torch.tensor([[1.0], [11.0], [19.0]]))


def test_reflow_placeholder_fails_loudly() -> None:
    coupling = ReflowCouplingPlaceholder()

    with pytest.raises(NotImplementedError, match="placeholder"):
        coupling.pair(torch.zeros(2, 2), torch.ones(2, 2))


def test_model_generated_coupling_uses_teacher_endpoint() -> None:
    teacher = ConstantVelocity(torch.tensor([1.0, -2.0]))
    coupling = ModelGeneratedCoupling(
        teacher_model=teacher,
        solver=EulerSolver(),
        nfe=4,
        schedule="uniform",
    )
    x0 = torch.tensor([[0.0, 0.0], [2.0, 3.0]])
    sampled_x1 = torch.full_like(x0, 100.0)

    paired_x0, paired_x1 = coupling.pair(x0, sampled_x1)

    assert torch.allclose(paired_x0, x0)
    assert torch.allclose(paired_x1, x0 + torch.tensor([1.0, -2.0]))


def test_model_generated_coupling_rejects_dimension_mismatch() -> None:
    coupling = ModelGeneratedCoupling(
        teacher_model=MLPVelocity(dim=3, hidden_dim=8, depth=1, time_embedding_dim=4),
        solver=EulerSolver(),
        nfe=2,
    )

    with pytest.raises(ValueError, match="teacher dimension"):
        coupling.pair(torch.zeros(4, 2), torch.zeros(4, 2))


def test_factory_builds_model_generated_coupling_from_checkpoint(tmp_path) -> None:
    teacher = MLPVelocity(dim=2, hidden_dim=8, depth=1, time_embedding_dim=4)
    checkpoint_path = tmp_path / "teacher.pt"
    save_checkpoint(
        checkpoint_path,
        model=teacher,
        optimizer=None,
        step=3,
        config={
            "source": {"name": "gaussian", "dim": 2},
            "model": {
                "name": "mlp",
                "hidden_dim": 8,
                "depth": 1,
                "activation": "silu",
                "time_embedding_dim": 4,
            },
        },
        metrics={},
    )

    coupling = build_coupling(
        {
            "coupling": {
                "name": "model_generated",
                "checkpoint_path": str(checkpoint_path),
                "solver": "euler",
                "nfe": 2,
            }
        }
    )

    assert isinstance(coupling, ModelGeneratedCoupling)
    assert coupling.nfe == 2
    assert coupling.solver.name == "euler"
