import numpy as np
import pytest
import torch
from torch import nn

from fm_lab.diagnostics.probes.gradients import (
    collect_gradient_rows,
    resolve_probe_layers,
)
from fm_lab.diagnostics.probes.manifest import ProbeBatch
from fm_lab.paths import LinearPath
from fm_lab.training.losses import build_objective


class TinyConditionalVelocity(nn.Module):
    is_class_conditional = True

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del context
        return self.output(torch.tanh(self.hidden(x + t[:, None])))


def _probe_batches() -> tuple[ProbeBatch, ...]:
    batches = []
    for batch_id in range(3):
        x0 = torch.tensor([[0.1, -0.2], [0.3, 0.4]]) + batch_id * 0.1
        x1 = torch.tensor([[0.8, 0.2], [-0.1, 0.7]]) - batch_id * 0.05
        batches.append(
            ProbeBatch(
                x0=x0,
                x1=x1,
                t=torch.tensor([0.2, 0.7]),
                labels=torch.tensor([0, 1]),
                original_indices=np.array([2 * batch_id, 2 * batch_id + 1]),
                stratum_ids=np.array([0, 0]),
                microbatch_ids=np.array([batch_id, batch_id]),
            )
        )
    return tuple(batches)


def test_resolve_probe_layers_preserves_requested_order_and_identity() -> None:
    model = TinyConditionalVelocity()

    layers = resolve_probe_layers(
        model,
        ("output.weight", "hidden.weight"),
    )

    assert [layer.name for layer in layers] == ["output.weight", "hidden.weight"]
    assert layers[0].parameter is model.output.weight
    assert layers[1].shape == tuple(model.hidden.weight.shape)


def test_resolve_probe_layers_rejects_non_weight_and_duplicate_parameters() -> None:
    model = TinyConditionalVelocity()
    with pytest.raises(ValueError, match="weight"):
        resolve_probe_layers(model, ("hidden.bias",))
    with pytest.raises(ValueError, match="duplicate"):
        resolve_probe_layers(model, ("hidden.weight", "hidden.weight"))


def test_gradient_probe_collects_requested_layers_without_mutation() -> None:
    torch.manual_seed(7)
    model = TinyConditionalVelocity()
    model.train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    objective = build_objective(
        {
            "name": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
            "modifiers": [],
        }
    )

    rows = collect_gradient_rows(
        model=model,
        objective=objective,
        path=LinearPath(),
        batches=_probe_batches(),
        layer_names=("hidden.weight", "output.weight"),
    )

    assert rows["hidden.weight"].raw.shape == (3, model.hidden.weight.numel())
    assert torch.all(rows["hidden.weight"].norms > 0)
    assert torch.allclose(
        torch.linalg.vector_norm(rows["hidden.weight"].normalized, dim=1),
        torch.ones(3),
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(before[name], model.state_dict()[name]) for name in before)
    assert model.training is True
