import pytest
import torch
from torch import nn

from fm_lab.diagnostics.probes.perturbations import virtual_layer_update


def test_virtual_layer_update_restores_exact_parameter_after_success_and_error() -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    original = model[0].weight.detach().clone()
    direction = torch.arange(6, dtype=torch.float32) + 1
    direction /= torch.linalg.vector_norm(direction)

    with virtual_layer_update(
        model,
        layer_name="0.weight",
        direction=direction,
        relative_step=1e-3,
    ) as applied_norm:
        assert not torch.equal(model[0].weight, original)
        assert applied_norm == pytest.approx(1e-3 * float(torch.linalg.vector_norm(original)))
    assert torch.equal(model[0].weight, original)

    with pytest.raises(RuntimeError, match="inside"):
        with virtual_layer_update(
            model,
            layer_name="0.weight",
            direction=direction,
            relative_step=1e-3,
        ):
            raise RuntimeError("inside")
    assert torch.equal(model[0].weight, original)


@pytest.mark.parametrize(
    ("direction", "relative_step", "message"),
    [
        (torch.ones(5), 1e-3, "shape"),
        (torch.ones(6), 0.0, "positive"),
        (torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.float32), 1e-3, "unit"),
    ],
)
def test_virtual_layer_update_rejects_invalid_update(
    direction: torch.Tensor,
    relative_step: float,
    message: str,
) -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))

    with pytest.raises(ValueError, match=message):
        with virtual_layer_update(
            model,
            layer_name="0.weight",
            direction=direction,
            relative_step=relative_step,
        ):
            pass
