import pytest
import torch

from fm_lab.couplings import MinibatchOTCoupling, ReflowCouplingPlaceholder


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
