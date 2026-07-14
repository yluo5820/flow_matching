import pytest
import torch
from torch import nn

from fm_lab.paths import LinearPath
from fm_lab.training.long_tail import CBDMModifier, build_continuous_modifiers
from fm_lab.training.losses import build_objective


class LabelTablePrediction(nn.Module):
    is_class_conditional = True

    def __init__(self) -> None:
        super().__init__()
        self.predictions = nn.Parameter(torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
        self.seen_t: list[torch.Tensor] = []
        self.seen_labels: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        labels = context["class_labels"]
        self.seen_t.append(t.detach().clone())
        self.seen_labels.append(labels.detach().clone())
        return self.predictions[labels].reshape_as(x)


@pytest.mark.parametrize(
    ("distribution", "expected"),
    [
        ("train", [0.1, 0.9]),
        ("sqrt", [0.25, 0.75]),
        ("uniform", [0.5, 0.5]),
    ],
)
def test_cbdm_builds_auxiliary_distributions(
    distribution: str,
    expected: list[float],
) -> None:
    modifier = CBDMModifier(
        class_counts=[1, 9],
        target_distribution=distribution,
        tau=0.1,
        gamma=0.25,
        comparison_space="velocity",
    )

    assert torch.allclose(modifier.auxiliary_probabilities, torch.tensor(expected))


def test_cbdm_weight_is_one_minus_continuous_time() -> None:
    modifier = CBDMModifier(
        class_counts=[1, 9],
        target_distribution="uniform",
        tau=2.0,
        gamma=0.25,
        comparison_space="velocity",
    )
    assert torch.equal(
        modifier.time_weight(torch.tensor([0.0, 0.25, 1.0])),
        torch.tensor([1.0, 0.75, 0.0]),
    )


def test_continuous_modifier_builder_validates_names_and_class_counts() -> None:
    with pytest.raises(ValueError, match="class_counts"):
        build_continuous_modifiers([{"name": "cbdm"}], None)
    with pytest.raises(ValueError, match="Duplicate continuous modifier: cbdm"):
        build_continuous_modifiers(
            [{"name": "cbdm"}, {"name": "cbdm"}],
            [1, 1],
        )
    with pytest.raises(ValueError, match="cbdm, oc, and cm"):
        build_continuous_modifiers([{"name": "other"}], [1, 1])


def _cbdm_gradients(
    gamma: float,
) -> tuple[torch.Tensor, dict[str, object], LabelTablePrediction]:
    objective = build_objective(
        {
            "name": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
            "modifiers": [
                {
                    "name": "cbdm",
                    "target_distribution": "train",
                    "tau": 0.1,
                    "gamma": gamma,
                    "comparison_space": "velocity",
                }
            ],
        },
        class_counts=[1, 1_000_000],
    )
    model = LabelTablePrediction()
    torch.manual_seed(0)
    loss, metrics = objective(
        model=model,
        path=LinearPath(),
        x0=torch.zeros(1, 2),
        x1=torch.ones(1, 2),
        t=torch.tensor([0.25]),
        class_labels=torch.tensor([0]),
        original_class_labels=torch.tensor([0]),
    )
    loss.backward()
    assert model.predictions.grad is not None
    return model.predictions.grad.clone(), metrics, model


def test_cbdm_regularizer_stops_auxiliary_gradient() -> None:
    gradients, metrics, model = _cbdm_gradients(gamma=0.0)

    assert torch.equal(model.seen_labels[0], torch.tensor([0]))
    assert torch.equal(model.seen_labels[1], torch.tensor([1]))
    assert torch.equal(model.seen_t[0], torch.tensor([0.25]))
    assert torch.equal(model.seen_t[1], torch.tensor([0.25]))
    assert metrics["base.loss"] == 0.0
    assert metrics["cbdm.regularizer"] == pytest.approx(0.3)
    assert metrics["cbdm.commitment"] == 0.0
    assert metrics["cbdm.auxiliary_distribution"] == "train"
    assert torch.count_nonzero(gradients[0]) == 2
    assert torch.count_nonzero(gradients[1]) == 0


def test_cbdm_commitment_does_not_change_base_prediction_gradient() -> None:
    without_commitment, _, _ = _cbdm_gradients(gamma=0.0)
    with_commitment, _, _ = _cbdm_gradients(gamma=0.5)

    assert torch.equal(with_commitment[0], without_commitment[0])
    assert torch.count_nonzero(with_commitment[1]) == 2


def test_flow_matching_objective_composes_cbdm_in_declared_comparison_space() -> None:
    objective = build_objective(
        {
            "name": "flow_matching",
            "model_output": "target",
            "loss_space": "target",
            "modifiers": [
                {
                    "name": "cbdm",
                    "target_distribution": "train",
                    "tau": 0.1,
                    "gamma": 0.25,
                    "comparison_space": "velocity",
                }
            ],
        },
        class_counts=[1, 1_000_000],
    )
    model = LabelTablePrediction()
    torch.manual_seed(0)

    loss, metrics = objective(
        model=model,
        path=LinearPath(),
        x0=torch.zeros(1, 2),
        x1=torch.ones(1, 2),
        t=torch.tensor([0.5]),
        class_labels=torch.tensor([0]),
        original_class_labels=torch.tensor([0]),
    )

    # Target outputs 1 and 3 become velocities 1 and 5 at t=0.5.
    assert torch.allclose(loss, torch.tensor(1.0))
    assert metrics["base.loss"] == 0.0
    assert metrics["cbdm.regularizer"] == pytest.approx(0.8)
    assert metrics["cbdm.commitment"] == pytest.approx(0.2)
    assert objective.metadata()["modifiers"] == [
        {
            "name": "cbdm",
            "target_distribution": "train",
            "tau": 0.1,
            "gamma": 0.25,
            "comparison_space": "velocity",
            "class_counts": [1, 1_000_000],
        }
    ]
