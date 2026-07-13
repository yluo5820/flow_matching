import pytest
import torch
from torch import nn

from fm_lab.training.losses import build_objective


class FixedConditionalPrediction(nn.Module):
    is_class_conditional = True

    def __init__(self, prediction: torch.Tensor) -> None:
        super().__init__()
        self.prediction = nn.Parameter(prediction.clone())
        self.seen_t: torch.Tensor | None = None
        self.seen_labels: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        self.seen_t = t.detach().clone()
        self.seen_labels = context["class_labels"].detach().clone()
        return self.prediction.expand_as(x)


class LabelTablePrediction(nn.Module):
    is_class_conditional = True

    def __init__(self) -> None:
        super().__init__()
        self.predictions = nn.Parameter(torch.tensor([[1.0, 1.0], [3.0, 3.0]]))
        self.seen_labels: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del t
        labels = context["class_labels"]
        self.seen_labels.append(labels.detach().clone())
        return self.predictions[labels].reshape_as(x)


def _objective(mode: str):
    return build_objective(
        {"name": "discrete_diffusion", "prediction_type": mode},
        diffusion_config={
            "timesteps": 10,
            "beta_start": 1e-4,
            "beta_end": 2e-2,
            "variance": "fixed_large",
        },
    )


def test_discrete_epsilon_objective_is_exact_noise_mse() -> None:
    noise = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    prediction = torch.tensor([[0.0, 1.0]])
    model = FixedConditionalPrediction(prediction)
    labels = torch.tensor([3, -1])

    loss, metrics = _objective("epsilon")(
        model=model,
        path=None,
        x0=noise,
        x1=torch.randn_like(noise),
        t=torch.tensor([2, 7]),
        class_labels=labels,
    )

    assert torch.allclose(loss, (prediction.expand_as(noise) - noise).square().mean())
    assert metrics["loss"] == float(loss.detach())
    assert torch.equal(model.seen_t, torch.tensor([2, 7]))
    assert torch.equal(model.seen_labels, labels)


def test_x_prediction_velocity_loss_is_zero_for_clean_prediction() -> None:
    clean = torch.tensor([[0.2, -0.4]])
    model = FixedConditionalPrediction(clean)

    loss, _ = _objective("x_vloss")(
        model=model,
        path=None,
        x0=torch.tensor([[0.7, -1.1]]),
        x1=clean,
        t=torch.tensor([6]),
        class_labels=torch.tensor([1]),
    )

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_discrete_objective_samples_integer_timesteps_and_backpropagates() -> None:
    model = FixedConditionalPrediction(torch.zeros(1, 2))
    objective = _objective("epsilon")

    loss, _ = objective(
        model=model,
        path=None,
        x0=torch.randn(4, 2),
        x1=torch.randn(4, 2),
        t=torch.rand(4),
        class_labels=torch.arange(4),
    )
    loss.backward()

    assert model.seen_t is not None
    assert model.seen_t.dtype == torch.long
    assert torch.all((model.seen_t >= 0) & (model.seen_t < 10))
    assert model.prediction.grad is not None


def test_discrete_objective_metadata_records_schedule_and_track() -> None:
    metadata = _objective("epsilon").metadata()

    assert metadata == {
        "name": "discrete_diffusion",
        "prediction_type": "epsilon",
        "loss": "mse",
        "timesteps": 10,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "variance": "fixed_large",
    }


def test_cbdm_builds_auxiliary_distribution_from_class_counts() -> None:
    objective = build_objective(
        {
            "name": "cbdm",
            "prediction_type": "epsilon",
            "cbdm": {"target_distribution": "sqrt", "tau": 0.001, "gamma": 0.25},
        },
        diffusion_config={"timesteps": 10},
        class_counts=[1, 9],
    )

    assert objective.method == "cbdm"
    assert torch.allclose(objective.auxiliary_probabilities, torch.tensor([0.25, 0.75]))
    assert objective.metadata()["cbdm"] == {
        "target_distribution": "sqrt",
        "tau": 0.001,
        "gamma": 0.25,
        "class_counts": [1, 9],
    }


def test_cbdm_regularizer_stops_auxiliary_gradient_when_commitment_is_zero() -> None:
    objective = build_objective(
        {
            "name": "cbdm",
            "prediction_type": "epsilon",
            "cbdm": {"target_distribution": "train", "tau": 0.1, "gamma": 0.0},
        },
        diffusion_config={"timesteps": 10},
        class_counts=[1, 1_000_000],
    )
    model = LabelTablePrediction()
    torch.manual_seed(0)

    loss, metrics = objective(
        model=model,
        path=None,
        x0=torch.zeros(1, 2),
        x1=torch.zeros(1, 2),
        t=torch.tensor([2]),
        class_labels=torch.tensor([0]),
        original_class_labels=torch.tensor([0]),
    )
    loss.backward()

    assert torch.equal(model.seen_labels[0], torch.tensor([0]))
    assert torch.equal(model.seen_labels[1], torch.tensor([1]))
    assert torch.allclose(loss, torch.tensor(1.8))
    assert metrics["diffusion_loss"] == 1.0
    assert metrics["cbdm_regularizer"] == pytest.approx(0.8)
    assert metrics["cbdm_commitment"] == 0.0
    assert model.predictions.grad is not None
    assert torch.count_nonzero(model.predictions.grad[0]) == 2
    assert torch.count_nonzero(model.predictions.grad[1]) == 0


def test_cbdm_commitment_does_not_change_base_prediction_gradient() -> None:
    def gradients(gamma: float) -> torch.Tensor:
        objective = build_objective(
            {
                "name": "cbdm",
                "prediction_type": "epsilon",
                "cbdm": {"target_distribution": "train", "tau": 0.1, "gamma": gamma},
            },
            diffusion_config={"timesteps": 10},
            class_counts=[1, 1_000_000],
        )
        model = LabelTablePrediction()
        torch.manual_seed(0)
        loss, _ = objective(
            model=model,
            path=None,
            x0=torch.zeros(1, 2),
            x1=torch.zeros(1, 2),
            t=torch.tensor([2]),
            class_labels=torch.tensor([0]),
            original_class_labels=torch.tensor([0]),
        )
        loss.backward()
        assert model.predictions.grad is not None
        return model.predictions.grad

    without_commitment = gradients(0.0)
    with_commitment = gradients(0.5)

    assert torch.equal(with_commitment[0], without_commitment[0])
    assert torch.count_nonzero(with_commitment[1]) == 2


def test_cbdm_x_vloss_regularizes_predictions_in_epsilon_space() -> None:
    objective = build_objective(
        {
            "name": "cbdm",
            "prediction_type": "x_vloss",
            "cbdm": {"target_distribution": "train", "tau": 0.1, "gamma": 0.0},
        },
        diffusion_config={"timesteps": 10},
        class_counts=[1, 1_000_000],
    )
    model = LabelTablePrediction()
    with torch.no_grad():
        model.predictions[0].zero_()
        model.predictions[1].fill_(1.0)
    discrete_t = torch.tensor([2])
    torch.manual_seed(0)

    loss, _ = objective(
        model=model,
        path=None,
        x0=torch.zeros(1, 2),
        x1=torch.zeros(1, 2),
        t=discrete_t,
        class_labels=torch.tensor([0]),
        original_class_labels=torch.tensor([0]),
    )
    auxiliary_epsilon = objective.diffusion.predict_epsilon_from_x0(
        torch.zeros(1, 2), discrete_t, torch.ones(1, 2)
    )

    assert torch.allclose(loss, 0.2 * auxiliary_epsilon.square().mean())


def test_cbdm_rejects_missing_labels_or_class_counts() -> None:
    with pytest.raises(ValueError, match="class_counts"):
        build_objective({"name": "cbdm"}, diffusion_config={"timesteps": 10})

    objective = build_objective(
        {"name": "cbdm"},
        diffusion_config={"timesteps": 10},
        class_counts=[1, 1],
    )
    with pytest.raises(ValueError, match="class labels"):
        objective(
            model=FixedConditionalPrediction(torch.zeros(1, 2)),
            path=None,
            x0=torch.zeros(1, 2),
            x1=torch.zeros(1, 2),
            t=torch.tensor([1]),
        )
