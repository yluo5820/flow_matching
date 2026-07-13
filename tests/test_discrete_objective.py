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
