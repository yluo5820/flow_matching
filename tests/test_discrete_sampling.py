import numpy as np
import pytest
import torch
from torch import nn

from fm_lab.couplings import IndependentCoupling
from fm_lab.diffusion import DiscreteDiffusion
from fm_lab.diffusion.sampling import (
    balanced_class_labels,
    paper_omega_to_guidance_scale,
    sample_discrete_diffusion,
)
from fm_lab.paths import LinearPath
from fm_lab.solvers import EulerSolver
from fm_lab.training.trainer import train_flow_matching
from fm_lab.utils.checkpoints import load_checkpoint


class RecordingConditionalModel(nn.Module):
    is_class_conditional = True

    def __init__(self) -> None:
        super().__init__()
        self.labels: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        self.labels.append(context["class_labels"].detach().clone())
        self.timesteps.append(t.detach().clone())
        labels = context["class_labels"].to(x.dtype)
        return labels[:, None].expand_as(x) * 0.01


class TinyTrainableConditionalModel(nn.Module):
    is_class_conditional = True

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        return self.linear(torch.cat((x, t.float()[:, None] / 4.0), dim=1))


class TinySource:
    dim = 2

    def sample(self, n: int, device=None) -> torch.Tensor:
        return torch.randn(n, 2, device=device)

    def metadata(self) -> dict:
        return {"name": "tiny_source", "dim": 2}


class TinyLabeledTarget:
    dim = 2

    def sample(self, n: int, device=None) -> torch.Tensor:
        return self.sample_with_labels(n, device=device)[0]

    def sample_with_labels(self, n: int, device=None):
        return torch.rand(n, 2, device=device) * 2 - 1, torch.arange(n, device=device) % 2

    def metadata(self) -> dict:
        return {"name": "tiny_target", "dim": 2}


def _tiny_training_config(
    *,
    lr: float = 1e-3,
    ema_decay: float | None = 0.9,
) -> dict:
    training = {"steps": 1, "batch_size": 2, "lr": lr}
    if ema_decay is not None:
        training["ema_decay"] = ema_decay
    return {
        "experiment": {"seed": 7},
        "conditioning": {"enabled": True, "num_classes": 2, "dropout_probability": 0.1},
        "diffusion": {
            "timesteps": 4,
            "beta_start": 1e-4,
            "beta_end": 1e-2,
            "variance": "fixed_large",
        },
        "objective": {"name": "discrete_diffusion", "prediction_type": "epsilon"},
        "training": training,
        "sampling": {
            "n_samples": 4,
            "sample_batch_size": 2,
            "sampler": "ddim",
            "ddim_skip": 2,
            "eta": 0.0,
            "classes": [0, 1],
            "classifier_free_guidance": {
                "enabled": True,
                "convention": "fm_lab",
                "scale": 1.0,
            },
            "live_ema_comparison": {"enabled": True, "n_samples": 2},
        },
    }


def test_discrete_training_smoke_writes_generated_samples(tmp_path) -> None:
    config = _tiny_training_config()

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=TinyLabeledTarget(),
        source=TinySource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=TinyTrainableConditionalModel(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert metrics["sampling"]["sampler"] == "ddim"
    assert (tmp_path / "samples" / "ddim.npy").exists()
    assert (tmp_path / "samples" / "generated_labels.npy").exists()
    assert (tmp_path / "plots" / "generated_samples.png").exists()
    comparison = metrics["sampling"]["live_ema_comparison"]
    assert comparison["n_samples"] == 2
    assert (tmp_path / "samples" / "live_diagnostic.npy").exists()
    assert (tmp_path / "samples" / "ema_diagnostic.npy").exists()
    assert (tmp_path / "plots" / "live_vs_ema.png").exists()
    assert "ema_model_state_dict" in load_checkpoint(tmp_path / "checkpoint.pt")


def test_live_ema_comparison_reuses_initial_noise_for_equal_weights(tmp_path) -> None:
    config = _tiny_training_config(lr=0.0)
    config["sampling"]["live_ema_comparison"]["n_samples"] = 3

    train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=TinyLabeledTarget(),
        source=TinySource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=TinyTrainableConditionalModel(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    live = np.load(tmp_path / "samples" / "live_diagnostic.npy")
    ema = np.load(tmp_path / "samples" / "ema_diagnostic.npy")
    assert live.shape[0] == 3
    assert np.array_equal(live, ema)


def test_live_ema_comparison_requires_ema_model(tmp_path) -> None:
    config = _tiny_training_config(ema_decay=None)

    with pytest.raises(ValueError, match="live_ema_comparison requires EMA"):
        train_flow_matching(
            config=config,
            run_dir=tmp_path,
            target=TinyLabeledTarget(),
            source=TinySource(),
            coupling=IndependentCoupling(),
            path=LinearPath(),
            model=TinyTrainableConditionalModel(),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )


def test_paper_omega_conversion_matches_cfg_equations() -> None:
    assert paper_omega_to_guidance_scale(1.5) == 2.5
    assert paper_omega_to_guidance_scale(0.0) == 1.0


def test_balanced_labels_are_deterministic_and_nearly_equal() -> None:
    labels = balanced_class_labels(11, num_classes=3, device=torch.device("cpu"))

    assert torch.equal(labels, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]))
    assert torch.equal(torch.bincount(labels), torch.tensor([4, 4, 3]))


def test_ddpm_sampler_returns_flat_samples_and_forwards_cfg_labels() -> None:
    diffusion = DiscreteDiffusion(timesteps=4, beta_start=1e-4, beta_end=1e-2)
    model = RecordingConditionalModel()
    labels = torch.tensor([0, 2])

    samples = sample_discrete_diffusion(
        model=model,
        diffusion=diffusion,
        sample_shape=(2, 5),
        class_labels=labels,
        prediction_type="epsilon",
        sampler="ddpm",
        guidance_scale=2.5,
    )

    assert samples.shape == (2, 5)
    assert len(model.timesteps) == 4
    assert torch.equal(model.timesteps[0], torch.tensor([3, 3, 3, 3]))
    assert torch.equal(model.labels[0], torch.tensor([0, 2, -1, -1]))


def test_ddim_eta_zero_is_reproducible_and_uses_requested_skip() -> None:
    diffusion = DiscreteDiffusion(timesteps=10)
    labels = torch.tensor([1, 3])

    torch.manual_seed(7)
    first_model = RecordingConditionalModel()
    first = sample_discrete_diffusion(
        model=first_model,
        diffusion=diffusion,
        sample_shape=(2, 4),
        class_labels=labels,
        prediction_type="epsilon",
        sampler="ddim",
        ddim_skip=3,
        eta=0.0,
        guidance_scale=1.0,
    )
    torch.manual_seed(7)
    second = sample_discrete_diffusion(
        model=RecordingConditionalModel(),
        diffusion=diffusion,
        sample_shape=(2, 4),
        class_labels=labels,
        prediction_type="epsilon",
        sampler="ddim",
        ddim_skip=3,
        eta=0.0,
        guidance_scale=1.0,
    )

    assert torch.equal(first, second)
    assert [int(t[0]) for t in first_model.timesteps] == [9, 6, 3, 0]


def test_x_vloss_sampling_uses_clean_image_predictions() -> None:
    diffusion = DiscreteDiffusion(timesteps=4)

    samples = sample_discrete_diffusion(
        model=RecordingConditionalModel(),
        diffusion=diffusion,
        sample_shape=(2, 3),
        class_labels=torch.tensor([0, 1]),
        prediction_type="x_vloss",
        sampler="ddim",
        ddim_skip=2,
        guidance_scale=1.0,
    )

    assert samples.shape == (2, 3)
