import torch
from torch import nn

from fm_lab.diffusion import DiscreteDiffusion
from fm_lab.diffusion.sampling import (
    balanced_class_labels,
    paper_omega_to_guidance_scale,
    sample_discrete_diffusion,
)


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
