"""Sampling loops for discrete DDPM and DDIM models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from fm_lab.diffusion.discrete import DiscreteDiffusion
from fm_lab.training.prediction import classifier_free_guided_prediction


def paper_omega_to_guidance_scale(omega: float) -> float:
    """Convert `cond + omega*(cond-uncond)` to the repository CFG scale."""

    return 1.0 + float(omega)


def balanced_class_labels(
    n_samples: int,
    *,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    if num_classes < 1:
        raise ValueError("num_classes must be positive.")
    return torch.arange(n_samples, device=device, dtype=torch.long) % num_classes


@torch.no_grad()
def sample_discrete_diffusion(
    *,
    model: nn.Module,
    diffusion: DiscreteDiffusion,
    sample_shape: Sequence[int],
    class_labels: torch.Tensor,
    prediction_type: str,
    sampler: str = "ddim",
    guidance_scale: float = 1.0,
    ddim_skip: int = 20,
    eta: float = 0.0,
    clip_x0: bool = True,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = tuple(int(value) for value in sample_shape)
    if len(shape) < 2 or shape[0] != class_labels.shape[0]:
        raise ValueError("sample_shape batch must match class_labels.")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative.")
    normalized_sampler = sampler.lower()
    if normalized_sampler not in {"ddpm", "ddim"}:
        raise ValueError("sampler must be 'ddpm' or 'ddim'.")
    if ddim_skip < 1:
        raise ValueError("ddim_skip must be positive.")

    device = class_labels.device
    samples = (
        torch.randn(shape, device=device)
        if initial_noise is None
        else initial_noise.to(device=device).clone()
    )
    if samples.shape != shape:
        raise ValueError("initial_noise must match sample_shape.")
    diffusion_prediction_type = "x" if prediction_type == "x_vloss" else prediction_type
    if diffusion_prediction_type not in {"epsilon", "x", "velocity"}:
        raise ValueError("Unsupported discrete sampling prediction_type.")

    was_training = model.training
    model.eval()
    try:
        if normalized_sampler == "ddpm":
            timestep_values = list(range(diffusion.timesteps - 1, -1, -1))
        else:
            timestep_values = list(range(diffusion.timesteps - 1, -1, -ddim_skip))
            if timestep_values[-1] != 0:
                timestep_values.append(0)

        for index, timestep in enumerate(timestep_values):
            t = torch.full(
                (shape[0],), timestep, device=device, dtype=torch.long
            )
            prediction = classifier_free_guided_prediction(
                model,
                samples,
                t,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
            )
            if normalized_sampler == "ddpm":
                samples = diffusion.p_sample(
                    samples,
                    t,
                    prediction,
                    prediction_type=diffusion_prediction_type,
                    clip_x0=clip_x0,
                )
            else:
                previous = timestep_values[index + 1] if index + 1 < len(timestep_values) else -1
                previous_t = torch.full_like(t, previous)
                samples = diffusion.ddim_step(
                    samples,
                    t,
                    previous_t,
                    prediction,
                    prediction_type=diffusion_prediction_type,
                    eta=eta,
                    clip_x0=clip_x0,
                )
    finally:
        model.train(was_training)
    return samples
