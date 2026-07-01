"""Gaussian diffusion interpolants for score/noise training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fm_lab.paths.base import expand_time


@dataclass(frozen=True)
class GaussianDiffusionSample:
    """Joint sample and training targets for a Gaussian diffusion path."""

    xt: torch.Tensor
    epsilon: torch.Tensor
    score_target: torch.Tensor
    velocity_target: torch.Tensor
    alpha_t: torch.Tensor
    sigma_t: torch.Tensor


@dataclass
class GaussianDiffusionPath:
    """Interpolant `x_t = alpha(t) x_1 + sigma(t) epsilon`.

    The trainer already samples a source batch `x0` and target batch `x1`.
    For this path, `x0` is interpreted as the Gaussian noise sample
    `epsilon`, while `x1` is the clean data sample.
    """

    schedule: str = "trig"
    sigma_min: float = 1e-4
    name: str = "gaussian_diffusion"

    def __post_init__(self) -> None:
        self.schedule = self.schedule.lower()
        if self.schedule not in {"linear", "trig", "cosine"}:
            raise ValueError("GaussianDiffusionPath schedule must be 'linear' or 'trig'.")
        if self.sigma_min <= 0:
            raise ValueError("GaussianDiffusionPath sigma_min must be positive.")

    def sample_training_tuple(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> GaussianDiffusionSample:
        """Return `x_t`, epsilon, score target, velocity target, alpha, and sigma."""

        del kwargs
        if x0.shape != x1.shape:
            raise ValueError("GaussianDiffusionPath requires x0 and x1 with matching shape.")

        epsilon = x0
        alpha_t, sigma_t, alpha_dot_t, sigma_dot_t = self._schedule(t)
        alpha = expand_time(alpha_t, x1)
        sigma = expand_time(sigma_t, x1)
        alpha_dot = expand_time(alpha_dot_t, x1)
        sigma_dot = expand_time(sigma_dot_t, x1)
        sigma_for_score = expand_time(sigma_t.clamp_min(self.sigma_min), x1)

        xt = alpha * x1 + sigma * epsilon
        score_target = -epsilon / sigma_for_score
        velocity_target = alpha_dot * x1 + sigma_dot * epsilon
        return GaussianDiffusionSample(
            xt=xt,
            epsilon=epsilon,
            score_target=score_target,
            velocity_target=velocity_target,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
        )

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.sample_training_tuple(x0, x1, t, **kwargs).xt

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.sample_training_tuple(x0, x1, t, **kwargs).velocity_target

    def metadata(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "schedule": self.schedule,
            "sigma_min": self.sigma_min,
        }

    def _schedule(
        self,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if t.ndim == 0:
            t = t[None]
        if self.schedule == "linear":
            alpha = t
            sigma = 1.0 - t
            alpha_dot = torch.ones_like(t)
            sigma_dot = -torch.ones_like(t)
            return alpha, sigma, alpha_dot, sigma_dot

        angle = 0.5 * math.pi * t
        alpha = torch.sin(angle)
        sigma = torch.cos(angle)
        alpha_dot = 0.5 * math.pi * torch.cos(angle)
        sigma_dot = -0.5 * math.pi * torch.sin(angle)
        return alpha, sigma, alpha_dot, sigma_dot
