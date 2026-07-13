"""Shared discrete DDPM and DDIM equations."""

from __future__ import annotations

import torch


class DiscreteDiffusion:
    """Parameter-free linear-beta diffusion process."""

    def __init__(
        self,
        *,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        variance: str = "fixed_large",
    ) -> None:
        if timesteps < 2:
            raise ValueError("diffusion.timesteps must be at least 2.")
        if not 0.0 < beta_start <= beta_end < 1.0:
            raise ValueError("Diffusion betas must satisfy 0 < beta_start <= beta_end < 1.")
        if variance not in {"fixed_large", "fixed_small"}:
            raise ValueError("diffusion.variance must be 'fixed_large' or 'fixed_small'.")

        self.timesteps = int(timesteps)
        self.variance = variance
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat(
            (torch.ones(1, dtype=torch.float64), self.alpha_bars[:-1])
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef1 = (
            self.betas * self.alpha_bars_prev.sqrt() / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bars_prev)
            * self.alphas.sqrt()
            / (1.0 - self.alpha_bars)
        )
        self.fixed_large_variance = torch.cat(
            (self.posterior_variance[1:2], self.betas[1:])
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        self._validate_shapes(x0, t, noise)
        alpha_bar = self._extract(self.alpha_bars, t, x0)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def predict_x0_from_epsilon(
        self, xt: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, t, xt)
        return (xt - (1.0 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt()

    def predict_epsilon_from_x0(
        self, xt: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, t, xt)
        return (xt - alpha_bar.sqrt() * x0) / (1.0 - alpha_bar).sqrt()

    def velocity_target(
        self, x0: torch.Tensor, epsilon: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, t, x0)
        return alpha_bar.sqrt() * epsilon - (1.0 - alpha_bar).sqrt() * x0

    def predict_x0_from_velocity(
        self, xt: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        alpha_bar = self._extract(self.alpha_bars, t, xt)
        return alpha_bar.sqrt() * xt - (1.0 - alpha_bar).sqrt() * velocity

    def q_posterior(
        self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = (
            self._extract(self.posterior_mean_coef1, t, xt) * x0
            + self._extract(self.posterior_mean_coef2, t, xt) * xt
        )
        variance = self._extract(self.posterior_variance, t, xt)
        log_variance = variance.clamp_min(1e-20).log()
        return mean, variance, log_variance

    def p_sample(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor,
        *,
        prediction_type: str = "epsilon",
        noise: torch.Tensor | None = None,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        x0 = self._prediction_to_x0(xt, t, prediction, prediction_type)
        if clip_x0:
            x0 = x0.clamp(-1.0, 1.0)
        mean, _, _ = self.q_posterior(x0, xt, t)
        variance_values = (
            self.fixed_large_variance
            if self.variance == "fixed_large"
            else self.posterior_variance
        )
        variance = self._extract(variance_values, t, xt)
        noise = torch.randn_like(xt) if noise is None else noise
        nonzero = (t != 0).to(xt.dtype).reshape((-1,) + (1,) * (xt.ndim - 1))
        return mean + nonzero * variance.sqrt() * noise

    def ddim_step(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        previous_t: torch.Tensor,
        prediction: torch.Tensor,
        *,
        prediction_type: str = "epsilon",
        eta: float = 0.0,
        noise: torch.Tensor | None = None,
        clip_x0: bool = True,
    ) -> torch.Tensor:
        if eta < 0:
            raise ValueError("DDIM eta must be non-negative.")
        x0 = self._prediction_to_x0(xt, t, prediction, prediction_type)
        if clip_x0:
            x0 = x0.clamp(-1.0, 1.0)
        epsilon = self.predict_epsilon_from_x0(xt, t, x0)
        alpha_bar_t = self._extract(self.alpha_bars, t, xt)
        alpha_bar_prev = self._extract_previous_alpha_bar(previous_t, xt)
        sigma = eta * torch.sqrt(
            ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * (1.0 - alpha_bar_t / alpha_bar_prev)
        )
        direction = torch.sqrt((1.0 - alpha_bar_prev - sigma.square()).clamp_min(0.0))
        noise = torch.randn_like(xt) if noise is None else noise
        return alpha_bar_prev.sqrt() * x0 + direction * epsilon + sigma * noise

    def _prediction_to_x0(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        prediction: torch.Tensor,
        prediction_type: str,
    ) -> torch.Tensor:
        normalized = prediction_type.lower()
        if normalized == "epsilon":
            return self.predict_x0_from_epsilon(xt, t, prediction)
        if normalized in {"x", "x0"}:
            return prediction
        if normalized in {"velocity", "v"}:
            return self.predict_x0_from_velocity(xt, t, prediction)
        raise ValueError("prediction_type must be 'epsilon', 'x', or 'velocity'.")

    def _extract(
        self, values: torch.Tensor, t: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        self._validate_timesteps(t)
        selected = values.to(device=t.device)[t].to(dtype=reference.dtype)
        return selected.reshape((-1,) + (1,) * (reference.ndim - 1))

    def _extract_previous_alpha_bar(
        self, previous_t: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        if previous_t.ndim != 1 or previous_t.shape[0] != reference.shape[0]:
            raise ValueError("previous_t must have shape [batch].")
        if torch.any(previous_t < -1) or torch.any(previous_t >= self.timesteps):
            raise ValueError("previous_t values must be in [-1, timesteps).")
        result = torch.ones(previous_t.shape[0], device=previous_t.device, dtype=reference.dtype)
        mask = previous_t >= 0
        if mask.any():
            result[mask] = self.alpha_bars.to(previous_t.device)[previous_t[mask]].to(
                reference.dtype
            )
        return result.reshape((-1,) + (1,) * (reference.ndim - 1))

    def _validate_timesteps(self, t: torch.Tensor) -> None:
        if t.ndim != 1 or t.dtype not in {torch.int32, torch.int64}:
            raise ValueError("t must be a one-dimensional integer tensor.")
        if torch.any(t < 0) or torch.any(t >= self.timesteps):
            raise ValueError("t values must be in [0, timesteps).")

    def _validate_shapes(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> None:
        self._validate_timesteps(t)
        if t.shape[0] != x0.shape[0]:
            raise ValueError("t batch size must match x0.")
        if noise.shape != x0.shape:
            raise ValueError("noise must have the same shape as x0.")
