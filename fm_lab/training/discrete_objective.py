"""Training objectives for finite-step Gaussian diffusion."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fm_lab.diffusion import DiscreteDiffusion
from fm_lab.training.prediction import model_prediction


class DiscreteDiffusionObjective:
    """Epsilon parity or clean-image prediction with velocity-space MSE."""

    name = "discrete_diffusion"

    def __init__(
        self,
        *,
        prediction_type: str = "epsilon",
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        variance: str = "fixed_large",
    ) -> None:
        normalized = prediction_type.lower()
        if normalized not in {"epsilon", "x_vloss"}:
            raise ValueError(
                "Discrete diffusion prediction_type must be 'epsilon' or 'x_vloss'."
            )
        self.prediction_type = normalized
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.diffusion = DiscreteDiffusion(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            variance=variance,
        )

    def __call__(
        self,
        *,
        model: nn.Module,
        path: Any,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        compute_diagnostics: bool = True,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del path, compute_diagnostics
        discrete_t = (
            t.to(dtype=torch.long)
            if t.dtype in {torch.int32, torch.int64}
            else torch.randint(
                self.diffusion.timesteps,
                (x1.shape[0],),
                device=x1.device,
            )
        )
        noise = x0
        xt = self.diffusion.q_sample(x1, discrete_t, noise=noise)
        prediction = model_prediction(
            model,
            xt,
            discrete_t,
            class_labels=class_labels,
        )
        if self.prediction_type == "epsilon":
            loss = F.mse_loss(prediction, noise)
        else:
            predicted_epsilon = self.diffusion.predict_epsilon_from_x0(
                xt, discrete_t, prediction
            )
            predicted_velocity = self.diffusion.velocity_target(
                prediction, predicted_epsilon, discrete_t
            )
            target_velocity = self.diffusion.velocity_target(x1, noise, discrete_t)
            loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss, {"loss": float(loss.detach())}

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prediction_type": self.prediction_type,
            "loss": "mse",
            "timesteps": self.diffusion.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "variance": self.diffusion.variance,
        }
