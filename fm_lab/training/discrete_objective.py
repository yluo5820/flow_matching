"""Training objectives for finite-step Gaussian diffusion."""

from __future__ import annotations

from collections.abc import Sequence
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
        method: str = "ddpm",
        class_counts: Sequence[int] | None = None,
        cbdm_target_distribution: str = "train",
        cbdm_tau: float = 0.001,
        cbdm_gamma: float = 0.25,
    ) -> None:
        normalized = prediction_type.lower()
        if normalized not in {"epsilon", "x_vloss"}:
            raise ValueError(
                "Discrete diffusion prediction_type must be 'epsilon' or 'x_vloss'."
            )
        self.prediction_type = normalized
        self.method = method.lower()
        if self.method not in {"ddpm", "cbdm"}:
            raise ValueError("Discrete diffusion method must be 'ddpm' or 'cbdm'.")
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.diffusion = DiscreteDiffusion(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            variance=variance,
        )
        self.class_counts = tuple(int(value) for value in (class_counts or ()))
        self.cbdm_target_distribution = cbdm_target_distribution.lower()
        self.cbdm_tau = float(cbdm_tau)
        self.cbdm_gamma = float(cbdm_gamma)
        if self.method == "cbdm":
            self._validate_cbdm_config()
            self.auxiliary_probabilities = self._build_auxiliary_probabilities()
        else:
            self.auxiliary_probabilities = torch.empty(0)

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
        original_class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del path, compute_diagnostics, original_class_labels
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
        if self.method == "cbdm" and class_labels is None:
            raise ValueError("CBDM requires class labels for conditional predictions.")
        prediction = model_prediction(
            model,
            xt,
            discrete_t,
            class_labels=class_labels,
        )
        predicted_epsilon = self._as_epsilon(prediction, xt, discrete_t)
        if self.prediction_type == "epsilon":
            diffusion_loss = F.mse_loss(prediction, noise)
        else:
            predicted_velocity = self.diffusion.velocity_target(
                prediction, predicted_epsilon, discrete_t
            )
            target_velocity = self.diffusion.velocity_target(x1, noise, discrete_t)
            diffusion_loss = F.mse_loss(predicted_velocity, target_velocity)
        loss = diffusion_loss
        metrics = {"diffusion_loss": float(diffusion_loss.detach())}
        if self.method == "cbdm":
            regularizer, commitment = self._cbdm_losses(
                model=model,
                xt=xt,
                discrete_t=discrete_t,
                prediction=predicted_epsilon,
            )
            loss = loss + regularizer + commitment
            metrics["cbdm_regularizer"] = float(regularizer.detach())
            metrics["cbdm_commitment"] = float(commitment.detach())
        metrics["loss"] = float(loss.detach())
        return loss, metrics

    def _validate_cbdm_config(self) -> None:
        if not self.class_counts:
            raise ValueError("CBDM requires class_counts from the training target.")
        if any(value <= 0 for value in self.class_counts):
            raise ValueError("CBDM class_counts must all be positive.")
        if self.cbdm_target_distribution not in {"train", "sqrt", "uniform"}:
            raise ValueError(
                "CBDM target_distribution must be 'train', 'sqrt', or 'uniform'."
            )
        if self.cbdm_tau < 0 or self.cbdm_gamma < 0:
            raise ValueError("CBDM tau and gamma must be non-negative.")

    def _build_auxiliary_probabilities(self) -> torch.Tensor:
        counts = torch.tensor(self.class_counts, dtype=torch.float64)
        if self.cbdm_target_distribution == "sqrt":
            counts = counts.sqrt()
        elif self.cbdm_target_distribution == "uniform":
            counts = torch.ones_like(counts)
        return (counts / counts.sum()).to(dtype=torch.float32)

    def _as_epsilon(
        self,
        prediction: torch.Tensor,
        xt: torch.Tensor,
        discrete_t: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return prediction
        return self.diffusion.predict_epsilon_from_x0(xt, discrete_t, prediction)

    def _cbdm_losses(
        self,
        *,
        model: nn.Module,
        xt: torch.Tensor,
        discrete_t: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cbdm_tau == 0:
            zero = prediction.new_tensor(0.0)
            return zero, zero
        probabilities = self.auxiliary_probabilities.to(device=xt.device)
        auxiliary_labels = torch.multinomial(
            probabilities,
            num_samples=xt.shape[0],
            replacement=True,
        )
        auxiliary_output = model_prediction(
            model,
            xt,
            discrete_t,
            class_labels=auxiliary_labels,
        )
        auxiliary_prediction = self._as_epsilon(auxiliary_output, xt, discrete_t)
        timestep_weight = self.cbdm_tau * discrete_t.to(dtype=prediction.dtype)
        first = (prediction - auxiliary_prediction.detach()).square().flatten(1).mean(1)
        second = (prediction.detach() - auxiliary_prediction).square().flatten(1).mean(1)
        regularizer = (timestep_weight * first).mean()
        commitment = self.cbdm_gamma * (timestep_weight * second).mean()
        return regularizer, commitment

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "name": self.name,
            "prediction_type": self.prediction_type,
            "loss": "mse",
            "timesteps": self.diffusion.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "variance": self.diffusion.variance,
        }
        if self.method == "cbdm":
            metadata["cbdm"] = {
                "target_distribution": self.cbdm_target_distribution,
                "tau": self.cbdm_tau,
                "gamma": self.cbdm_gamma,
                "class_counts": list(self.class_counts),
            }
        return metadata
