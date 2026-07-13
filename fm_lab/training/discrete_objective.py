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
        oc_transfer_mode: str = "t2h",
        oc_cut_time: int = -1,
    ) -> None:
        normalized = prediction_type.lower()
        if normalized not in {"epsilon", "x_vloss"}:
            raise ValueError(
                "Discrete diffusion prediction_type must be 'epsilon' or 'x_vloss'."
            )
        self.prediction_type = normalized
        self.method = method.lower()
        if self.method not in {"ddpm", "cbdm", "oc"}:
            raise ValueError("Discrete diffusion method must be 'ddpm', 'cbdm', or 'oc'.")
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
        self.oc_transfer_mode = oc_transfer_mode.lower()
        self.oc_cut_time = int(oc_cut_time)
        if self.method == "cbdm":
            self._validate_cbdm_config()
            self.auxiliary_probabilities = self._build_auxiliary_probabilities()
        else:
            self.auxiliary_probabilities = torch.empty(0)
        if self.method == "oc":
            self._validate_oc_config()

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
        if self.method == "cbdm" and class_labels is None:
            raise ValueError("CBDM requires class labels for conditional predictions.")
        target_clean = x1
        target_noise = noise
        oc_metrics: dict[str, float] = {}
        if self.method == "oc":
            if class_labels is None:
                raise ValueError("OC requires class labels for conditional predictions.")
            if original_class_labels is None:
                raise ValueError("OC requires original class labels before CFG dropout.")
            target_clean, target_noise, oc_metrics = self._oc_training_targets(
                xt=xt,
                clean=x1,
                noise=noise,
                discrete_t=discrete_t,
                original_labels=original_class_labels,
            )
        prediction = model_prediction(
            model,
            xt,
            discrete_t,
            class_labels=class_labels,
        )
        predicted_epsilon = self._as_epsilon(prediction, xt, discrete_t)
        if self.prediction_type == "epsilon":
            diffusion_loss = F.mse_loss(prediction, target_noise)
        else:
            predicted_velocity = self.diffusion.velocity_target(
                prediction, predicted_epsilon, discrete_t
            )
            target_velocity = self.diffusion.velocity_target(
                target_clean, target_noise, discrete_t
            )
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
        metrics.update(oc_metrics)
        metrics["loss"] = float(loss.detach())
        return loss, metrics

    def _validate_cbdm_config(self) -> None:
        self._validate_class_counts("CBDM")
        if self.cbdm_target_distribution not in {"train", "sqrt", "uniform"}:
            raise ValueError(
                "CBDM target_distribution must be 'train', 'sqrt', or 'uniform'."
            )
        if self.cbdm_tau < 0 or self.cbdm_gamma < 0:
            raise ValueError("CBDM tau and gamma must be non-negative.")

    def _validate_oc_config(self) -> None:
        self._validate_class_counts("OC")
        if self.oc_transfer_mode not in {"t2h", "h2t", "full"}:
            raise ValueError("OC transfer_mode must be 't2h', 'h2t', or 'full'.")
        if self.oc_cut_time < -1:
            raise ValueError("OC cut_time must be -1 or a non-negative timestep.")

    def _validate_class_counts(self, method: str) -> None:
        if not self.class_counts:
            raise ValueError(f"{method} requires class_counts from the training target.")
        if any(value <= 0 for value in self.class_counts):
            raise ValueError(f"{method} class_counts must all be positive.")

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

    @torch.no_grad()
    def _oc_reference_weights(
        self,
        *,
        noisy_clean: torch.Tensor,
        clean: torch.Tensor,
        discrete_t: torch.Tensor,
    ) -> torch.Tensor:
        noisy_flat = noisy_clean.flatten(1)
        clean_flat = clean.flatten(1)
        squared_distance = (
            noisy_flat.square().sum(dim=1, keepdim=True)
            + clean_flat.square().sum(dim=1).unsqueeze(0)
            - 2.0 * noisy_flat @ clean_flat.T
        ).clamp_min_(0.0)
        alpha_bar = self.diffusion.alpha_bars[discrete_t].to(
            device=clean.device,
            dtype=clean.dtype,
        )
        sigma_squared = 1.0 / alpha_bar - 1.0
        logits = -squared_distance / (2.0 * sigma_squared[:, None])
        logits = logits - logits.amax(dim=1, keepdim=True)
        return logits.softmax(dim=1)

    @torch.no_grad()
    def _oc_filter_reference_indices(
        self,
        *,
        candidate_indices: torch.Tensor,
        original_labels: torch.Tensor,
        discrete_t: torch.Tensor,
    ) -> torch.Tensor:
        identity = torch.arange(len(candidate_indices), device=candidate_indices.device)
        if self.oc_transfer_mode == "full":
            accepted = torch.ones_like(candidate_indices, dtype=torch.bool)
        else:
            counts = torch.tensor(
                self.class_counts,
                device=original_labels.device,
                dtype=torch.long,
            )
            old_counts = counts[original_labels]
            reference_labels = original_labels[candidate_indices]
            new_counts = counts[reference_labels]
            if self.oc_transfer_mode == "t2h":
                accepted = new_counts >= old_counts
            else:
                accepted = new_counts <= old_counts
        if self.oc_cut_time >= 0:
            accepted = accepted & (discrete_t < self.oc_cut_time)
        return torch.where(accepted, candidate_indices, identity)

    @torch.no_grad()
    def _oc_transfer_targets(
        self,
        *,
        xt: torch.Tensor,
        clean: torch.Tensor,
        discrete_t: torch.Tensor,
        reference_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transferred_clean = clean[reference_indices]
        transferred_epsilon = self.diffusion.predict_epsilon_from_x0(
            xt,
            discrete_t,
            transferred_clean,
        )
        return transferred_clean, transferred_epsilon

    @torch.no_grad()
    def _oc_training_targets(
        self,
        *,
        xt: torch.Tensor,
        clean: torch.Tensor,
        noise: torch.Tensor,
        discrete_t: torch.Tensor,
        original_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        alpha_bar = self.diffusion.alpha_bars[discrete_t].to(
            device=clean.device,
            dtype=clean.dtype,
        )
        shape = (len(clean),) + (1,) * (clean.ndim - 1)
        sigma = (1.0 / alpha_bar - 1.0).sqrt().reshape(shape)
        noisy_clean = clean + sigma * noise
        weights = self._oc_reference_weights(
            noisy_clean=noisy_clean,
            clean=clean,
            discrete_t=discrete_t,
        )
        candidates = torch.multinomial(weights, num_samples=1).squeeze(1)
        references = self._oc_filter_reference_indices(
            candidate_indices=candidates,
            original_labels=original_labels,
            discrete_t=discrete_t,
        )
        transferred_clean, transferred_epsilon = self._oc_transfer_targets(
            xt=xt,
            clean=clean,
            discrete_t=discrete_t,
            reference_indices=references,
        )
        metrics = self._oc_transfer_metrics(
            references=references,
            original_labels=original_labels,
            discrete_t=discrete_t,
        )
        return transferred_clean, transferred_epsilon, metrics

    def _oc_transfer_metrics(
        self,
        *,
        references: torch.Tensor,
        original_labels: torch.Tensor,
        discrete_t: torch.Tensor,
    ) -> dict[str, float]:
        identity = torch.arange(len(references), device=references.device)
        transferred = references != identity
        metrics = {"oc_transfer_rate": float(transferred.float().mean())}
        ordered_classes = torch.tensor(self.class_counts).argsort(descending=True)
        for name, classes in zip(
            ("many", "medium", "few"),
            torch.tensor_split(ordered_classes, 3),
            strict=True,
        ):
            mask = torch.isin(original_labels.cpu(), classes)
            metrics[f"oc_transfer_rate_{name}"] = self._masked_rate(transferred, mask)
        first_boundary = self.diffusion.timesteps // 3
        second_boundary = 2 * self.diffusion.timesteps // 3
        time_masks = {
            "early": discrete_t < first_boundary,
            "middle": (discrete_t >= first_boundary) & (discrete_t < second_boundary),
            "late": discrete_t >= second_boundary,
        }
        for name, mask in time_masks.items():
            metrics[f"oc_transfer_rate_{name}"] = self._masked_rate(transferred, mask)
        return metrics

    @staticmethod
    def _masked_rate(values: torch.Tensor, mask: torch.Tensor) -> float:
        mask = mask.to(device=values.device)
        if not bool(mask.any()):
            return 0.0
        return float(values[mask].float().mean())

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
        if self.method == "oc":
            metadata["oc"] = {
                "transfer_mode": self.oc_transfer_mode,
                "cut_time": self.oc_cut_time,
            }
        return metadata
