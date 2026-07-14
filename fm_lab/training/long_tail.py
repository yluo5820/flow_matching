"""Composable long-tail modifiers for continuous training objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from fm_lab.paths.base import FlowPath, expand_time
from fm_lab.paths.prediction import (
    PathPrediction,
    PathPredictionState,
    normalize_prediction_kind,
)
from fm_lab.training.prediction import model_prediction


@dataclass(frozen=True)
class ContinuousObjectiveContext:
    """Shared continuous objective values exposed to loss modifiers."""

    model: torch.nn.Module
    path: FlowPath
    state: PathPredictionState
    xt: torch.Tensor
    t: torch.Tensor
    class_labels: torch.Tensor | None
    original_class_labels: torch.Tensor | None
    base_prediction: PathPrediction
    source: torch.Tensor
    target: torch.Tensor
    base_loss_per_sample: torch.Tensor

    @property
    def observed_class_labels(self) -> torch.Tensor | None:
        """Return labels observed by the base conditional model evaluation."""

        return self.class_labels


class ContinuousObjectiveModifier(Protocol):
    """A loss term composed with a base continuous objective."""

    name: str

    def __call__(
        self,
        context: ContinuousObjectiveContext,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return a scalar modifier loss and namespaced logging metrics."""

    def metadata(self) -> dict[str, Any]:
        """Return a serializable modifier description."""


@dataclass
class CBDMModifier:
    """Class-balancing diffusion regularization at continuous time."""

    class_counts: Sequence[int]
    target_distribution: str = "train"
    tau: float = 0.001
    gamma: float = 0.25
    comparison_space: str = "velocity"
    name: str = "cbdm"

    def __post_init__(self) -> None:
        self.class_counts = tuple(int(value) for value in self.class_counts)
        if not self.class_counts:
            raise ValueError("CBDM requires class_counts from the training target.")
        if any(value <= 0 for value in self.class_counts):
            raise ValueError("CBDM class_counts must all be positive.")
        self.target_distribution = self.target_distribution.lower()
        if self.target_distribution not in {"train", "sqrt", "uniform"}:
            raise ValueError(
                "CBDM target_distribution must be 'train', 'sqrt', or 'uniform'."
            )
        self.tau = float(self.tau)
        self.gamma = float(self.gamma)
        if (
            not math.isfinite(self.tau)
            or not math.isfinite(self.gamma)
            or self.tau < 0
            or self.gamma < 0
        ):
            raise ValueError("CBDM tau and gamma must be finite and non-negative.")
        self.comparison_space = normalize_prediction_kind(self.comparison_space).value
        self.auxiliary_probabilities = self._build_auxiliary_probabilities()

    def _build_auxiliary_probabilities(self) -> torch.Tensor:
        counts = torch.tensor(self.class_counts, dtype=torch.float64)
        if self.target_distribution == "sqrt":
            counts = counts.sqrt()
        elif self.target_distribution == "uniform":
            counts = torch.ones_like(counts)
        return (counts / counts.sum()).to(dtype=torch.float32)

    @staticmethod
    def time_weight(t: torch.Tensor) -> torch.Tensor:
        """Weight high-noise samples most under the source-at-zero convention."""

        return 1.0 - t

    def __call__(
        self,
        context: ContinuousObjectiveContext,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if context.class_labels is None:
            raise ValueError("CBDM requires class labels for conditional predictions.")
        if self.tau == 0:
            zero = context.base_prediction.value.new_tensor(0.0)
            return zero, {
                "cbdm.regularizer": 0.0,
                "cbdm.commitment": 0.0,
                "cbdm.auxiliary_distribution": self.target_distribution,
            }

        auxiliary_labels = torch.multinomial(
            self.auxiliary_probabilities.to(device=context.xt.device),
            num_samples=context.xt.shape[0],
            replacement=True,
        )
        auxiliary_output = model_prediction(
            context.model,
            context.xt,
            context.t,
            class_labels=auxiliary_labels,
        )
        auxiliary_prediction = context.state.prediction(
            auxiliary_output,
            context.base_prediction.kind,
        )
        base_value = context.base_prediction.convert(self.comparison_space)
        auxiliary_value = auxiliary_prediction.convert(self.comparison_space)
        regularizer_distance = (
            (base_value - auxiliary_value.detach()).square().flatten(1).mean(1)
        )
        commitment_distance = (
            (base_value.detach() - auxiliary_value).square().flatten(1).mean(1)
        )
        weight = self.tau * self.time_weight(context.t).to(dtype=base_value.dtype)
        regularizer = (weight * regularizer_distance).mean()
        commitment = self.gamma * (weight * commitment_distance).mean()
        return regularizer + commitment, {
            "cbdm.regularizer": float(regularizer.detach().cpu()),
            "cbdm.commitment": float(commitment.detach().cpu()),
            "cbdm.auxiliary_distribution": self.target_distribution,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_distribution": self.target_distribution,
            "tau": self.tau,
            "gamma": self.gamma,
            "comparison_space": self.comparison_space,
            "class_counts": list(self.class_counts),
        }


@dataclass(frozen=True)
class TransferredTargets:
    """Source/target supervision selected by an endpoint-transfer modifier."""

    source: torch.Tensor
    target: torch.Tensor
    metrics: dict[str, float]


@dataclass
class OCModifier:
    """Online compensation target transfer for continuous linear paths."""

    class_counts: Sequence[int]
    transfer_mode: str = "t2h"
    cut_t: float | None = None
    min_denom: float = 1e-3
    name: str = "oc"

    def __post_init__(self) -> None:
        self.class_counts = tuple(int(value) for value in self.class_counts)
        if not self.class_counts or any(value <= 0 for value in self.class_counts):
            raise ValueError("OC class_counts must all be positive.")
        self.transfer_mode = self.transfer_mode.lower()
        if self.transfer_mode not in {"t2h", "h2t", "full"}:
            raise ValueError("OC transfer_mode must be 't2h', 'h2t', or 'full'.")
        if self.cut_t is not None:
            self.cut_t = float(self.cut_t)
            if not math.isfinite(self.cut_t) or not 0.0 <= self.cut_t <= 1.0:
                raise ValueError("OC cut_t must be null or a finite value in [0, 1].")
        self.min_denom = float(self.min_denom)
        if not math.isfinite(self.min_denom) or self.min_denom <= 0:
            raise ValueError("OC min_denom must be finite and positive.")

    def apply_cutoff(
        self,
        accepted: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the noisy-source-to-clean-target continuous cutoff."""

        if self.cut_t is None:
            return accepted
        return accepted & (t >= self.cut_t)

    @torch.no_grad()
    def reference_weights(
        self,
        *,
        noisy_target: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return stabilized reference probabilities under linear path geometry."""

        noisy_flat = noisy_target.flatten(1)
        target_flat = target.flatten(1)
        squared_distance = (
            noisy_flat.square().sum(dim=1, keepdim=True)
            + target_flat.square().sum(dim=1).unsqueeze(0)
            - 2.0 * noisy_flat @ target_flat.T
        ).clamp_min_(0.0)
        t_safe = t.clamp_min(self.min_denom)
        noise_to_signal_sq = ((1.0 - t) / t_safe).square()
        logits = -squared_distance / (2.0 * noise_to_signal_sq[:, None].clamp_min(self.min_denom))
        logits = logits - logits.amax(dim=1, keepdim=True)
        return logits.softmax(dim=1)

    @torch.no_grad()
    def filter_reference_indices(
        self,
        *,
        candidate_indices: torch.Tensor,
        original_labels: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Reject transfers that violate frequency direction or cutoff."""

        identity = torch.arange(len(candidate_indices), device=candidate_indices.device)
        if self.transfer_mode == "full":
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
            if self.transfer_mode == "t2h":
                accepted = new_counts >= old_counts
            else:
                accepted = new_counts <= old_counts
        accepted = self.apply_cutoff(accepted, t)
        return torch.where(accepted, candidate_indices, identity)

    @torch.no_grad()
    def transfer_targets(
        self,
        context: ContinuousObjectiveContext,
    ) -> TransferredTargets:
        """Select paired source/target supervision without changing the sampled input."""

        if context.path.name != "linear":
            raise ValueError("Continuous OC target transfer requires a linear path.")
        if context.original_class_labels is None:
            raise ValueError("Continuous OC target transfer requires original class labels.")
        t_expanded = expand_time(context.t.clamp_min(self.min_denom), context.xt)
        noisy_target = context.xt / t_expanded
        weights = self.reference_weights(
            noisy_target=noisy_target,
            target=context.target,
            t=context.t,
        )
        candidates = torch.multinomial(weights, num_samples=1).squeeze(1)
        references = self.filter_reference_indices(
            candidate_indices=candidates,
            original_labels=context.original_class_labels,
            t=context.t,
        )
        return TransferredTargets(
            source=context.source[references],
            target=context.target[references],
            metrics=self._transfer_metrics(references, context.t),
        )

    def _transfer_metrics(
        self,
        references: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, float]:
        identity = torch.arange(len(references), device=references.device)
        transferred = references != identity
        masks = {
            "noisy": t < (1.0 / 3.0),
            "middle": (t >= (1.0 / 3.0)) & (t < (2.0 / 3.0)),
            "clean": t >= (2.0 / 3.0),
        }
        metrics = {"oc.transfer_rate": float(transferred.float().mean().cpu())}
        for name, mask in masks.items():
            metrics[f"oc.transfer_rate.{name}"] = self._masked_rate(
                transferred,
                mask,
            )
        return metrics

    @staticmethod
    def _masked_rate(values: torch.Tensor, mask: torch.Tensor) -> float:
        mask = mask.to(device=values.device)
        if not bool(mask.any()):
            return 0.0
        return float(values[mask].float().mean().cpu())

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "transfer_mode": self.transfer_mode,
            "cut_t": self.cut_t,
            "min_denom": self.min_denom,
            "class_counts": list(self.class_counts),
        }


def build_continuous_modifiers(
    configs: Sequence[Mapping[str, Any]] | None,
    class_counts: Sequence[int] | None,
) -> tuple[ContinuousObjectiveModifier, ...]:
    """Build continuous modifiers in declared order."""

    configs = () if configs is None else configs
    normalized_configs: list[tuple[str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for config in configs:
        if not isinstance(config, Mapping):
            raise ValueError("Each continuous modifier config must be a mapping.")
        name = str(config.get("name", "")).lower()
        if name in seen:
            raise ValueError(f"Duplicate continuous modifier: {name}")
        seen.add(name)
        if name not in {"cbdm", "oc", "cm"}:
            raise ValueError(
                f"Unsupported continuous modifier: {name}. "
                "Supported values are cbdm, oc, and cm."
            )
        normalized_configs.append((name, config))

    if normalized_configs:
        if class_counts is None or not tuple(class_counts):
            raise ValueError(
                "Continuous long-tail modifiers require class_counts from the training target."
            )
        normalized_counts = tuple(int(value) for value in class_counts)
        if any(value <= 0 for value in normalized_counts):
            raise ValueError("Continuous modifier class_counts must all be positive.")
    else:
        normalized_counts = ()

    modifiers: list[ContinuousObjectiveModifier] = []
    for name, config in normalized_configs:
        if name == "cbdm":
            modifiers.append(
                CBDMModifier(
                    class_counts=normalized_counts,
                    target_distribution=str(config.get("target_distribution", "train")),
                    tau=float(config.get("tau", 0.001)),
                    gamma=float(config.get("gamma", 0.25)),
                    comparison_space=str(config.get("comparison_space", "velocity")),
                )
            )
            continue
        if name == "oc":
            modifiers.append(
                OCModifier(
                    class_counts=normalized_counts,
                    transfer_mode=str(config.get("transfer_mode", "t2h")),
                    cut_t=config.get("cut_t"),
                    min_denom=float(config.get("min_denom", 1e-3)),
                )
            )
            continue
        raise NotImplementedError(
            f"Continuous {name.upper()} modifier is recognized but not implemented yet."
        )
    return tuple(modifiers)
