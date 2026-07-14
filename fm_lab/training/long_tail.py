"""Composable long-tail modifiers for continuous training objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from fm_lab.paths.base import FlowPath
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
        raise NotImplementedError(
            f"Continuous {name.upper()} modifier is recognized but not implemented yet."
        )
    return tuple(modifiers)
