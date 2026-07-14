"""Validated training-time sampling policies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TrainingTimeSampler:
    """Sample continuous training times from a configured distribution."""

    name: str = "uniform"
    mean: float = -0.8
    std: float = 0.8
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.name not in {"uniform", "logit_normal"}:
            raise ValueError(
                "training.time_sampling.name must be 'uniform' or 'logit_normal'"
            )
        if not math.isfinite(self.eps) or not 0.0 <= self.eps < 0.5:
            raise ValueError("training.time_sampling.eps must be finite and in [0, 0.5)")
        if not math.isfinite(self.mean):
            raise ValueError("training.time_sampling.mean must be finite")
        if not math.isfinite(self.std) or self.std <= 0.0:
            raise ValueError("training.time_sampling.std must be finite and positive")

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one time per batch element on ``device``."""

        if self.name == "uniform":
            samples = torch.rand(batch_size, device=device, generator=generator)
            return self.eps + (1.0 - 2.0 * self.eps) * samples
        return torch.randn(
            batch_size, device=device, generator=generator
        ).mul(self.std).add(self.mean).sigmoid()


def build_training_time_sampler(
    config: str | Mapping[str, Any] | None,
) -> TrainingTimeSampler:
    """Build a validated sampler from legacy string or mapping configuration."""

    if config is None:
        return TrainingTimeSampler()
    if isinstance(config, str):
        if config != "uniform":
            raise ValueError(
                "training.time_sampling string form only supports 'uniform'"
            )
        return TrainingTimeSampler()
    if not isinstance(config, Mapping):
        raise ValueError("training.time_sampling must be a string or mapping")

    name = config.get("name", "uniform")
    allowed_keys = {"name", "eps"}
    if name == "logit_normal":
        allowed_keys.update({"mean", "std"})
    unsupported = sorted(set(config) - allowed_keys)
    if unsupported:
        raise ValueError(
            "training.time_sampling has unsupported keys: " + ", ".join(unsupported)
        )
    return TrainingTimeSampler(
        name=str(name),
        mean=float(config.get("mean", -0.8)),
        std=float(config.get("std", 0.8)),
        eps=float(config.get("eps", 1e-5)),
    )
