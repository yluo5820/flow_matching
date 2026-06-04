"""Base interfaces for target distributions."""

from __future__ import annotations

from typing import Protocol

import torch


class TargetDistribution(Protocol):
    """Distribution that provides target/data samples."""

    name: str
    dim: int

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Draw `n` samples with shape `(n, dim)`."""

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        """Return log probability when analytically available."""

    def metadata(self) -> dict:
        """Return serializable distribution metadata."""
