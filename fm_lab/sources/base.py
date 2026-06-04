"""Base interfaces for source distributions."""

from __future__ import annotations

from typing import Protocol

import torch


class SourceDistribution(Protocol):
    """Distribution that provides source samples."""

    name: str
    dim: int

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        """Draw `n` samples with shape `(n, dim)`."""

    def metadata(self) -> dict:
        """Return serializable source metadata."""
