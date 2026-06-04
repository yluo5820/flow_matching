"""Gaussian source distributions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianSource:
    dim: int = 2
    std: float = 1.0
    mean: float = 0.0
    name: str = "gaussian"

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = torch.device("cpu" if device is None else device)
        return self.mean + self.std * torch.randn(n, self.dim, device=device)

    def metadata(self) -> dict:
        return {"name": self.name, "dim": self.dim, "std": self.std, "mean": self.mean}
