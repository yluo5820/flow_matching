"""MLP velocity models for low-dimensional toy experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar flow time."""

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("Time embedding dim must be at least 2.")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError(f"Expected time tensor with shape (batch,), got {tuple(t.shape)}")

        half = self.dim // 2
        exponent = (
            -math.log(self.max_period)
            * torch.arange(half, device=t.device)
            / max(half - 1, 1)
        )
        freqs = torch.exp(exponent)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLPVelocity(nn.Module):
    """Small residual-free MLP for `v_theta(x, t)` on toy data."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        activation: str = "silu",
        time_embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)

        layers: list[nn.Module] = []
        input_dim = dim + time_embedding_dim
        for layer_idx in range(depth):
            layers.append(nn.Linear(input_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            layers.append(_activation(activation))
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del context
        time_features = self.time_embedding(t)
        return self.net(torch.cat([x, time_features], dim=1))
