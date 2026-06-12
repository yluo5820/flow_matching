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


class DirectionSpeedMLP(nn.Module):
    """Label-conditioned direction/speed velocity model.

    The source label `a` fixes a direction n(a), while a scalar speed network
    predicts signed progress along that line.
    """

    requires_source_label = True

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        activation: str = "silu",
        time_embedding_dim: int = 64,
        direction_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.direction_eps = direction_eps
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.direction_net = _make_mlp(
            input_dim=dim,
            output_dim=dim,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
        )
        self.speed_net = _make_mlp(
            input_dim=2 * dim + time_embedding_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
        )

    def direction(self, source_label: torch.Tensor) -> torch.Tensor:
        raw_direction = self.direction_net(source_label)
        norm = raw_direction.norm(dim=1, keepdim=True)
        return raw_direction / (norm + self.direction_eps)

    def speed(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        source_label: torch.Tensor,
    ) -> torch.Tensor:
        time_features = self.time_embedding(t)
        return self.speed_net(torch.cat([x, source_label, time_features], dim=1)).squeeze(-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = _source_label_from_context(context)
        direction = self.direction(source_label)
        speed = self.speed(x, t, source_label)
        return speed[:, None] * direction


def _make_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for layer_idx in range(depth):
        layers.append(nn.Linear(input_dim if layer_idx == 0 else hidden_dim, hidden_dim))
        layers.append(_activation(activation))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _source_label_from_context(context) -> torch.Tensor:
    if isinstance(context, torch.Tensor):
        return context
    if isinstance(context, dict):
        for key in ("source_label", "x0", "a"):
            value = context.get(key)
            if value is not None:
                return value
    raise ValueError("DirectionSpeedMLP requires source_label context.")
