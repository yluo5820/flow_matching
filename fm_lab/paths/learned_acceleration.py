"""Pair-dependent learned acceleration interpolants."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fm_lab.paths.base import expand_time


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


@dataclass(frozen=True)
class _BasisValues:
    h: torch.Tensor
    dh: torch.Tensor
    d2h: torch.Tensor


class LearnedAccelerationPath(nn.Module):
    """Low-order learned interpolant `x0 + t Delta + h(t) A_psi(x0, x1)`."""

    name = "learned_acceleration"

    def __init__(
        self,
        dim: int,
        *,
        basis: str = "quadratic",
        hidden_dim: int = 128,
        depth: int = 3,
        activation: str = "silu",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("LearnedAccelerationPath dim must be positive.")
        if hidden_dim < 1:
            raise ValueError("LearnedAccelerationPath hidden_dim must be positive.")
        if depth < 1:
            raise ValueError("LearnedAccelerationPath depth must be positive.")
        basis = basis.lower()
        if basis not in {"quadratic", "endpoint_bump"}:
            raise ValueError(
                "LearnedAccelerationPath basis must be 'quadratic' or 'endpoint_bump'."
            )

        self.dim = dim
        self.basis = basis
        self.eps = eps

        layers: list[nn.Module] = []
        input_dim = 2 * dim
        for layer_idx in range(depth):
            layers.append(nn.Linear(input_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            layers.append(_activation(activation))
        output = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def acceleration(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return the pair-dependent learned acceleration mode `A_psi`."""

        delta = x1 - x0
        return self.net(torch.cat([x0, delta], dim=1))

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        t_expanded = expand_time(t, x0)
        basis = self._basis(t_expanded)
        delta = x1 - x0
        return x0 + t_expanded * delta + basis.h * self.acceleration(x0, x1)

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        t_expanded = expand_time(t, x0)
        basis = self._basis(t_expanded)
        delta = x1 - x0
        return delta + basis.dh * self.acceleration(x0, x1)

    def conditional_acceleration(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return `d^2 I_psi / dt^2` for diagnostics."""

        t_expanded = expand_time(t, x0)
        basis = self._basis(t_expanded)
        return basis.d2h * self.acceleration(x0, x1)

    def acceleration_penalty(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return `E ||A_psi(x0, x1)||^2`."""

        return self.acceleration(x0, x1).square().sum(dim=1).mean()

    def diagnostics(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, float]:
        """Summarize path geometry and degeneracy checks for logging."""

        with torch.no_grad():
            t_expanded = expand_time(t, x0)
            basis = self._basis(t_expanded)
            delta = x1 - x0
            acceleration = self.acceleration(x0, x1)
            correction = basis.h * acceleration
            target_velocity = delta + basis.dh * acceleration
            conditional_acceleration = basis.d2h * acceleration
            delta_norm = delta.norm(dim=1)
            acceleration_norm = acceleration.norm(dim=1)
            correction_norm = correction.norm(dim=1)
            target_velocity_norm = target_velocity.norm(dim=1)
            conditional_acceleration_norm = conditional_acceleration.norm(dim=1)

            t0 = torch.zeros_like(t)
            t1 = torch.ones_like(t)
            endpoint_error = torch.maximum(
                (self.sample_xt(x0, x1, t0) - x0).norm(dim=1),
                (self.sample_xt(x0, x1, t1) - x1).norm(dim=1),
            )
            relative_acceleration = acceleration_norm / (delta_norm + self.eps)
            relative_deviation = correction_norm / (delta_norm + self.eps)

            return {
                "interpolant_acceleration_norm_mean": _mean(acceleration_norm),
                "interpolant_acceleration_norm_p90": _quantile(acceleration_norm, 0.90),
                "interpolant_relative_acceleration_mean": _mean(relative_acceleration),
                "interpolant_relative_acceleration_p90": _quantile(
                    relative_acceleration,
                    0.90,
                ),
                "interpolant_path_deviation_mean": _mean(correction_norm),
                "interpolant_path_deviation_p90": _quantile(correction_norm, 0.90),
                "interpolant_relative_deviation_mean": _mean(relative_deviation),
                "interpolant_relative_deviation_p90": _quantile(relative_deviation, 0.90),
                "interpolant_target_velocity_norm_mean": _mean(target_velocity_norm),
                "interpolant_target_velocity_norm_p90": _quantile(
                    target_velocity_norm,
                    0.90,
                ),
                "interpolant_conditional_acceleration_norm_mean": _mean(
                    conditional_acceleration_norm
                ),
                "interpolant_endpoint_error_max": float(endpoint_error.max().cpu()),
            }

    def metadata(self) -> dict[str, str | int | float]:
        return {
            "name": self.name,
            "basis": self.basis,
            "dim": self.dim,
            "eps": self.eps,
        }

    def _basis(self, t: torch.Tensor) -> _BasisValues:
        if self.basis == "quadratic":
            return _BasisValues(
                h=t * (1.0 - t),
                dh=1.0 - 2.0 * t,
                d2h=torch.full_like(t, -2.0),
            )
        if self.basis == "endpoint_bump":
            return _BasisValues(
                h=t.square() * (1.0 - t).square(),
                dh=2.0 * t * (1.0 - t) * (1.0 - 2.0 * t),
                d2h=2.0 - 12.0 * t + 12.0 * t.square(),
            )
        raise ValueError(f"Unsupported learned acceleration basis: {self.basis}")


def _mean(values: torch.Tensor) -> float:
    return float(values.detach().float().mean().cpu())


def _quantile(values: torch.Tensor, q: float) -> float:
    values_cpu = values.detach().float().flatten().cpu()
    if values_cpu.numel() == 0:
        return float("nan")
    return float(torch.quantile(values_cpu, q))
