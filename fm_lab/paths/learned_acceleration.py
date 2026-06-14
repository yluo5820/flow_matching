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
        basis = _normalize_basis_name(basis)
        if basis not in {"quadratic", "endpoint_bump", "factorized_polynomial"}:
            raise ValueError(
                "LearnedAccelerationPath basis must be 'quadratic', 'endpoint_bump', "
                "or 'factorized_polynomial'."
            )

        self.dim = dim
        self.basis = basis
        self.eps = eps
        self.n_coefficients = 3 if basis == "factorized_polynomial" else 1

        layers: list[nn.Module] = []
        input_dim = 2 * dim
        for layer_idx in range(depth):
            layers.append(nn.Linear(input_dim if layer_idx == 0 else hidden_dim, hidden_dim))
            layers.append(_activation(activation))
        output = nn.Linear(hidden_dim, self.n_coefficients * dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def coefficients(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return learned correction coefficients with shape `(batch, terms, dim)`."""

        delta = x1 - x0
        raw_coefficients = self.net(torch.cat([x0, delta], dim=1))
        return raw_coefficients.reshape(x0.shape[0], self.n_coefficients, self.dim)

    def acceleration(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return learned correction coefficients, flattened for multi-term bases."""

        coefficients = self.coefficients(x0, x1)
        if self.n_coefficients == 1:
            return coefficients[:, 0, :]
        return coefficients.flatten(start_dim=1)

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
        correction = _combine_basis(basis.h, self.coefficients(x0, x1))
        return x0 + t_expanded * delta + correction

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
        correction_velocity = _combine_basis(basis.dh, self.coefficients(x0, x1))
        return delta + correction_velocity

    def conditional_acceleration(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return `d^2 I_psi / dt^2` for diagnostics."""

        t_expanded = expand_time(t, x0)
        basis = self._basis(t_expanded)
        return _combine_basis(basis.d2h, self.coefficients(x0, x1))

    def acceleration_penalty(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return `E ||A_psi(x0, x1)||^2`."""

        return self.coefficients(x0, x1).square().sum(dim=(1, 2)).mean()

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
            coefficients = self.coefficients(x0, x1)
            correction = _combine_basis(basis.h, coefficients)
            target_velocity = delta + _combine_basis(basis.dh, coefficients)
            conditional_acceleration = _combine_basis(basis.d2h, coefficients)
            delta_norm = delta.norm(dim=1)
            coefficient_norm = coefficients.flatten(start_dim=1).norm(dim=1)
            correction_norm = correction.norm(dim=1)
            target_velocity_norm = target_velocity.norm(dim=1)
            conditional_acceleration_norm = conditional_acceleration.norm(dim=1)

            t0 = torch.zeros_like(t)
            t1 = torch.ones_like(t)
            endpoint_error = torch.maximum(
                (self.sample_xt(x0, x1, t0) - x0).norm(dim=1),
                (self.sample_xt(x0, x1, t1) - x1).norm(dim=1),
            )
            relative_acceleration = coefficient_norm / (delta_norm + self.eps)
            relative_deviation = correction_norm / (delta_norm + self.eps)

            return {
                "interpolant_acceleration_norm_mean": _mean(coefficient_norm),
                "interpolant_acceleration_norm_p90": _quantile(coefficient_norm, 0.90),
                "interpolant_coefficient_norm_mean": _mean(coefficient_norm),
                "interpolant_coefficient_norm_p90": _quantile(coefficient_norm, 0.90),
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
            "n_coefficients": self.n_coefficients,
            "eps": self.eps,
        }

    def _basis(self, t: torch.Tensor) -> _BasisValues:
        if self.basis == "quadratic":
            return _BasisValues(
                h=(t * (1.0 - t))[:, None, :],
                dh=(1.0 - 2.0 * t)[:, None, :],
                d2h=torch.full_like(t, -2.0)[:, None, :],
            )
        if self.basis == "endpoint_bump":
            return _BasisValues(
                h=(t.square() * (1.0 - t).square())[:, None, :],
                dh=(2.0 * t * (1.0 - t) * (1.0 - 2.0 * t))[:, None, :],
                d2h=(2.0 - 12.0 * t + 12.0 * t.square())[:, None, :],
            )
        if self.basis == "factorized_polynomial":
            return _BasisValues(
                h=torch.stack(
                    (
                        t * (1.0 - t),
                        t.square() * (1.0 - t),
                        t.pow(3) * (1.0 - t),
                    ),
                    dim=1,
                ),
                dh=torch.stack(
                    (
                        1.0 - 2.0 * t,
                        2.0 * t - 3.0 * t.square(),
                        3.0 * t.square() - 4.0 * t.pow(3),
                    ),
                    dim=1,
                ),
                d2h=torch.stack(
                    (
                        torch.full_like(t, -2.0),
                        2.0 - 6.0 * t,
                        6.0 * t - 12.0 * t.square(),
                    ),
                    dim=1,
                ),
            )
        raise ValueError(f"Unsupported learned acceleration basis: {self.basis}")


def _combine_basis(weights: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    return (weights * coefficients).sum(dim=1)


def _normalize_basis_name(name: str) -> str:
    normalized = name.lower()
    aliases = {
        "full": "factorized_polynomial",
        "full_polynomial": "factorized_polynomial",
        "polynomial": "factorized_polynomial",
        "quartic": "factorized_polynomial",
        "t_factorized": "factorized_polynomial",
        "t_factorized_polynomial": "factorized_polynomial",
    }
    return aliases.get(normalized, normalized)


def _mean(values: torch.Tensor) -> float:
    return float(values.detach().float().mean().cpu())


def _quantile(values: torch.Tensor, q: float) -> float:
    values_cpu = values.detach().float().flatten().cpu()
    if values_cpu.numel() == 0:
        return float("nan")
    return float(torch.quantile(values_cpu, q))
