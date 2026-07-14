"""Euclidean linear interpolation path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fm_lab.paths.base import expand_time
from fm_lab.paths.prediction import (
    PathPrediction,
    PredictionKind,
    normalize_prediction_kind,
)


@dataclass(frozen=True)
class LinearPredictionState:
    """Conversion state for a point on a linear interpolation path."""

    xt: torch.Tensor
    t: torch.Tensor
    min_denom: float = 1e-3

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_denom) or self.min_denom <= 0:
            raise ValueError("min_denom must be positive")
        try:
            torch.broadcast_shapes(expand_time(self.t, self.xt).shape, self.xt.shape)
        except RuntimeError as exc:
            raise ValueError("t must broadcast against xt") from exc

    @property
    def supported_kinds(self) -> frozenset[PredictionKind]:
        return frozenset(PredictionKind)

    def prediction(
        self,
        value: torch.Tensor,
        kind: str | PredictionKind,
    ) -> PathPrediction:
        normalized = normalize_prediction_kind(kind)
        if value.shape != self.xt.shape:
            raise ValueError("prediction value must match xt shape")
        return PathPrediction(value=value, kind=normalized, state=self)

    def convert(
        self,
        value: torch.Tensor,
        source_kind: PredictionKind,
        target_kind: PredictionKind,
    ) -> torch.Tensor:
        if source_kind is target_kind:
            return value
        t = expand_time(self.t, self.xt)
        if source_kind is PredictionKind.VELOCITY:
            return (
                self.xt - t * value
                if target_kind is PredictionKind.SOURCE
                else self.xt + (1 - t) * value
            )
        if source_kind is PredictionKind.TARGET:
            velocity = (value - self.xt) / (1 - t).clamp_min(self.min_denom)
        else:
            velocity = (self.xt - value) / t.clamp_min(self.min_denom)
        if target_kind is PredictionKind.VELOCITY:
            return velocity
        return (
            self.xt - t * velocity
            if target_kind is PredictionKind.SOURCE
            else self.xt + (1 - t) * velocity
        )


@dataclass
class LinearPath:
    name: str = "linear"

    def sample_xt(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        t_expanded = expand_time(t, x0)
        return (1.0 - t_expanded) * x0 + t_expanded * x1

    def target_velocity(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return x1 - x0

    def prediction_state(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        *,
        min_denom: float = 1e-3,
    ) -> LinearPredictionState:
        return LinearPredictionState(xt=xt, t=t, min_denom=min_denom)
