"""Path-aware prediction value objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch


class PredictionKind(str, Enum):  # noqa: UP042 - required public compatibility
    """Canonical meanings for model predictions along a probability path."""

    SOURCE = "source"
    TARGET = "target"
    VELOCITY = "velocity"


def normalize_prediction_kind(value: str | PredictionKind) -> PredictionKind:
    """Normalize a prediction-kind alias to its canonical value."""

    if isinstance(value, PredictionKind):
        return value
    aliases = {
        "source": PredictionKind.SOURCE,
        "epsilon": PredictionKind.SOURCE,
        "noise": PredictionKind.SOURCE,
        "target": PredictionKind.TARGET,
        "x": PredictionKind.TARGET,
        "x1": PredictionKind.TARGET,
        "clean": PredictionKind.TARGET,
        "velocity": PredictionKind.VELOCITY,
        "v": PredictionKind.VELOCITY,
        "field": PredictionKind.VELOCITY,
    }
    try:
        return aliases[str(value).lower()]
    except KeyError as exc:
        raise ValueError(
            "prediction kind must be source, target, or velocity"
        ) from exc


@runtime_checkable
class PathPredictionState(Protocol):
    """Path state capable of converting between supported prediction kinds."""

    @property
    def supported_kinds(self) -> frozenset[PredictionKind]:
        """Return the prediction kinds supported by this state."""

    def prediction(
        self,
        value: torch.Tensor,
        kind: str | PredictionKind,
    ) -> PathPrediction:
        """Bind a prediction value and kind to this path state."""

    def convert(
        self,
        value: torch.Tensor,
        source_kind: PredictionKind,
        target_kind: PredictionKind,
    ) -> torch.Tensor:
        """Convert a prediction tensor between canonical kinds."""


@dataclass(frozen=True)
class PathPrediction:
    """An immutable prediction delegated to its originating path state."""

    value: torch.Tensor
    kind: PredictionKind
    state: PathPredictionState

    def convert(self, kind: str | PredictionKind) -> torch.Tensor:
        return self.state.convert(
            self.value,
            self.kind,
            normalize_prediction_kind(kind),
        )

    def as_source(self) -> torch.Tensor:
        return self.convert(PredictionKind.SOURCE)

    def as_target(self) -> torch.Tensor:
        return self.convert(PredictionKind.TARGET)

    def as_velocity(self) -> torch.Tensor:
        return self.convert(PredictionKind.VELOCITY)
