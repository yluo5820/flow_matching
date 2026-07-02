"""Prediction-target adapters for training and sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fm_lab.paths.base import FlowPath, expand_time


def normalize_model_output(value: str | None) -> str:
    raw = "velocity" if value is None else str(value).lower()
    aliases = {
        "velocity": "velocity",
        "v": "velocity",
        "field": "velocity",
        "x": "x",
        "x1": "x",
        "data": "x",
        "clean": "x",
        "clean_data": "x",
        "clean_image": "x",
    }
    if raw not in aliases:
        raise ValueError("model_output must be 'velocity' or 'x'.")
    return aliases[raw]


def normalize_x_prediction_loss_space(value: str | None) -> str:
    raw = "clean" if value is None else str(value).lower()
    aliases = {
        "clean": "clean",
        "x": "clean",
        "data": "clean",
        "velocity": "velocity",
        "v": "velocity",
    }
    if raw not in aliases:
        raise ValueError("x_prediction.loss_space must be 'clean' or 'velocity'.")
    return aliases[raw]


def model_prediction(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    source_label: torch.Tensor | None = None,
) -> torch.Tensor:
    if bool(getattr(model, "requires_source_label", False)):
        if source_label is None:
            raise ValueError("Source-label-conditioned model requires source labels.")
        return model(x, t, context={"source_label": source_label})
    return model(x, t)


def x_prediction_to_velocity(
    x_prediction: torch.Tensor,
    xt: torch.Tensor,
    t: torch.Tensor,
    path: FlowPath,
    *,
    min_denom: float = 1e-3,
) -> torch.Tensor:
    if min_denom <= 0:
        raise ValueError("x_prediction.min_denom must be positive.")

    path_name = str(getattr(path, "name", path.__class__.__name__)).lower()
    if path_name == "linear":
        denom = expand_time((1.0 - t).clamp_min(min_denom), xt)
        return (x_prediction - xt) / denom

    schedule = getattr(path, "_schedule", None)
    if callable(schedule):
        alpha_t, sigma_t, alpha_dot_t, sigma_dot_t = schedule(t)
        alpha = expand_time(alpha_t, xt)
        sigma = expand_time(sigma_t.clamp_min(min_denom), xt)
        alpha_dot = expand_time(alpha_dot_t, xt)
        sigma_dot = expand_time(sigma_dot_t, xt)
        epsilon_prediction = (xt - alpha * x_prediction) / sigma
        return alpha_dot * x_prediction + sigma_dot * epsilon_prediction

    raise ValueError(
        "x-prediction velocity conversion currently supports linear paths "
        "and Gaussian diffusion paths."
    )


@dataclass(frozen=True)
class XPredictionConfig:
    loss_space: str = "clean"
    min_denom: float = 1e-3


class VelocityFromXPrediction(nn.Module):
    """Wrap an x-predicting model so callers see a velocity field."""

    def __init__(
        self,
        model: nn.Module,
        path: FlowPath,
        *,
        min_denom: float = 1e-3,
    ) -> None:
        super().__init__()
        self.model = model
        self.path = path
        self.min_denom = float(min_denom)
        self.requires_source_label = bool(getattr(model, "requires_source_label", False))

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = None
        if context is not None:
            source_label = context.get("source_label")
        x_prediction = model_prediction(
            self.model,
            x,
            t,
            source_label=source_label,
        )
        return x_prediction_to_velocity(
            x_prediction,
            x,
            t,
            self.path,
            min_denom=self.min_denom,
        )


def velocity_model_for_objective(
    model: nn.Module,
    path: FlowPath,
    objective: object,
) -> nn.Module:
    output = getattr(objective, "model_output", None)
    if output is None:
        output = getattr(objective, "prediction_type", "velocity")
    output_text = str(output).lower()
    if output_text not in {"x", "x1", "data", "clean", "clean_data", "clean_image"}:
        return model
    min_denom = float(getattr(objective, "x_prediction_min_denom", 1e-3))
    return VelocityFromXPrediction(model, path, min_denom=min_denom)
