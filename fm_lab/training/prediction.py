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
    class_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    if bool(getattr(model, "requires_source_label", False)):
        if source_label is None:
            raise ValueError("Source-label-conditioned model requires source labels.")
        return model(x, t, context={"source_label": source_label})
    if bool(getattr(model, "is_class_conditional", False)):
        if class_labels is None:
            raise ValueError("Class-conditional model requires class labels.")
        return model(x, t, context={"class_labels": class_labels})
    return model(x, t)


def classifier_free_guided_prediction(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    class_labels: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Evaluate conditional and null predictions in one batched model call."""

    if not bool(getattr(model, "is_class_conditional", False)):
        raise ValueError("Classifier-free guidance requires a class-conditional model.")
    if guidance_scale == 1.0:
        return model_prediction(model, x, t, class_labels=class_labels)
    doubled_x = torch.cat([x, x], dim=0)
    doubled_t = torch.cat([t, t], dim=0)
    doubled_labels = torch.cat([class_labels, torch.full_like(class_labels, -1)], dim=0)
    conditional, unconditional = model_prediction(
        model,
        doubled_x,
        doubled_t,
        class_labels=doubled_labels,
    ).chunk(2, dim=0)
    return unconditional + guidance_scale * (conditional - unconditional)


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
        self.is_class_conditional = bool(getattr(model, "is_class_conditional", False))

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = None
        class_labels = None
        if context is not None:
            source_label = context.get("source_label")
            class_labels = context.get("class_labels")
        x_prediction = model_prediction(
            self.model,
            x,
            t,
            source_label=source_label,
            class_labels=class_labels,
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
