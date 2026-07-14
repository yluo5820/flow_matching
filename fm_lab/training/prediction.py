"""Prediction-target adapters for training and sampling."""

from __future__ import annotations

import math

import torch
from torch import nn

from fm_lab.paths.base import ConvertibleFlowPath, FlowPath
from fm_lab.paths.prediction import PredictionKind, normalize_prediction_kind


def model_prediction(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    source_label: torch.Tensor | None = None,
    class_labels: torch.Tensor | None = None,
    use_capacity: bool | None = None,
) -> torch.Tensor:
    if bool(getattr(model, "requires_source_label", False)):
        if source_label is None:
            raise ValueError("Source-label-conditioned model requires source labels.")
        return model(x, t, context={"source_label": source_label})
    if bool(getattr(model, "is_class_conditional", False)):
        if class_labels is None:
            raise ValueError("Class-conditional model requires class labels.")
        context = {"class_labels": class_labels}
        if use_capacity is not None:
            context["use_capacity"] = use_capacity
        return model(x, t, context=context)
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


class VelocityFromPrediction(nn.Module):
    """Wrap a canonical path prediction so callers see a velocity field."""

    def __init__(
        self,
        model: nn.Module,
        path: FlowPath,
        *,
        model_output: str | PredictionKind,
        min_denom: float = 1e-3,
    ) -> None:
        super().__init__()
        self.model = model
        self.path = path
        self.model_output = normalize_prediction_kind(model_output)
        self.min_denom = float(min_denom)
        if not math.isfinite(self.min_denom) or self.min_denom <= 0:
            raise ValueError("objective.min_denom must be finite and positive")
        self.requires_source_label = bool(getattr(model, "requires_source_label", False))
        self.is_class_conditional = bool(getattr(model, "is_class_conditional", False))

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = None
        class_labels = None
        if context is not None:
            source_label = context.get("source_label")
            class_labels = context.get("class_labels")
        use_capacity = None if context is None else context.get("use_capacity")
        prediction = model_prediction(
            self.model,
            x,
            t,
            source_label=source_label,
            class_labels=class_labels,
            use_capacity=use_capacity,
        )
        if not isinstance(self.path, ConvertibleFlowPath):
            raise ValueError(
                "Prediction conversion requires a ConvertibleFlowPath with "
                "prediction_state()."
            )
        state = self.path.prediction_state(x, t, min_denom=self.min_denom)
        return state.prediction(prediction, self.model_output).as_velocity()


def velocity_model_for_objective(
    model: nn.Module,
    path: FlowPath,
    objective: object,
) -> nn.Module:
    output = output_kind_for_objective(objective)
    if output is PredictionKind.VELOCITY:
        return model
    min_denom = float(getattr(objective, "min_denom", 1e-3))
    return VelocityFromPrediction(
        model,
        path,
        model_output=output,
        min_denom=min_denom,
    )


def output_kind_for_objective(objective: object) -> PredictionKind:
    """Return the canonical meaning of a continuous objective's model output."""

    output = getattr(objective, "model_output", None)
    if output is None:
        output = getattr(objective, "prediction_type", None)
    if output is None:
        raise ValueError("Continuous objective is missing model output metadata.")
    return normalize_prediction_kind(output)
