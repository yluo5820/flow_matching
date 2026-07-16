"""Exact microbatch-gradient rows for selected model layers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.manifest import ProbeBatch


@dataclass(frozen=True)
class ProbeLayer:
    name: str
    parameter: nn.Parameter
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GradientRows:
    raw: torch.Tensor
    norms: torch.Tensor
    normalized: torch.Tensor


def resolve_probe_layers(
    model: nn.Module,
    names: Sequence[str],
) -> tuple[ProbeLayer, ...]:
    """Resolve exact trainable weight parameters in requested order."""

    if not names:
        raise ValueError("At least one probe layer is required.")
    parameters = dict(model.named_parameters())
    seen: set[int] = set()
    resolved: list[ProbeLayer] = []
    for raw_name in names:
        name = str(raw_name)
        if not name.endswith(".weight"):
            raise ValueError(f"Probe layer must name a .weight parameter: {name}")
        if name not in parameters:
            raise ValueError(f"Probe layer parameter does not exist: {name}")
        parameter = parameters[name]
        if not parameter.requires_grad:
            raise ValueError(f"Probe layer parameter is frozen: {name}")
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"Probe layer list contains a duplicate parameter: {name}")
        seen.add(identity)
        resolved.append(
            ProbeLayer(
                name=name,
                parameter=parameter,
                shape=tuple(parameter.shape),
            )
        )
    return tuple(resolved)


def collect_gradient_rows(
    *,
    model: nn.Module,
    objective: Any,
    path: Any,
    batches: Iterable[ProbeBatch],
    layer_names: Sequence[str],
) -> dict[str, GradientRows]:
    """Collect one ordinary-objective gradient row per fixed microbatch."""

    if getattr(objective, "modifiers", ()):
        raise ValueError("Gradient probes must use the ordinary objective without modifiers.")
    layers = resolve_probe_layers(model, layer_names)
    rows_by_layer: dict[str, list[torch.Tensor]] = {
        layer.name: [] for layer in layers
    }
    was_training = model.training
    model.eval()
    try:
        for batch_index, batch in enumerate(batches):
            loss, _ = objective(
                model=model,
                path=path,
                x0=batch.x0,
                x1=batch.x1,
                t=batch.t,
                compute_diagnostics=False,
                class_labels=batch.labels,
                original_class_labels=batch.labels,
            )
            gradients = torch.autograd.grad(
                loss,
                tuple(layer.parameter for layer in layers),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            for layer, gradient in zip(layers, gradients, strict=True):
                row = gradient.detach().reshape(-1).float().cpu()
                if not torch.isfinite(row).all():
                    raise ValueError(
                        f"Non-finite gradient for {layer.name} in batch {batch_index}."
                    )
                norm = torch.linalg.vector_norm(row)
                if not torch.isfinite(norm) or float(norm) == 0.0:
                    raise ValueError(
                        f"Zero gradient for {layer.name} in batch {batch_index}."
                    )
                rows_by_layer[layer.name].append(row)
    finally:
        model.train(was_training)

    if not rows_by_layer[layers[0].name]:
        raise ValueError("Gradient probes require at least one batch.")
    results: dict[str, GradientRows] = {}
    for layer in layers:
        raw = torch.stack(rows_by_layer[layer.name]).contiguous()
        norms = torch.linalg.vector_norm(raw, dim=1)
        results[layer.name] = GradientRows(
            raw=raw,
            norms=norms,
            normalized=raw / norms[:, None],
        )
    return results
