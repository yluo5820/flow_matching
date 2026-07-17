"""Reversible parameter perturbations for diagnostic probes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from fm_lab.diagnostics.probes.gradients import resolve_probe_layers


@contextmanager
def virtual_layer_update(
    model: nn.Module,
    *,
    layer_name: str,
    direction: torch.Tensor,
    relative_step: float,
) -> Iterator[float]:
    """Apply one relative layerwise update and restore the parameter bit-exactly."""

    if not np.isfinite(relative_step) or float(relative_step) <= 0:
        raise ValueError("Virtual-update relative_step must be positive and finite.")
    layer = resolve_probe_layers(model, (layer_name,))[0]
    flat_direction = direction.detach().reshape(-1).float().cpu()
    if flat_direction.numel() != layer.parameter.numel():
        raise ValueError("Virtual-update direction has the wrong shape.")
    if not torch.isfinite(flat_direction).all():
        raise ValueError("Virtual-update direction must be finite.")
    direction_norm = torch.linalg.vector_norm(flat_direction)
    if not torch.isclose(
        direction_norm,
        torch.ones_like(direction_norm),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Virtual-update direction must be unit norm.")
    original = layer.parameter.detach().clone()
    parameter_norm = torch.linalg.vector_norm(original)
    if not torch.isfinite(parameter_norm) or float(parameter_norm) == 0.0:
        raise ValueError("Virtual-update layer must have a finite nonzero norm.")
    applied_norm = float(relative_step) * float(parameter_norm)
    update = flat_direction.to(
        device=layer.parameter.device,
        dtype=layer.parameter.dtype,
    ).reshape(layer.shape)
    with torch.no_grad():
        layer.parameter.add_(update, alpha=applied_norm)
    try:
        yield applied_norm
    finally:
        with torch.no_grad():
            layer.parameter.copy_(original)
        if not torch.equal(layer.parameter.detach(), original):
            raise RuntimeError("Virtual update failed to restore the base parameter.")
