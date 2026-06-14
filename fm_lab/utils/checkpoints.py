"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer | None] | None,
    path_module: nn.Module | None = None,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Save model, optimizer, config, and metrics to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "config": config,
        "metrics": metrics,
    }
    if optimizer is not None:
        if isinstance(optimizer, dict):
            payload["optimizer_state_dict"] = {
                name: value.state_dict()
                for name, value in optimizer.items()
                if value is not None
            }
        else:
            payload["optimizer_state_dict"] = optimizer.state_dict()
    if path_module is not None:
        payload["path_state_dict"] = path_module.state_dict()
    torch.save(payload, output_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload."""

    return torch.load(Path(path), map_location=map_location)
