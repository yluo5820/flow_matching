"""Checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def capture_rng_state() -> dict[str, Any]:
    """Capture all random generators used by the training loop."""

    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    ema_model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer | None] | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    path_module: nn.Module | None = None,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    rng_state: dict[str, Any] | None = None,
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
                name: value.state_dict() for name, value in optimizer.items() if value is not None
            }
        else:
            payload["optimizer_state_dict"] = optimizer.state_dict()
    if path_module is not None:
        payload["path_state_dict"] = path_module.state_dict()
    if ema_model is not None:
        payload["ema_model_state_dict"] = ema_model.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if history is not None:
        payload["history"] = history
    if rng_state is not None:
        payload["rng_state_dict"] = rng_state
    torch.save(payload, output_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload."""

    return torch.load(Path(path), map_location=map_location)
