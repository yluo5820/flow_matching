"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and Torch when Torch is available."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    state: dict[str, Any] = {
        "seed": seed,
        "python": True,
        "numpy": True,
        "torch": False,
        "deterministic": deterministic,
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment.
        state["torch_error"] = repr(exc)
        return state

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    state["torch"] = True
    state["cuda_available"] = bool(torch.cuda.is_available())
    return state
