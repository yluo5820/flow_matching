"""Linear algebra helpers for diagnostics."""

from __future__ import annotations

import torch


def svdvals(matrix: torch.Tensor) -> torch.Tensor:
    """Compute singular values, using CPU for MPS tensors when needed."""

    if matrix.device.type == "mps":
        return torch.linalg.svdvals(matrix.cpu()).to(matrix.device)
    return torch.linalg.svdvals(matrix)
