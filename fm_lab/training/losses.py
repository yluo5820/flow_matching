"""Flow matching training losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from fm_lab.paths.base import FlowPath


def sample_uniform_time(batch_size: int, device: torch.device, eps: float = 1e-5) -> torch.Tensor:
    """Sample times from `(eps, 1 - eps)` to avoid endpoint-only batches."""

    return eps + (1.0 - 2.0 * eps) * torch.rand(batch_size, device=device)


def flow_matching_loss(
    model: torch.nn.Module,
    path: FlowPath,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute MSE conditional flow matching loss."""

    xt = path.sample_xt(x0, x1, t)
    target_velocity = path.target_velocity(x0, x1, t)
    predicted_velocity = model(xt, t)
    loss = F.mse_loss(predicted_velocity, target_velocity)
    return loss, {"loss": float(loss.detach().cpu())}
