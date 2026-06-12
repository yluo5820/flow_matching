"""Flow matching training losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

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

    return FlowMatchingObjective()(model=model, path=path, x0=x0, x1=x1, t=t)


class TrainingObjective(Protocol):
    name: str

    def __call__(
        self,
        *,
        model: torch.nn.Module,
        path: FlowPath,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute a scalar training loss and detached logging metrics."""

    def metadata(self) -> dict[str, Any]:
        """Return a serializable objective description."""


@dataclass
class FlowMatchingObjective:
    """Conditional flow matching objective with optional learned-flow regularizers."""

    loss: str = "mse"
    straightness_weight: float = 0.0
    straightness_sample_size: int | None = None
    name: str = "flow_matching"

    def __call__(
        self,
        *,
        model: torch.nn.Module,
        path: FlowPath,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        xt = path.sample_xt(x0, x1, t)
        target_velocity = path.target_velocity(x0, x1, t)
        predicted_velocity = model(xt, t)
        matching_loss = _velocity_loss(predicted_velocity, target_velocity, self.loss)
        total_loss = matching_loss
        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "flow_matching_loss": float(matching_loss.detach().cpu()),
        }

        if self.straightness_weight > 0:
            straightness = learned_flow_straightness_loss(
                model=model,
                x=xt,
                t=t,
                sample_size=self.straightness_sample_size,
            )
            weighted_straightness = self.straightness_weight * straightness
            total_loss = total_loss + weighted_straightness
            metrics["loss"] = float(total_loss.detach().cpu())
            metrics["straightness_loss"] = float(straightness.detach().cpu())
            metrics["straightness_weighted"] = float(weighted_straightness.detach().cpu())

        return total_loss, metrics

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loss": self.loss,
            "straightness": {
                "weight": self.straightness_weight,
                "sample_size": self.straightness_sample_size,
            },
        }


def learned_flow_straightness_loss(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    sample_size: int | None = None,
) -> torch.Tensor:
    """Penalize learned material acceleration: d_t v + J_x v · v."""

    if sample_size is not None and sample_size < x.shape[0]:
        indices = torch.randperm(x.shape[0], device=x.device)[:sample_size]
        x = x[indices]
        t = t[indices]

    x_reg = x.detach().requires_grad_(True)
    t_reg = t.detach().requires_grad_(True)
    velocity = model(x_reg, t_reg)
    residual_components = []
    for component in range(velocity.shape[1]):
        grad_x, grad_t = torch.autograd.grad(
            velocity[:, component].sum(),
            (x_reg, t_reg),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        if grad_x is None:
            grad_x = torch.zeros_like(x_reg)
        if grad_t is None:
            grad_t = torch.zeros_like(t_reg)
        directional_derivative = (grad_x * velocity).sum(dim=1)
        residual_components.append(grad_t + directional_derivative)

    residual = torch.stack(residual_components, dim=1)
    return residual.square().sum(dim=1).mean()


def build_objective(config: dict[str, Any] | None = None) -> TrainingObjective:
    """Build a training objective from config."""

    config = {} if config is None else config
    name = str(config.get("name", "flow_matching")).lower()
    if name not in {"flow_matching", "conditional_flow_matching", "cfm"}:
        raise ValueError(f"Unsupported objective: {name}")

    straightness_config = config.get("straightness", {})
    straightness_weight = float(straightness_config.get("weight", 0.0))
    straightness_sample_size = straightness_config.get("sample_size")
    if straightness_weight < 0:
        raise ValueError("objective.straightness.weight must be non-negative.")
    if straightness_sample_size is not None:
        straightness_sample_size = int(straightness_sample_size)
        if straightness_sample_size < 1:
            raise ValueError("objective.straightness.sample_size must be positive.")

    return FlowMatchingObjective(
        loss=str(config.get("loss", "mse")).lower(),
        straightness_weight=straightness_weight,
        straightness_sample_size=straightness_sample_size,
        name=name,
    )


def _velocity_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    loss: str,
) -> torch.Tensor:
    if loss == "mse":
        return F.mse_loss(predicted_velocity, target_velocity)
    raise ValueError(f"Unsupported velocity loss: {loss}")
