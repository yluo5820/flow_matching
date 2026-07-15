"""Flow matching training losses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch.nn import functional as F

from fm_lab.paths.base import ConvertibleFlowPath, FlowPath
from fm_lab.paths.prediction import PredictionKind, normalize_prediction_kind
from fm_lab.training.long_tail import (
    CMModifier,
    ContinuousEndpointTransferModifier,
    ContinuousModifier,
    ContinuousObjectiveContext,
    build_continuous_modifiers,
)
from fm_lab.training.prediction import (
    model_prediction,
    velocity_model_for_objective,
)
from fm_lab.training.time_sampling import TrainingTimeSampler


def sample_uniform_time(batch_size: int, device: torch.device, eps: float = 1e-5) -> torch.Tensor:
    """Sample times from `(eps, 1 - eps)` to avoid endpoint-only batches."""

    return TrainingTimeSampler(eps=eps).sample(batch_size, device)


def flow_matching_loss(
    model: torch.nn.Module,
    path: FlowPath,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
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
        compute_diagnostics: bool = True,
        class_labels: torch.Tensor | None = None,
        original_class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute a scalar training loss and detached logging metrics."""

    def metadata(self) -> dict[str, Any]:
        """Return a serializable objective description."""


@dataclass(frozen=True)
class KernelVStarConfig:
    """Configuration for low-dimensional kernel estimates of `v*_phi`."""

    mode: str | None = None
    estimator_size: int = 256
    query_size: int = 64
    bandwidth: str | float = "median"
    bandwidth_scale: float = 1.0
    min_bandwidth: float = 1e-3
    eps: float = 1e-8


@dataclass(frozen=True)
class KernelVStarEstimate:
    query_x: torch.Tensor
    query_t: torch.Tensor
    vstar: torch.Tensor
    metrics: dict[str, float]


@dataclass
class FlowMatchingObjective:
    """Conditional flow matching objective with optional learned-flow regularizers."""

    loss: str = "mse"
    model_output: str = "velocity"
    loss_space: str = "velocity"
    min_denom: float = 1e-3
    straightness_weight: float = 0.0
    straightness_sample_size: int | None = None
    interpolant_acceleration_weight: float = 0.0
    learned_interpolant: KernelVStarConfig = field(default_factory=KernelVStarConfig)
    modifiers: tuple[ContinuousModifier, ...] = field(default_factory=tuple)
    name: str = "flow_matching"

    def __post_init__(self) -> None:
        self.model_output = normalize_prediction_kind(self.model_output).value
        self.loss_space = normalize_prediction_kind(self.loss_space).value
        self.loss = self.loss.lower()
        if not math.isfinite(self.min_denom) or self.min_denom <= 0:
            raise ValueError("objective.min_denom must be finite and positive")

    def __call__(
        self,
        *,
        model: torch.nn.Module,
        path: FlowPath,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        compute_diagnostics: bool = True,
        class_labels: torch.Tensor | None = None,
        original_class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._loss(
            model=model,
            path=path,
            x0=x0,
            x1=x1,
            t=t,
            compute_diagnostics=compute_diagnostics,
            include_flow_matching=True,
            include_straightness=True,
            include_interpolant_acceleration=True,
            detach_path=False,
            straightness_detach_inputs=True,
            class_labels=class_labels,
            original_class_labels=original_class_labels,
        )

    def theta_update_loss(
        self,
        *,
        model: torch.nn.Module,
        path: FlowPath,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Loss for updating the Eulerian velocity model while freezing the path."""

        return self._loss(
            model=model,
            path=path,
            x0=x0,
            x1=x1,
            t=t,
            compute_diagnostics=False,
            include_flow_matching=True,
            include_straightness=True,
            include_interpolant_acceleration=False,
            detach_path=True,
            straightness_detach_inputs=True,
            class_labels=class_labels,
            original_class_labels=class_labels,
        )

    def psi_update_loss(
        self,
        *,
        model: torch.nn.Module,
        path: FlowPath,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Loss for updating the learned interpolant while treating the model as fixed."""

        return self._loss(
            model=model,
            path=path,
            x0=x0,
            x1=x1,
            t=t,
            compute_diagnostics=False,
            include_flow_matching=False,
            include_straightness=True,
            include_interpolant_acceleration=True,
            detach_path=False,
            straightness_detach_inputs=False,
            class_labels=class_labels,
            original_class_labels=class_labels,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loss": self.loss,
            "model_output": self.model_output,
            "loss_space": self.loss_space,
            "min_denom": self.min_denom,
            "straightness": {
                "weight": self.straightness_weight,
                "sample_size": self.straightness_sample_size,
            },
            "interpolant_acceleration": {
                "weight": self.interpolant_acceleration_weight,
            },
            "learned_interpolant": {
                "mode": self.learned_interpolant.mode,
                "estimator_size": self.learned_interpolant.estimator_size,
                "query_size": self.learned_interpolant.query_size,
                "bandwidth": self.learned_interpolant.bandwidth,
                "bandwidth_scale": self.learned_interpolant.bandwidth_scale,
                "min_bandwidth": self.learned_interpolant.min_bandwidth,
            },
            "modifiers": [modifier.metadata() for modifier in self.modifiers],
        }

    def _loss(
        self,
        *,
        model: torch.nn.Module,
        path: FlowPath,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        compute_diagnostics: bool,
        include_flow_matching: bool,
        include_straightness: bool,
        include_interpolant_acceleration: bool,
        detach_path: bool,
        straightness_detach_inputs: bool,
        class_labels: torch.Tensor | None,
        original_class_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        xt = path.sample_xt(x0, x1, t)
        target_velocity = path.target_velocity(x0, x1, t)
        if detach_path:
            xt = xt.detach()
            target_velocity = target_velocity.detach()

        total_loss = x0.new_tensor(0.0)
        metrics: dict[str, Any] = {
            "model_output": self.model_output,
            "loss_space": self.loss_space,
        }
        velocity_model = velocity_model_for_objective(model, path, self)
        if include_flow_matching:
            cm_enabled = any(isinstance(modifier, CMModifier) for modifier in self.modifiers)
            prediction = model_prediction(
                model,
                xt,
                t,
                class_labels=class_labels,
                use_capacity=True if cm_enabled else None,
            )
            state = None
            if (
                self.model_output == PredictionKind.VELOCITY.value
                and self.loss_space == PredictionKind.VELOCITY.value
                and not self.modifiers
            ):
                prediction_in_loss_space = prediction
                target_in_loss_space = target_velocity
            else:
                if not isinstance(path, ConvertibleFlowPath):
                    raise ValueError(
                        "Prediction conversion requires a ConvertibleFlowPath with "
                        "prediction_state()."
                    )
                state = path.prediction_state(xt, t, min_denom=self.min_denom)
                base_prediction = state.prediction(
                    prediction,
                    self.model_output,
                )
                prediction_in_loss_space = base_prediction.convert(self.loss_space)
                oc_modifiers = [
                    modifier
                    for modifier in self.modifiers
                    if isinstance(modifier, ContinuousEndpointTransferModifier)
                ]
                if oc_modifiers:
                    context = ContinuousObjectiveContext(
                        model=model,
                        path=path,
                        state=state,
                        xt=xt,
                        t=t,
                        class_labels=class_labels,
                        original_class_labels=original_class_labels,
                        base_prediction=base_prediction,
                        source=x0,
                        target=x1,
                        base_loss_per_sample=prediction_in_loss_space.new_zeros(
                            len(prediction_in_loss_space)
                        ),
                    )
                    transferred = oc_modifiers[0].transfer_targets(context)
                    supervision_source = transferred.source
                    supervision_target = transferred.target
                    metrics.update(transferred.metrics)
                    supervision_velocity = supervision_target - supervision_source
                else:
                    supervision_source = x0
                    supervision_target = x1
                    supervision_velocity = target_velocity
                supervision_by_kind = {
                    PredictionKind.SOURCE.value: supervision_source,
                    PredictionKind.TARGET.value: supervision_target,
                    PredictionKind.VELOCITY.value: supervision_velocity,
                }
                target_in_loss_space = state.prediction(
                    supervision_by_kind[self.model_output],
                    self.model_output,
                ).convert(self.loss_space)
            per_sample_loss = _prediction_loss_per_sample(
                prediction_in_loss_space,
                target_in_loss_space,
                self.loss,
            )
            matching_loss = per_sample_loss.mean()
            metrics["flow_matching_loss"] = float(matching_loss.detach().cpu())
            metrics["base.loss"] = float(matching_loss.detach().cpu())
            total_loss = total_loss + matching_loss
            if self.modifiers:
                assert state is not None
                context = ContinuousObjectiveContext(
                    model=model,
                    path=path,
                    state=state,
                    xt=xt,
                    t=t,
                    class_labels=class_labels,
                    original_class_labels=original_class_labels,
                    base_prediction=base_prediction,
                    source=supervision_source,
                    target=supervision_target,
                    base_loss_per_sample=per_sample_loss,
                )
                for modifier in self.modifiers:
                    if isinstance(modifier, ContinuousEndpointTransferModifier):
                        continue
                    modifier_loss, modifier_metrics = modifier(context)
                    total_loss = total_loss + modifier_loss
                    metrics.update(modifier_metrics)

        if include_straightness and self.straightness_weight > 0:
            if self.learned_interpolant.mode == "kernel_vstar":
                straightness, kernel_metrics = kernel_vstar_straightness_loss(
                    model=velocity_model,
                    path=path,
                    x0=x0,
                    x1=x1,
                    t=t,
                    config=self.learned_interpolant,
                    detach_inputs=straightness_detach_inputs,
                )
                metrics.update(kernel_metrics)
            else:
                straightness = learned_flow_straightness_loss(
                    model=velocity_model,
                    x=xt,
                    t=t,
                    sample_size=self.straightness_sample_size,
                    detach_inputs=straightness_detach_inputs,
                )
            weighted_straightness = self.straightness_weight * straightness
            total_loss = total_loss + weighted_straightness
            metrics["straightness_loss"] = float(straightness.detach().cpu())
            metrics["straightness_weighted"] = float(weighted_straightness.detach().cpu())
            if self.learned_interpolant.mode == "kernel_vstar":
                metrics["kernel_vstar_straightness_loss"] = metrics["straightness_loss"]
                metrics["kernel_vstar_straightness_weighted"] = metrics["straightness_weighted"]

        if include_interpolant_acceleration and self.interpolant_acceleration_weight > 0:
            acceleration = _interpolant_acceleration_loss(path=path, x0=x0, x1=x1)
            weighted_acceleration = self.interpolant_acceleration_weight * acceleration
            total_loss = total_loss + weighted_acceleration
            metrics["interpolant_acceleration_loss"] = float(acceleration.detach().cpu())
            metrics["interpolant_acceleration_weighted"] = float(
                weighted_acceleration.detach().cpu()
            )

        metrics["loss"] = float(total_loss.detach().cpu())
        if compute_diagnostics and hasattr(path, "diagnostics"):
            metrics.update(path.diagnostics(x0=x0, x1=x1, t=t))
        return total_loss, metrics


@dataclass
class DiffusionObjective:
    """Gaussian diffusion objective for epsilon, score, velocity, or clean-x prediction."""

    prediction_type: str = "epsilon"
    loss: str = "mse"
    name: str = "diffusion"

    def __post_init__(self) -> None:
        self.prediction_type = _normalize_diffusion_prediction_type(self.prediction_type)
        self.loss = self.loss.lower()
        if self.loss != "mse":
            raise ValueError("DiffusionObjective currently supports only mse loss.")

    def __call__(
        self,
        *,
        model: torch.nn.Module,
        path: FlowPath,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        compute_diagnostics: bool = True,
        class_labels: torch.Tensor | None = None,
        original_class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del original_class_labels
        if not hasattr(path, "sample_training_tuple"):
            raise ValueError("diffusion objective requires path.name: gaussian_diffusion.")

        sample = path.sample_training_tuple(x0, x1, t)
        target = self._target(sample)
        prediction = model_prediction(model, sample.xt, t, class_labels=class_labels)
        diffusion_loss = F.mse_loss(prediction, target)
        metrics = {
            "diffusion_loss": float(diffusion_loss.detach().cpu()),
            "diffusion_prediction_norm_mean": _mean_stat(prediction.detach().norm(dim=1)),
            "diffusion_target_norm_mean": _mean_stat(target.detach().norm(dim=1)),
            "diffusion_alpha_mean": _mean_stat(sample.alpha_t.detach()),
            "diffusion_sigma_mean": _mean_stat(sample.sigma_t.detach()),
            "loss": float(diffusion_loss.detach().cpu()),
        }
        if compute_diagnostics and hasattr(path, "diagnostics"):
            metrics.update(path.diagnostics(x0=x0, x1=x1, t=t))
        return diffusion_loss, metrics

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prediction_type": self.prediction_type,
            "loss": self.loss,
        }

    def _target(self, sample: Any) -> torch.Tensor:
        if self.prediction_type == "epsilon":
            return sample.epsilon
        if self.prediction_type == "score":
            return sample.score_target
        if self.prediction_type == "velocity":
            return sample.velocity_target
        if self.prediction_type == "x":
            return sample.x1
        raise ValueError(f"Unsupported diffusion prediction type: {self.prediction_type}")


@dataclass
class DirectionOnlyStraightObjective:
    """Label-conditioned direction-only straight flow objective."""

    direction_weight: float = 1.0
    speed_weight: float = 1.0
    eps: float = 1e-8
    name: str = "direction_only_straight"
    model_output: str = "velocity"
    loss_space: str = "velocity"

    def __call__(
        self,
        *,
        model: torch.nn.Module,
        path: FlowPath,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        compute_diagnostics: bool = True,
        class_labels: torch.Tensor | None = None,
        original_class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del class_labels, original_class_labels
        if not hasattr(model, "direction") or not hasattr(model, "speed"):
            raise ValueError("direction_only_straight objective requires DirectionSpeedMLP.")

        xt = path.sample_xt(x0, x1, t)
        target_velocity = path.target_velocity(x0, x1, t)
        direction = model.direction(x0)
        speed = model.speed(xt, t, x0)
        target_speed = (target_velocity * direction).sum(dim=1)
        predicted_velocity = speed[:, None] * direction

        target_norm_sq = target_velocity.square().sum(dim=1)
        cos2 = (target_speed.square() / (target_norm_sq + self.eps)).clamp(0.0, 1.0)
        direction_loss = (1.0 - cos2).mean()
        speed_loss = F.mse_loss(speed, target_speed)
        direction_weighted = self.direction_weight * direction_loss
        speed_weighted = self.speed_weight * speed_loss
        total_loss = direction_weighted + speed_weighted
        metrics = {"loss": float(total_loss.detach().cpu())}

        if compute_diagnostics:
            perpendicular = target_velocity - target_speed[:, None] * direction
            speed_abs = speed.detach().abs()
            metrics.update(
                {
                    "direction_loss": float(direction_loss.detach().cpu()),
                    "speed_loss": float(speed_loss.detach().cpu()),
                    "direction_weighted": float(direction_weighted.detach().cpu()),
                    "speed_weighted": float(speed_weighted.detach().cpu()),
                    "direction_speed_vector_mse": float(
                        F.mse_loss(predicted_velocity, target_velocity).detach().cpu()
                    ),
                    "direction_alignment_cos2_mean": _mean_stat(cos2),
                    "direction_alignment_cos2_p10": _quantile_stat(cos2, 0.10),
                    "direction_alignment_cos2_p50": _quantile_stat(cos2, 0.50),
                    "direction_alignment_cos2_p90": _quantile_stat(cos2, 0.90),
                    "perpendicular_residual_mean": float(
                        perpendicular.square().sum(dim=1).mean().detach().cpu()
                    ),
                    "speed_abs_mean": _mean_stat(speed_abs),
                    "speed_abs_p90": _quantile_stat(speed_abs, 0.90),
                    "direction_pairwise_abs_mean": _pairwise_abs_similarity_mean(direction),
                }
            )

        return total_loss, metrics

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_output": self.model_output,
            "loss_space": self.loss_space,
            "direction_weight": self.direction_weight,
            "speed_weight": self.speed_weight,
            "eps": self.eps,
            "speed": "signed",
        }


def learned_flow_straightness_loss(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    sample_size: int | None = None,
    detach_inputs: bool = True,
    advective_velocity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize learned material acceleration: d_t v + J_x v · v."""

    if sample_size is not None and sample_size < x.shape[0]:
        indices = torch.randperm(x.shape[0], device=x.device)[:sample_size]
        x = x[indices]
        t = t[indices]
        if advective_velocity is not None:
            advective_velocity = advective_velocity[indices]

    x_reg = x.detach().requires_grad_(True) if detach_inputs else x.requires_grad_(True)
    t_reg = t.detach().requires_grad_(True)
    velocity = model(x_reg, t_reg)
    if advective_velocity is None:
        advective = velocity
    else:
        advective = advective_velocity.detach() if detach_inputs else advective_velocity
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
        directional_derivative = (grad_x * advective).sum(dim=1)
        residual_components.append(grad_t + directional_derivative)

    residual = torch.stack(residual_components, dim=1)
    return residual.square().sum(dim=1).mean()


def kernel_vstar_estimate(
    *,
    path: FlowPath,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    config: KernelVStarConfig,
    query_indices: torch.Tensor | None = None,
    estimator_indices: torch.Tensor | None = None,
    detach_inputs: bool = True,
) -> KernelVStarEstimate:
    """Estimate `E[u_t | x_t = x]` by Gaussian-kernel conditional averaging."""

    batch_size = x0.shape[0]
    if batch_size < 1:
        raise ValueError("kernel_vstar_estimate requires a non-empty batch.")
    if query_indices is None:
        query_indices = torch.randperm(batch_size, device=x0.device)[
            : min(config.query_size, batch_size)
        ]
    if estimator_indices is None:
        estimator_indices = torch.randperm(batch_size, device=x0.device)[
            : min(config.estimator_size, batch_size)
        ]

    x0_query = x0[query_indices]
    x1_query = x1[query_indices]
    t_query = t[query_indices]
    query_x = path.sample_xt(x0_query, x1_query, t_query)

    x0_estimator = x0[estimator_indices]
    x1_estimator = x1[estimator_indices]
    query_count = query_indices.numel()
    estimator_count = estimator_indices.numel()
    dim = x0.shape[1]

    estimator_x0 = (
        x0_estimator[None, :, :]
        .expand(query_count, estimator_count, dim)
        .reshape(query_count * estimator_count, dim)
    )
    estimator_x1 = (
        x1_estimator[None, :, :]
        .expand(query_count, estimator_count, dim)
        .reshape(query_count * estimator_count, dim)
    )
    estimator_t = (
        t_query[:, None]
        .expand(query_count, estimator_count)
        .reshape(query_count * estimator_count)
    )
    estimator_xt = path.sample_xt(estimator_x0, estimator_x1, estimator_t).reshape(
        query_count,
        estimator_count,
        dim,
    )
    estimator_velocity = path.target_velocity(estimator_x0, estimator_x1, estimator_t).reshape(
        query_count,
        estimator_count,
        dim,
    )

    if detach_inputs:
        query_for_weights = query_x.detach()
        estimator_xt_for_weights = estimator_xt.detach()
        estimator_velocity_for_average = estimator_velocity.detach()
    else:
        query_for_weights = query_x
        estimator_xt_for_weights = estimator_xt
        estimator_velocity_for_average = estimator_velocity

    distance_sq = (estimator_xt_for_weights - query_for_weights[:, None, :]).square().sum(dim=2)
    bandwidth = _kernel_bandwidth(distance_sq, config)
    weights = torch.exp(-0.5 * distance_sq / bandwidth.square().clamp_min(config.eps))
    denominator = weights.sum(dim=1).clamp_min(config.eps)
    vstar = (weights[:, :, None] * estimator_velocity_for_average).sum(dim=1) / denominator[:, None]
    if detach_inputs:
        vstar = vstar.detach()

    effective_sample_size = denominator.square() / weights.square().sum(dim=1).clamp_min(
        config.eps
    )
    vstar_norm = vstar.detach().norm(dim=1)
    metrics = {
        "kernel_vstar_bandwidth": float(bandwidth.detach().cpu()),
        "kernel_vstar_effective_sample_size_mean": _mean_stat(effective_sample_size),
        "kernel_vstar_effective_sample_size_min": float(
            effective_sample_size.detach().min().cpu()
        ),
        "kernel_vstar_denominator_mean": _mean_stat(denominator),
        "kernel_vstar_denominator_min": float(denominator.detach().min().cpu()),
        "kernel_vstar_norm_mean": _mean_stat(vstar_norm),
        "kernel_vstar_norm_p90": _quantile_stat(vstar_norm, 0.90),
        "kernel_vstar_query_size": float(query_count),
        "kernel_vstar_estimator_size": float(estimator_count),
    }
    return KernelVStarEstimate(
        query_x=query_x,
        query_t=t_query,
        vstar=vstar,
        metrics=metrics,
    )


def kernel_vstar_straightness_loss(
    *,
    model: torch.nn.Module,
    path: FlowPath,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: torch.Tensor,
    config: KernelVStarConfig,
    detach_inputs: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    estimate = kernel_vstar_estimate(
        path=path,
        x0=x0,
        x1=x1,
        t=t,
        config=config,
        detach_inputs=detach_inputs,
    )
    loss = learned_flow_straightness_loss(
        model=model,
        x=estimate.query_x,
        t=estimate.query_t,
        detach_inputs=detach_inputs,
        advective_velocity=estimate.vstar,
    )
    return loss, estimate.metrics


def build_objective(
    config: dict[str, Any] | None = None,
    *,
    diffusion_config: dict[str, Any] | None = None,
    class_counts: Sequence[int] | None = None,
) -> TrainingObjective:
    """Build a training objective from config."""

    config = {} if config is None else config
    name = str(config.get("name", "flow_matching")).lower()
    diffusion_prediction_aliases = {
        "diffusion_epsilon": "epsilon",
        "epsilon_prediction": "epsilon",
        "noise_prediction": "epsilon",
        "diffusion_score": "score",
        "score_matching": "score",
        "diffusion_velocity": "velocity",
        "diffusion_x": "x",
        "x_prediction": "x",
        "clean_prediction": "x",
    }
    if name in {
        "diffusion",
        "gaussian_diffusion",
        "diffusion_objective",
        *diffusion_prediction_aliases.keys(),
    }:
        prediction_type = str(
            config.get("prediction_type", diffusion_prediction_aliases.get(name, "epsilon"))
        )
        return DiffusionObjective(
            prediction_type=prediction_type,
            loss=str(config.get("loss", "mse")).lower(),
            name=name,
        )
    if name in {"direction_only_straight", "direction_speed", "lagrangian_direction"}:
        direction_weight = float(config.get("direction_weight", 1.0))
        speed_weight = float(config.get("speed_weight", 1.0))
        if direction_weight < 0 or speed_weight < 0:
            raise ValueError("direction_only_straight weights must be non-negative.")
        return DirectionOnlyStraightObjective(
            direction_weight=direction_weight,
            speed_weight=speed_weight,
            eps=float(config.get("eps", 1e-8)),
            name=name,
        )
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

    interpolant_acceleration_config = config.get("interpolant_acceleration", {})
    interpolant_acceleration_weight = float(interpolant_acceleration_config.get("weight", 0.0))
    if interpolant_acceleration_weight < 0:
        raise ValueError("objective.interpolant_acceleration.weight must be non-negative.")
    learned_interpolant_config = _build_kernel_vstar_config(config.get("learned_interpolant", {}))
    modifiers = build_continuous_modifiers(config.get("modifiers", []), class_counts)

    return FlowMatchingObjective(
        loss=str(config.get("loss", "mse")).lower(),
        model_output=str(config.get("model_output", "velocity")),
        loss_space=str(config.get("loss_space", "velocity")),
        min_denom=float(config.get("min_denom", 1e-3)),
        straightness_weight=straightness_weight,
        straightness_sample_size=straightness_sample_size,
        interpolant_acceleration_weight=interpolant_acceleration_weight,
        learned_interpolant=learned_interpolant_config,
        modifiers=modifiers,
        name=name,
    )


def _build_kernel_vstar_config(config: dict[str, Any]) -> KernelVStarConfig:
    if not config:
        return KernelVStarConfig()
    mode = config.get("mode")
    if mode is not None:
        mode = str(mode).lower()
    if mode not in {None, "kernel_vstar"}:
        raise ValueError(f"Unsupported objective.learned_interpolant.mode: {mode}")

    estimator_size = int(config.get("estimator_size", 256))
    query_size = int(config.get("query_size", 64))
    bandwidth_value = config.get("bandwidth", "median")
    try:
        bandwidth: str | float = float(bandwidth_value)
    except (TypeError, ValueError):
        bandwidth = str(bandwidth_value).lower()
    if bandwidth != "median" and not isinstance(bandwidth, float):
        raise ValueError("objective.learned_interpolant.bandwidth must be 'median' or a float.")
    bandwidth_scale = float(config.get("bandwidth_scale", 1.0))
    min_bandwidth = float(config.get("min_bandwidth", 1e-3))
    if estimator_size < 1:
        raise ValueError("objective.learned_interpolant.estimator_size must be positive.")
    if query_size < 1:
        raise ValueError("objective.learned_interpolant.query_size must be positive.")
    if isinstance(bandwidth, float) and bandwidth <= 0:
        raise ValueError("objective.learned_interpolant.bandwidth must be positive.")
    if bandwidth_scale <= 0:
        raise ValueError("objective.learned_interpolant.bandwidth_scale must be positive.")
    if min_bandwidth <= 0:
        raise ValueError("objective.learned_interpolant.min_bandwidth must be positive.")
    return KernelVStarConfig(
        mode=mode,
        estimator_size=estimator_size,
        query_size=query_size,
        bandwidth=bandwidth,
        bandwidth_scale=bandwidth_scale,
        min_bandwidth=min_bandwidth,
    )


def _velocity_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    loss: str,
) -> torch.Tensor:
    if loss == "mse":
        return F.mse_loss(predicted_velocity, target_velocity)
    raise ValueError(f"Unsupported velocity loss: {loss}")


def _prediction_loss_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss: str,
) -> torch.Tensor:
    if loss == "mse":
        return (prediction - target).square().reshape(prediction.shape[0], -1).mean(dim=1)
    raise ValueError(f"Unsupported prediction loss: {loss}")


def _normalize_diffusion_prediction_type(prediction_type: str) -> str:
    normalized = prediction_type.lower()
    aliases = {
        "eps": "epsilon",
        "epsilon": "epsilon",
        "noise": "epsilon",
        "score": "score",
        "velocity": "velocity",
        "v": "velocity",
        "x": "x",
        "x0": "x",
        "x_start": "x",
        "data": "x",
        "clean": "x",
    }
    if normalized not in aliases:
        raise ValueError(
            "DiffusionObjective prediction_type must be epsilon, score, velocity, or x."
        )
    return aliases[normalized]


def _interpolant_acceleration_loss(
    *,
    path: FlowPath,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(path, "acceleration_penalty"):
        raise ValueError(
            "objective.interpolant_acceleration.weight requires a path with "
            "acceleration_penalty, such as path.name: learned_acceleration."
        )
    return path.acceleration_penalty(x0, x1)


def _kernel_bandwidth(distance_sq: torch.Tensor, config: KernelVStarConfig) -> torch.Tensor:
    if isinstance(config.bandwidth, float):
        bandwidth = max(config.bandwidth * config.bandwidth_scale, config.min_bandwidth)
        return distance_sq.new_tensor(bandwidth)
    distances = distance_sq.detach().sqrt().flatten()
    positive = distances[distances > 0]
    if positive.numel() == 0:
        bandwidth = distance_sq.new_tensor(config.min_bandwidth)
    else:
        bandwidth = torch.quantile(positive, 0.5).to(distance_sq.device)
    return (bandwidth * config.bandwidth_scale).clamp_min(config.min_bandwidth)


def _mean_stat(values: torch.Tensor) -> float:
    return float(values.detach().float().mean().cpu())


def _quantile_stat(values: torch.Tensor, q: float) -> float:
    values_cpu = values.detach().float().flatten().cpu()
    if values_cpu.numel() == 0:
        return float("nan")
    return float(torch.quantile(values_cpu, q))


def _pairwise_abs_similarity_mean(direction: torch.Tensor) -> float:
    if direction.shape[0] < 2:
        return float("nan")
    similarities = direction.detach() @ direction.detach().T
    mask = ~torch.eye(direction.shape[0], dtype=torch.bool, device=direction.device)
    return float(similarities[mask].abs().mean().cpu())
