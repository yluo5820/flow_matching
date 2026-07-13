"""Deterministic sampling guidance utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import torch
from torch import nn

from fm_lab.paths.base import expand_time
from fm_lab.training.prediction import model_prediction


@dataclass(frozen=True)
class PriorGuidanceConfig:
    scale: float = 1.0

    @property
    def enabled(self) -> bool:
        return not math.isclose(self.scale, 1.0)

    def summary(self) -> dict[str, float]:
        return {"scale": self.scale}


@dataclass(frozen=True)
class DensityGuidanceConfig:
    quantile: float = 0.5
    strength: float = 1.0
    min_score_norm: float = 1.0e-8
    min_alpha: float = 1.0e-3
    t_min: float = 1.0e-6
    t_max: float = 1.0
    prior_rescale_quantile: float | None = 0.5

    @property
    def enabled(self) -> bool:
        return self.strength != 0.0 and not math.isclose(self.quantile, 0.5)

    @property
    def normal_quantile(self) -> float:
        return NormalDist().inv_cdf(self.quantile)

    def summary(self) -> dict[str, float]:
        return {
            "quantile": self.quantile,
            "strength": self.strength,
            "min_score_norm": self.min_score_norm,
            "min_alpha": self.min_alpha,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "prior_rescale_quantile": self.prior_rescale_quantile,
        }


@dataclass(frozen=True)
class SamplingGuidanceConfig:
    prior: PriorGuidanceConfig = PriorGuidanceConfig()
    density: DensityGuidanceConfig | None = None

    def summary(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if self.prior.enabled:
            values["prior"] = self.prior.summary()
        if self.density is not None:
            values["density"] = self.density.summary()
        return values


def build_sampling_guidance_config(sampling_config: dict[str, Any]) -> SamplingGuidanceConfig:
    guidance_config = _dict_section(sampling_config.get("guidance", {}), "sampling.guidance")
    prior = _build_prior_guidance_config(
        _merged_section(
            guidance_config.get("prior"),
            sampling_config.get("prior_guidance"),
        )
    )
    density = _build_density_guidance_config(
        _merged_section(
            guidance_config.get("density"),
            sampling_config.get("density_guidance"),
        )
    )
    return SamplingGuidanceConfig(prior=prior, density=density)


def apply_prior_guidance(
    samples: torch.Tensor,
    *,
    source: Any,
    config: PriorGuidanceConfig,
) -> torch.Tensor:
    if not config.enabled:
        return samples
    source_name = str(getattr(source, "name", "")).lower()
    if source_name != "gaussian":
        raise ValueError("Prior guidance requires a Gaussian source distribution.")
    mean = float(getattr(source, "mean", 0.0))
    return mean + config.scale * (samples - mean)


def apply_density_prior_rescaling(
    samples: torch.Tensor,
    *,
    source: Any,
    config: DensityGuidanceConfig | None,
) -> torch.Tensor:
    if config is None or not config.enabled or config.prior_rescale_quantile is None:
        return samples
    source_name = str(getattr(source, "name", "")).lower()
    if source_name != "gaussian":
        raise ValueError("Density guidance prior rescaling requires a Gaussian source.")
    dim = int(samples[0].numel())
    target_norm = _chi_quantile(config.prior_rescale_quantile, dim)
    std = float(getattr(source, "std", 1.0))
    mean = float(getattr(source, "mean", 0.0))
    centered = samples - mean
    norms = centered.flatten(start_dim=1).norm(dim=1).clamp_min(1.0e-12)
    scales = std * target_norm / norms
    return mean + centered * scales.view(-1, *([1] * (samples.ndim - 1)))


def apply_density_guidance(
    *,
    base_model: nn.Module,
    velocity_model: nn.Module,
    path: Any,
    objective: Any,
    config: DensityGuidanceConfig | None,
) -> nn.Module:
    if config is None:
        return velocity_model
    if not config.enabled:
        return velocity_model
    if str(getattr(path, "name", path.__class__.__name__)).lower() != "gaussian_diffusion":
        raise ValueError("Density guidance requires a Gaussian diffusion path.")
    output = getattr(objective, "model_output", None)
    if output is None:
        output = getattr(objective, "prediction_type", "velocity")
    if str(output).lower() not in {"x", "x1", "data", "clean", "clean_data", "clean_image"}:
        raise ValueError("Density guidance currently requires a clean-x predicting model.")
    if not hasattr(path, "_schedule"):
        raise ValueError("Density guidance requires a path with a diffusion schedule.")
    return DensityGuidedDiffusionVelocity(base_model, path, config)


class DensityGuidedDiffusionVelocity(nn.Module):
    """Clean-x diffusion sampler with paper-style score coefficient guidance."""

    def __init__(
        self,
        model: nn.Module,
        path: Any,
        config: DensityGuidanceConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.path = path
        self.config = config
        self.requires_source_label = bool(getattr(model, "requires_source_label", False))
        self.is_class_conditional = bool(getattr(model, "is_class_conditional", False))

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = None if context is None else context.get("source_label")
        class_labels = None if context is None else context.get("class_labels")
        x_prediction = model_prediction(
            self.model,
            x,
            t,
            source_label=source_label,
            class_labels=class_labels,
        )
        alpha_t, sigma_t, alpha_dot_t, sigma_dot_t = self.path._schedule(t)
        alpha = expand_time(alpha_t.clamp_min(self.config.min_alpha), x)
        sigma = expand_time(sigma_t.clamp_min(getattr(self.path, "sigma_min", 1.0e-4)), x)
        alpha_dot = expand_time(alpha_dot_t, x)
        sigma_dot = expand_time(sigma_dot_t, x)

        epsilon_prediction = (x - expand_time(alpha_t, x) * x_prediction) / sigma
        score = -epsilon_prediction / sigma
        velocity = alpha_dot * x_prediction + sigma_dot * epsilon_prediction

        score_scale_delta = self._score_scale_delta(score=score, sigma=sigma, t=t)
        score_coefficient = alpha_dot * sigma.square() / alpha - sigma_dot * sigma
        return velocity + score_scale_delta * score_coefficient * score

    def _score_scale_delta(
        self,
        *,
        score: torch.Tensor,
        sigma: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        dim = int(score[0].numel())
        scaled_score_norm_sq = (sigma * score).flatten(start_dim=1).square().sum(dim=1)
        scaled_score_norm_sq = scaled_score_norm_sq.clamp_min(self.config.min_score_norm)
        delta = (
            self.config.strength
            * math.sqrt(2.0 * dim)
            * self.config.normal_quantile
            / scaled_score_norm_sq
        )
        active = ((t >= self.config.t_min) & (t <= self.config.t_max)).to(score.dtype)
        return expand_time(delta * active, score)


def _build_prior_guidance_config(values: dict[str, Any]) -> PriorGuidanceConfig:
    if not values:
        return PriorGuidanceConfig()
    enabled = bool(values.get("enabled", True))
    scale = float(values.get("scale", values.get("norm_scale", 1.0)))
    if not enabled:
        scale = 1.0
    if scale <= 0.0:
        raise ValueError("sampling guidance prior scale must be positive.")
    return PriorGuidanceConfig(scale=scale)


def _build_density_guidance_config(values: dict[str, Any]) -> DensityGuidanceConfig | None:
    if not values or not bool(values.get("enabled", True)):
        return None
    quantile = float(values.get("quantile", values.get("q", 0.5)))
    if not 0.0 < quantile < 1.0:
        raise ValueError("sampling guidance density quantile must be in (0, 1).")
    strength = float(values.get("strength", 1.0))
    min_score_norm = float(values.get("min_score_norm", 1.0e-8))
    min_alpha = float(values.get("min_alpha", 1.0e-3))
    t_min = float(values.get("t_min", 1.0e-6))
    t_max = float(values.get("t_max", 1.0))
    prior_rescale_quantile = values.get(
        "prior_rescale_quantile",
        values.get("prior_quantile", 0.5),
    )
    if prior_rescale_quantile is not None:
        prior_rescale_quantile = float(prior_rescale_quantile)
    if min_score_norm <= 0.0:
        raise ValueError("sampling guidance density min_score_norm must be positive.")
    if min_alpha <= 0.0:
        raise ValueError("sampling guidance density min_alpha must be positive.")
    if t_min > t_max:
        raise ValueError("sampling guidance density t_min must be <= t_max.")
    if prior_rescale_quantile is not None and not 0.0 < prior_rescale_quantile < 1.0:
        raise ValueError(
            "sampling guidance density prior_rescale_quantile must be in (0, 1)."
        )
    return DensityGuidanceConfig(
        quantile=quantile,
        strength=strength,
        min_score_norm=min_score_norm,
        min_alpha=min_alpha,
        t_min=t_min,
        t_max=t_max,
        prior_rescale_quantile=prior_rescale_quantile,
    )


def _merged_section(primary: Any, secondary: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if secondary is not None:
        values.update(_dict_section(secondary, "sampling guidance"))
    if primary is not None:
        values.update(_dict_section(primary, "sampling guidance"))
    return values


def _dict_section(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _chi_quantile(probability: float, dim: int) -> float:
    try:
        from scipy.stats import chi

        return float(chi.ppf(probability, df=dim))
    except Exception:
        # Wilson-Hilferty approximation for the chi median/quantile fallback.
        z = NormalDist().inv_cdf(probability)
        return math.sqrt(dim * (1.0 - 2.0 / (9.0 * dim) + z * math.sqrt(2.0 / (9.0 * dim))) ** 3)
