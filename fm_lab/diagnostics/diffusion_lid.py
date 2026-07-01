"""Diffusion-model intrinsic-dimension estimators."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NormalBundleEstimate:
    """Normal-bundle rank estimate for each query point."""

    intrinsic_dimension: torch.Tensor
    normal_dimension: torch.Tensor
    singular_values: torch.Tensor
    ambient_dimension: int
    sigma: float
    n_perturbations: int
    rank_threshold: float
    relative_threshold: bool


@dataclass(frozen=True)
class FlipdEstimate:
    """FLIPD estimate and the score divergence used to compute it."""

    intrinsic_dimension: torch.Tensor
    divergence: torch.Tensor
    ambient_dimension: int
    sigma: float
    hutchinson_samples: int | None


@torch.no_grad()
def normal_bundle_dimension(
    score_model: nn.Module,
    x: torch.Tensor,
    *,
    sigma: float,
    t: float | torch.Tensor,
    n_perturbations: int = 64,
    rank_threshold: float = 1e-3,
    relative_threshold: bool = True,
    generator: torch.Generator | None = None,
) -> NormalBundleEstimate:
    """Estimate LID as ambient dimension minus normal score rank.

    This is the score-based normal-bundle estimator: evaluate the score at
    several small Gaussian perturbations around each clean point, stack those
    score vectors, estimate their rank, and subtract that rank from the ambient
    dimension.
    """

    if sigma <= 0:
        raise ValueError("normal_bundle_dimension sigma must be positive.")
    if n_perturbations < 1:
        raise ValueError("normal_bundle_dimension n_perturbations must be positive.")
    if rank_threshold < 0:
        raise ValueError("normal_bundle_dimension rank_threshold must be non-negative.")
    if x.ndim != 2:
        raise ValueError("normal_bundle_dimension expects x with shape (batch, dim).")

    batch_size, ambient_dim = x.shape
    noise = torch.randn(
        batch_size,
        n_perturbations,
        ambient_dim,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )
    perturbed = x[:, None, :] + sigma * noise
    flat = perturbed.reshape(batch_size * n_perturbations, ambient_dim)
    t_flat = _expand_time_argument(t, flat.shape[0], device=x.device, dtype=x.dtype)
    scores = score_model(flat, t_flat).reshape(batch_size, n_perturbations, ambient_dim)
    scores = scores - scores.mean(dim=1, keepdim=True)
    singular_values = torch.linalg.svdvals(scores)
    normal_dimension = _rank_from_singular_values(
        singular_values,
        threshold=rank_threshold,
        relative=relative_threshold,
    )
    intrinsic_dimension = ambient_dim - normal_dimension.to(x.dtype)
    return NormalBundleEstimate(
        intrinsic_dimension=intrinsic_dimension,
        normal_dimension=normal_dimension,
        singular_values=singular_values,
        ambient_dimension=ambient_dim,
        sigma=float(sigma),
        n_perturbations=n_perturbations,
        rank_threshold=rank_threshold,
        relative_threshold=relative_threshold,
    )


def flipd_dimension(
    score_model: nn.Module,
    x: torch.Tensor,
    *,
    sigma: float,
    t: float | torch.Tensor,
    hutchinson_samples: int | None = 1,
    distribution: str = "rademacher",
    create_graph: bool = False,
) -> FlipdEstimate:
    """Estimate LID with the VE/Gaussian-convolution FLIPD formula.

    For a local Gaussian convolution with standard deviation `sigma`, this
    computes `ambient_dim + sigma^2 * div_x score_model(x, t)`. Use
    `hutchinson_samples=None` for exact divergence in low dimensions.
    """

    if sigma <= 0:
        raise ValueError("flipd_dimension sigma must be positive.")
    if x.ndim != 2:
        raise ValueError("flipd_dimension expects x with shape (batch, dim).")
    if hutchinson_samples is not None and hutchinson_samples < 1:
        raise ValueError("flipd_dimension hutchinson_samples must be positive or None.")

    query = x.detach().requires_grad_(True)
    t_query = _expand_time_argument(t, query.shape[0], device=x.device, dtype=x.dtype)
    score = score_model(query, t_query)
    if score.shape != query.shape:
        raise ValueError("score_model must return a tensor with the same shape as x.")
    if hutchinson_samples is None:
        divergence = _exact_divergence(score, query, create_graph=create_graph)
    else:
        divergence = _hutchinson_divergence(
            score,
            query,
            n_samples=hutchinson_samples,
            distribution=distribution,
            create_graph=create_graph,
        )
    ambient_dim = query.shape[1]
    intrinsic_dimension = ambient_dim + sigma * sigma * divergence
    if not create_graph:
        intrinsic_dimension = intrinsic_dimension.detach()
        divergence = divergence.detach()
    return FlipdEstimate(
        intrinsic_dimension=intrinsic_dimension,
        divergence=divergence,
        ambient_dimension=ambient_dim,
        sigma=float(sigma),
        hutchinson_samples=hutchinson_samples,
    )


def _rank_from_singular_values(
    singular_values: torch.Tensor,
    *,
    threshold: float,
    relative: bool,
) -> torch.Tensor:
    if relative:
        eps = torch.finfo(singular_values.dtype).eps
        cutoff = threshold * singular_values[:, :1].clamp_min(eps)
    else:
        cutoff = singular_values.new_full((singular_values.shape[0], 1), threshold)
    return (singular_values > cutoff).sum(dim=1).to(torch.int64)


def _exact_divergence(
    score: torch.Tensor,
    x: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    components: list[torch.Tensor] = []
    for dim in range(score.shape[1]):
        component = score[:, dim].sum()
        if component.requires_grad:
            gradient = torch.autograd.grad(
                component,
                x,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )[0]
        else:
            gradient = None
        if gradient is None:
            gradient = torch.zeros_like(x)
        components.append(gradient[:, dim])
    return torch.stack(components, dim=1).sum(dim=1)


def _hutchinson_divergence(
    score: torch.Tensor,
    x: torch.Tensor,
    *,
    n_samples: int,
    distribution: str,
    create_graph: bool,
) -> torch.Tensor:
    estimates = []
    for index in range(n_samples):
        probe = _hutchinson_probe(x, distribution)
        projection = (score * probe).sum()
        if projection.requires_grad:
            gradient = torch.autograd.grad(
                projection,
                x,
                create_graph=create_graph,
                retain_graph=index < n_samples - 1 or create_graph,
                allow_unused=True,
            )[0]
        else:
            gradient = None
        if gradient is None:
            gradient = torch.zeros_like(x)
        estimates.append((gradient * probe).sum(dim=1))
    return torch.stack(estimates, dim=0).mean(dim=0)


def _hutchinson_probe(x: torch.Tensor, distribution: str) -> torch.Tensor:
    normalized = distribution.lower()
    if normalized == "rademacher":
        return torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    if normalized == "normal":
        return torch.randn_like(x)
    raise ValueError("distribution must be 'rademacher' or 'normal'.")


def _expand_time_argument(
    t: float | torch.Tensor,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(t):
        time = t.to(device=device, dtype=dtype)
        if time.ndim == 0:
            return time.expand(batch_size)
        if time.shape[0] == batch_size:
            return time
        if time.numel() == 1:
            return time.reshape(1).expand(batch_size)
        raise ValueError("Time tensor must be scalar or have one value per batch row.")
    return torch.full((batch_size,), float(t), device=device, dtype=dtype)
