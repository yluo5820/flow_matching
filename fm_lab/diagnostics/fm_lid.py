"""Flow-matching intrinsic-dimension estimators."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fm_lab.diagnostics._linalg import svdvals
from fm_lab.solvers import Solver

RepresentationFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FMJacobianSpectrumEstimate:
    """Multiscale singular spectra for one data point."""

    t_values: torch.Tensor
    singular_values: tuple[torch.Tensor, ...]
    participation_rank: torch.Tensor
    entropy_rank: torch.Tensor
    threshold_rank: torch.Tensor
    eps: float
    num_directions: int
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "t_values": [float(value) for value in self.t_values.detach().cpu()],
            "fm_jacobian_spectra": [
                values.detach().cpu().tolist() for values in self.singular_values
            ],
            "fm_jacobian_participation_rank": self.participation_rank.detach().cpu().tolist(),
            "fm_jacobian_entropy_rank": self.entropy_rank.detach().cpu().tolist(),
            "fm_jacobian_threshold_rank": self.threshold_rank.detach().cpu().tolist(),
            "eps": self.eps,
            "num_directions": self.num_directions,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class FMFLIPDEstimate:
    """Multiscale FM-FLIPD estimates for a point or batch."""

    t_values: torch.Tensor
    lid: torch.Tensor
    divergence: torch.Tensor
    recovered_score_norm: torch.Tensor
    ambient_dimension: int
    schedule: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "t_values": [float(value) for value in self.t_values.detach().cpu()],
            "fm_flipd_lid": self.lid.detach().cpu().tolist(),
            "fm_flipd_divergence": self.divergence.detach().cpu().tolist(),
            "fm_flipd_recovered_score_norm": self.recovered_score_norm.detach().cpu().tolist(),
            "ambient_dimension": self.ambient_dimension,
            "schedule": self.schedule,
        }


@dataclass(frozen=True)
class GaussianFMScheduleValues:
    alpha: torch.Tensor
    sigma: torch.Tensor
    alpha_dot: torch.Tensor
    sigma_dot: torch.Tensor
    delta_dot: torch.Tensor


@dataclass(frozen=True)
class GaussianFMSchedule:
    """Gaussian path schedule `x_t = alpha(t) x_1 + sigma(t) z`."""

    name: str = "linear"
    clamp_min: float = 1e-4

    def __post_init__(self) -> None:
        normalized = self.name.lower()
        if normalized == "cosine":
            normalized = "trig"
        if normalized not in {"linear", "trig"}:
            raise ValueError("GaussianFMSchedule name must be 'linear' or 'trig'.")
        if self.clamp_min <= 0:
            raise ValueError("GaussianFMSchedule clamp_min must be positive.")
        object.__setattr__(self, "name", normalized)

    def values(
        self,
        t: float | torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> GaussianFMScheduleValues:
        time = _scalar_time(t, device=device, dtype=dtype)
        if self.name == "linear":
            alpha = time
            sigma = 1.0 - time
            alpha_dot = torch.ones_like(time)
            sigma_dot = -torch.ones_like(time)
        else:
            angle = 0.5 * math.pi * time
            alpha = torch.sin(angle)
            sigma = torch.cos(angle)
            alpha_dot = 0.5 * math.pi * torch.cos(angle)
            sigma_dot = -0.5 * math.pi * torch.sin(angle)

        alpha = alpha.clamp_min(self.clamp_min)
        sigma = sigma.clamp_min(self.clamp_min)
        delta_dot = sigma_dot / sigma - alpha_dot / alpha
        return GaussianFMScheduleValues(
            alpha=alpha,
            sigma=sigma,
            alpha_dot=alpha_dot,
            sigma_dot=sigma_dot,
            delta_dot=delta_dot,
        )


class FMJacobianSpectrumEstimator:
    """Finite-perturbation spectrum of the learned FM flow map."""

    def __init__(
        self,
        model: nn.Module,
        ode_solver: Solver,
        t_values: Sequence[float],
        *,
        eps: float = 1e-2,
        num_directions: int = 128,
        representation_fn: RepresentationFn | None = None,
        effective_rank: str = "participation",
        threshold: float = 1e-2,
        normalize_directions: bool = True,
        device: str | torch.device = "auto",
        nfe: int = 64,
        generator: torch.Generator | None = None,
    ) -> None:
        if eps <= 0:
            raise ValueError("FMJacobianSpectrumEstimator eps must be positive.")
        if num_directions < 1:
            raise ValueError("FMJacobianSpectrumEstimator num_directions must be positive.")
        if threshold < 0:
            raise ValueError("FMJacobianSpectrumEstimator threshold must be non-negative.")
        if nfe < 1:
            raise ValueError("FMJacobianSpectrumEstimator nfe must be positive.")
        normalized_rank = effective_rank.lower()
        if normalized_rank not in {"participation", "entropy", "threshold"}:
            raise ValueError("effective_rank must be participation, entropy, or threshold.")

        self.model = model
        self.ode_solver = ode_solver
        self.t_values = tuple(float(value) for value in t_values)
        self.eps = eps
        self.num_directions = num_directions
        self.representation_fn = representation_fn
        self.effective_rank_name = normalized_rank
        self.threshold = threshold
        self.normalize_directions = normalize_directions
        self.device = _resolve_device(device)
        self.nfe = nfe
        self.generator = generator

    @torch.no_grad()
    def compute_pushforward_matrix(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Return the finite-perturbation pushforward used by the spectrum API."""

        x1 = x.to(self.device)
        if x1.ndim < 1:
            raise ValueError("compute_pushforward_matrix expects one data point.")
        x1_batch = x1.unsqueeze(0)
        xt = self._integrate(x1_batch, t0=1.0, t1=float(t))[0]
        directions = sample_unit_directions(
            tuple(xt.shape),
            self.num_directions,
            device=self.device,
            dtype=xt.dtype,
            normalize=self.normalize_directions,
            generator=self.generator,
        )
        x1_perturbed = self._integrate(
            xt.unsqueeze(0) + self.eps * directions,
            t0=float(t),
            t1=1.0,
        )

        if self.representation_fn is None:
            base = x1_batch.reshape(1, -1)
            perturbed = x1_perturbed.reshape(self.num_directions, -1)
        else:
            base = self.representation_fn(x1_batch).reshape(1, -1)
            perturbed = self.representation_fn(x1_perturbed).reshape(
                self.num_directions,
                -1,
            )
        return ((perturbed - base) / self.eps).T

    @torch.no_grad()
    def compute_spectrum(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return svdvals(self.compute_pushforward_matrix(x, t))

    def estimate_point(self, x: torch.Tensor) -> FMJacobianSpectrumEstimate:
        spectra = tuple(self.compute_spectrum(x, t) for t in self.t_values)
        return FMJacobianSpectrumEstimate(
            t_values=torch.tensor(self.t_values, device=self.device),
            singular_values=spectra,
            participation_rank=torch.stack([participation_rank(s) for s in spectra]),
            entropy_rank=torch.stack([entropy_rank(s) for s in spectra]),
            threshold_rank=torch.stack(
                [threshold_rank(s, threshold=self.threshold) for s in spectra]
            ),
            eps=self.eps,
            num_directions=self.num_directions,
            threshold=self.threshold,
        )

    def estimate_batch(self, x_batch: torch.Tensor) -> list[FMJacobianSpectrumEstimate]:
        return [self.estimate_point(x) for x in x_batch]

    def effective_rank(self, singular_values: torch.Tensor) -> torch.Tensor:
        if self.effective_rank_name == "participation":
            return participation_rank(singular_values)
        if self.effective_rank_name == "entropy":
            return entropy_rank(singular_values)
        return threshold_rank(singular_values, threshold=self.threshold).to(singular_values.dtype)

    def _integrate(self, x0: torch.Tensor, *, t0: float, t1: float) -> torch.Tensor:
        if t0 == t1:
            return x0
        t_grid = torch.linspace(t0, t1, self.nfe + 1, device=x0.device, dtype=x0.dtype)

        def v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.model(x, t)

        return self.ode_solver.solve(v_fn, x0, t_grid, return_trajectory=False)


class FMFLIPDEstimator:
    """Continuity-equation FM analogue of FLIPD for Gaussian FM paths."""

    def __init__(
        self,
        model: nn.Module,
        path_schedule: GaussianFMSchedule | str,
        t_values: Sequence[float],
        *,
        num_trace_samples: int | None = 1,
        clamp_sigma: float = 1e-4,
        device: str | torch.device = "auto",
        trace_distribution: str = "rademacher",
    ) -> None:
        if isinstance(path_schedule, str):
            path_schedule = GaussianFMSchedule(path_schedule, clamp_min=clamp_sigma)
        if num_trace_samples is not None and num_trace_samples < 1:
            raise ValueError("num_trace_samples must be positive or None.")
        self.model = model
        self.schedule = path_schedule
        self.t_values = tuple(float(value) for value in t_values)
        self.num_trace_samples = num_trace_samples
        self.device = _resolve_device(device)
        self.trace_distribution = trace_distribution

    def estimate_at_time(
        self,
        x: torch.Tensor,
        t: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_batch, _ = _ensure_batch(x.to(self.device))
        batch_size = x_batch.shape[0]
        ambient_dim = x_batch[0].numel()
        values = self.schedule.values(t, device=x_batch.device, dtype=x_batch.dtype)
        alpha = _expand_scalar(values.alpha, x_batch)
        sigma = _expand_scalar(values.sigma, x_batch)
        alpha_dot = _expand_scalar(values.alpha_dot, x_batch)
        sigma_dot = _expand_scalar(values.sigma_dot, x_batch)

        y = (alpha * x_batch).detach().clone().requires_grad_(True)
        t_batch = torch.full((batch_size,), float(t), device=x_batch.device, dtype=x_batch.dtype)
        velocity = self.model(y, t_batch)
        divergence = self.velocity_divergence(y, t_batch, velocity=velocity)
        score = self.recover_score_from_velocity(
            y,
            values=values,
            velocity=velocity,
            alpha=alpha,
            sigma=sigma,
            alpha_dot=alpha_dot,
            sigma_dot=sigma_dot,
        )

        y_flat = y.reshape(batch_size, -1)
        velocity_flat = velocity.reshape(batch_size, -1)
        score_flat = score.reshape(batch_size, -1)
        alpha_ratio = values.alpha_dot / values.alpha
        term_scale = ambient_dim * alpha_ratio
        drift_correction = (score_flat * (alpha_ratio * y_flat - velocity_flat)).sum(dim=1)
        numerator = term_scale - divergence + drift_correction
        lid = ambient_dim + numerator / values.delta_dot
        score_norm = score_flat.norm(dim=1)
        return lid.detach(), divergence.detach(), score_norm.detach()

    def estimate_batch(self, x_batch: torch.Tensor) -> FMFLIPDEstimate:
        x_prepared, _ = _ensure_batch(x_batch.to(self.device))
        lids = []
        divergences = []
        score_norms = []
        for t in self.t_values:
            lid, divergence, score_norm = self.estimate_at_time(x_prepared, t)
            lids.append(lid)
            divergences.append(divergence)
            score_norms.append(score_norm)
        return FMFLIPDEstimate(
            t_values=torch.tensor(self.t_values, device=self.device),
            lid=torch.stack(lids, dim=0),
            divergence=torch.stack(divergences, dim=0),
            recovered_score_norm=torch.stack(score_norms, dim=0),
            ambient_dimension=x_prepared[0].numel(),
            schedule=self.schedule.name,
        )

    def estimate_point(self, x: torch.Tensor) -> FMFLIPDEstimate:
        estimate = self.estimate_batch(x.unsqueeze(0))
        return FMFLIPDEstimate(
            t_values=estimate.t_values,
            lid=estimate.lid[:, 0],
            divergence=estimate.divergence[:, 0],
            recovered_score_norm=estimate.recovered_score_norm[:, 0],
            ambient_dimension=estimate.ambient_dimension,
            schedule=estimate.schedule,
        )

    def recover_score_from_velocity(
        self,
        y: torch.Tensor,
        *,
        values: GaussianFMScheduleValues,
        velocity: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        alpha_dot: torch.Tensor,
        sigma_dot: torch.Tensor,
    ) -> torch.Tensor:
        sigma_ratio = sigma_dot / sigma
        coefficient = alpha_dot - sigma_ratio * alpha
        coefficient = coefficient.clamp_min(torch.finfo(y.dtype).eps)
        posterior_mean = (velocity - sigma_ratio * y) / coefficient
        return (alpha * posterior_mean - y) / values.sigma.square()

    def velocity_divergence(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        *,
        velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if velocity is None:
            velocity = self.model(y, t)
        if self.num_trace_samples is None:
            return _exact_divergence(velocity, y)
        return _hutchinson_divergence(
            velocity,
            y,
            n_samples=self.num_trace_samples,
            distribution=self.trace_distribution,
        )


def sample_unit_directions(
    shape: Sequence[int],
    num_directions: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    normalize: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    directions = torch.randn(
        num_directions,
        *shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if not normalize:
        return directions
    flattened = directions.reshape(num_directions, -1)
    norms = flattened.norm(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    return (flattened / norms).reshape_as(directions)


def participation_rank(singular_values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    squared = singular_values.square()
    return squared.sum().square() / squared.square().sum().clamp_min(eps)


def entropy_rank(singular_values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    weights = singular_values.square()
    probabilities = weights / weights.sum().clamp_min(eps)
    return torch.exp(-(probabilities * torch.log(probabilities.clamp_min(eps))).sum())


def threshold_rank(
    singular_values: torch.Tensor,
    *,
    threshold: float = 1e-2,
    eps: float = 1e-12,
) -> torch.Tensor:
    return (singular_values / singular_values.max().clamp_min(eps) > threshold).sum()


def summarize_lid_values(
    values: torch.Tensor,
    *,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
) -> dict[str, float | list[float]]:
    flattened = values.detach().float().flatten()
    if flattened.numel() == 0:
        raise ValueError("summarize_lid_values requires at least one value.")
    quantile_tensor = torch.tensor(quantiles, device=flattened.device, dtype=flattened.dtype)
    return {
        "mean_lid": float(flattened.mean().cpu()),
        "std_lid": float(flattened.std(unbiased=False).cpu()),
        "median_lid": float(flattened.median().cpu()),
        "quantiles": torch.quantile(flattened, quantile_tensor).cpu().tolist(),
    }


def _exact_divergence(velocity: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    batch_size = x.shape[0]
    flat_velocity = velocity.reshape(batch_size, -1)
    components = []
    for dim in range(flat_velocity.shape[1]):
        component = flat_velocity[:, dim].sum()
        if component.requires_grad:
            gradient = torch.autograd.grad(
                component,
                x,
                retain_graph=True,
                allow_unused=True,
            )[0]
        else:
            gradient = None
        if gradient is None:
            gradient = torch.zeros_like(x)
        components.append(gradient.reshape(batch_size, -1)[:, dim])
    return torch.stack(components, dim=1).sum(dim=1)


def _hutchinson_divergence(
    velocity: torch.Tensor,
    x: torch.Tensor,
    *,
    n_samples: int,
    distribution: str,
) -> torch.Tensor:
    estimates = []
    for index in range(n_samples):
        probe = _hutchinson_probe(x, distribution)
        projection = (velocity * probe).sum()
        if projection.requires_grad:
            gradient = torch.autograd.grad(
                projection,
                x,
                retain_graph=index < n_samples - 1,
                allow_unused=True,
            )[0]
        else:
            gradient = None
        if gradient is None:
            gradient = torch.zeros_like(x)
        estimates.append((gradient * probe).reshape(x.shape[0], -1).sum(dim=1))
    return torch.stack(estimates, dim=0).mean(dim=0)


def _hutchinson_probe(x: torch.Tensor, distribution: str) -> torch.Tensor:
    normalized = distribution.lower()
    if normalized == "rademacher":
        return torch.empty_like(x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    if normalized == "normal":
        return torch.randn_like(x)
    raise ValueError("trace_distribution must be 'rademacher' or 'normal'.")


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 1:
        return x.unsqueeze(0), True
    if x.ndim < 1:
        raise ValueError("Expected a point or batch tensor.")
    return x, False


def _scalar_time(
    t: float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(t):
        if t.numel() != 1:
            raise ValueError("Schedule time must be scalar.")
        return t.to(device=device, dtype=dtype).reshape(())
    return torch.tensor(float(t), device=device, dtype=dtype)


def _expand_scalar(value: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    expanded = value.reshape((1,) * x.ndim)
    return expanded.expand_as(x)
