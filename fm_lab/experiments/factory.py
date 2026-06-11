"""Factories for experiment components."""

from __future__ import annotations

from typing import Any

import torch

from fm_lab.couplings import IndependentCoupling, MinibatchOTCoupling, ReflowCouplingPlaceholder
from fm_lab.data import (
    Annulus,
    Checkerboard,
    ConcentricCircles,
    GaussianMixture2D,
    GaussianMixture3D,
    HelixMixture,
    MultiSwissRoll,
    MultiTorus,
    NestedSphericalShells,
    SphericalShell,
    SwissRoll,
    Torus,
    TwoMoons,
)
from fm_lab.models import MLPVelocity
from fm_lab.paths import LinearPath, SphericalPath, TangentNormalPath
from fm_lab.solvers import (
    EulerSolver,
    HeunSolver,
    MidpointSolver,
    RK4Solver,
    ScipyDopri5Solver,
    Solver,
)
from fm_lab.sources import GaussianSource, SphericalShellSource


def build_target(config: dict[str, Any]):
    data_config = config.get("data", {})
    name = data_config.get("name", "two_moons").lower()
    if name in {"two_moons", "moons"}:
        return TwoMoons(
            noise=float(data_config.get("noise", 0.05)),
            scale=float(data_config.get("scale", 1.0)),
        )
    if name == "checkerboard":
        return Checkerboard(
            grid_size=int(data_config.get("grid_size", 4)),
            extent=float(data_config.get("extent", 2.0)),
            noise=float(data_config.get("noise", 0.02)),
        )
    if name in {"gaussian_mixture", "gmm"}:
        return GaussianMixture2D(
            n_modes=int(data_config.get("n_modes", 8)),
            radius=float(data_config.get("radius", 2.0)),
            std=float(data_config.get("std", 0.08)),
        )
    if name in {"gaussian_mixture_3d", "gmm_3d"}:
        return GaussianMixture3D(
            n_modes=int(data_config.get("n_modes", 12)),
            radius=float(data_config.get("radius", 2.0)),
            std=float(data_config.get("std", 0.08)),
        )
    if name in {"concentric_circles", "circles"}:
        radii = tuple(float(value) for value in data_config.get("radii", [0.8, 1.6]))
        return ConcentricCircles(radii=radii, noise=float(data_config.get("noise", 0.04)))
    if name == "annulus":
        return Annulus(
            inner_radius=float(data_config.get("inner_radius", 0.8)),
            outer_radius=float(data_config.get("outer_radius", 1.6)),
        )
    if name in {"spherical_shell", "sphere", "shell"}:
        return SphericalShell(
            dim=int(data_config.get("dim", 3)),
            radius=float(data_config.get("radius", 1.0)),
            noise=float(data_config.get("noise", 0.02)),
        )
    if name in {"nested_spherical_shells", "nested_shells"}:
        radii = tuple(float(value) for value in data_config.get("radii", [0.7, 1.2, 1.7]))
        return NestedSphericalShells(
            radii=radii,
            dim=int(data_config.get("dim", 3)),
            noise=float(data_config.get("noise", 0.02)),
        )
    if name == "swiss_roll":
        return SwissRoll(
            noise=float(data_config.get("noise", 0.05)),
            scale=float(data_config.get("scale", 1.0)),
        )
    if name == "multi_swiss_roll":
        return MultiSwissRoll(
            n_rolls=int(data_config.get("n_rolls", 3)),
            noise=float(data_config.get("noise", 0.04)),
            scale=float(data_config.get("scale", 0.75)),
            separation=float(data_config.get("separation", 2.0)),
        )
    if name == "torus":
        return Torus(
            major_radius=float(data_config.get("major_radius", 1.2)),
            minor_radius=float(data_config.get("minor_radius", 0.35)),
            noise=float(data_config.get("noise", 0.02)),
        )
    if name == "multi_torus":
        return MultiTorus(
            n_tori=int(data_config.get("n_tori", 3)),
            major_radius=float(data_config.get("major_radius", 0.75)),
            minor_radius=float(data_config.get("minor_radius", 0.22)),
            separation=float(data_config.get("separation", 2.2)),
            noise=float(data_config.get("noise", 0.02)),
        )
    if name == "helix_mixture":
        return HelixMixture(
            n_helixes=int(data_config.get("n_helixes", 4)),
            turns=float(data_config.get("turns", 3.0)),
            radius=float(data_config.get("radius", 0.35)),
            pitch=float(data_config.get("pitch", 1.8)),
            separation=float(data_config.get("separation", 1.5)),
            noise=float(data_config.get("noise", 0.03)),
        )
    raise ValueError(f"Unsupported target distribution: {name}")


def build_source(config: dict[str, Any]):
    source_config = config.get("source", {})
    name = source_config.get("name", "gaussian").lower()
    if name in {"gaussian", "standard_gaussian"}:
        return GaussianSource(
            dim=int(source_config.get("dim", 2)),
            std=float(source_config.get("std", 1.0)),
            mean=float(source_config.get("mean", 0.0)),
        )
    if name in {"spherical_shell", "sphere", "shell"}:
        return SphericalShellSource(
            dim=int(source_config.get("dim", 2)),
            radius=float(source_config.get("radius", 1.0)),
            noise=float(source_config.get("noise", 0.0)),
        )
    raise ValueError(f"Unsupported source distribution: {name}")


def build_coupling(config: dict[str, Any]):
    coupling_config = config.get("coupling", {})
    name = coupling_config.get("name", "independent").lower()
    if name == "independent":
        return IndependentCoupling(shuffle_target=bool(coupling_config.get("shuffle_target", True)))
    if name in {"minibatch_ot", "ot"}:
        return MinibatchOTCoupling(max_exact_size=int(coupling_config.get("max_exact_size", 2048)))
    if name in {"reflow", "reflow_placeholder"}:
        return ReflowCouplingPlaceholder(checkpoint_path=coupling_config.get("checkpoint_path"))
    raise ValueError(f"Unsupported coupling: {name}")


def build_path(config: dict[str, Any]):
    path_config = config.get("path", {})
    name = path_config.get("name", "linear").lower()
    if name in {"linear", "rectified"}:
        return LinearPath()
    if name == "spherical":
        return SphericalPath(
            eps=float(path_config.get("eps", 1e-6)),
            interpolate_radius=bool(path_config.get("interpolate_radius", True)),
        )
    if name in {"tangent_normal", "polar"}:
        return TangentNormalPath(eps=float(path_config.get("eps", 1e-6)))
    raise ValueError(f"Unsupported path: {name}")


def build_model(config: dict[str, Any], dim: int):
    model_config = config.get("model", {})
    name = model_config.get("name", "mlp").lower()
    if name == "mlp":
        return MLPVelocity(
            dim=dim,
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            depth=int(model_config.get("depth", 4)),
            activation=model_config.get("activation", "silu"),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 64)),
        )
    raise ValueError(f"Unsupported model: {name}")


def build_solvers(config: dict[str, Any]) -> list[Solver]:
    solver_config = config.get("solvers", {})
    names = solver_config.get("names", ["euler"])
    solvers = []
    for name in names:
        normalized = name.lower()
        if normalized == "euler":
            solvers.append(EulerSolver())
        elif normalized == "heun":
            solvers.append(HeunSolver())
        elif normalized == "midpoint":
            solvers.append(MidpointSolver())
        elif normalized == "rk4":
            solvers.append(RK4Solver())
        elif normalized in {"dopri", "dopri5", "rk45"}:
            solvers.append(
                ScipyDopri5Solver(
                    rtol=float(solver_config.get("rtol", 1e-5)),
                    atol=float(solver_config.get("atol", 1e-6)),
                )
            )
        else:
            raise ValueError(f"Unsupported solver: {name}")
    return solvers


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
