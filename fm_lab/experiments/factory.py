"""Factories for experiment components."""

from __future__ import annotations

from typing import Any

import torch

from fm_lab.couplings import (
    IndependentCoupling,
    MinibatchOTCoupling,
    ModelGeneratedCoupling,
    ReflowCouplingPlaceholder,
)
from fm_lab.data import (
    Annulus,
    Checkerboard,
    ConcentricCircles,
    GaussianMixture2D,
    GaussianMixture3D,
    HelixMixture,
    ImageVariantImages,
    LineSegment3D,
    MNISTImages,
    MoebiusStrip,
    MultiSwissRoll,
    MultiTorus,
    NestedSphericalShells,
    PlanarDisk,
    SphericalShell,
    SwissRoll,
    Torus,
    TrefoilKnot,
    TwoMoons,
)
from fm_lab.models import (
    DirectionSpeedImageUNet,
    DirectionSpeedMLP,
    ImageUNetVelocity,
    MLPVelocity,
)
from fm_lab.paths import (
    GaussianDiffusionPath,
    LearnedAccelerationPath,
    LinearPath,
    SphericalPath,
    TangentNormalPath,
)
from fm_lab.solvers import (
    EulerSolver,
    HeunSolver,
    MidpointSolver,
    RK4Solver,
    ScipyDopri5Solver,
    Solver,
)
from fm_lab.sources import GaussianSource, SphericalShellSource
from fm_lab.utils.checkpoints import load_checkpoint


def build_target(config: dict[str, Any]):
    data_config = config.get("data", {})
    name = data_config.get("name", "two_moons").lower()
    if data_config.get("variant_id"):
        return ImageVariantImages(
            variant_id=str(data_config["variant_id"]),
            workspace=data_config.get("workspace", "outputs/geometry_explorer"),
            normalize=str(data_config.get("normalize", "zero_one")),
            dequantize=bool(data_config.get("dequantize", False)),
        )
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
    if name == "mnist":
        return MNISTImages(
            root=data_config.get("root", "data/mnist"),
            train=bool(data_config.get("train", True)),
            download=bool(data_config.get("download", False)),
            normalize=str(data_config.get("normalize", "zero_one")),
            dequantize=bool(data_config.get("dequantize", False)),
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
    if name == "helix":
        return HelixMixture(
            n_helixes=1,
            turns=float(data_config.get("turns", 3.0)),
            radius=float(data_config.get("radius", 0.6)),
            pitch=float(data_config.get("pitch", 2.4)),
            separation=0.0,
            noise=float(data_config.get("noise", 0.0)),
        )
    if name in {"moebius_strip", "mobius_strip"}:
        return MoebiusStrip(
            major_radius=float(data_config.get("major_radius", 1.2)),
            half_width=float(data_config.get("half_width", 0.35)),
            noise=float(data_config.get("noise", 0.0)),
        )
    if name in {"line_segment_3d", "line_segment", "line"}:
        return LineSegment3D(
            length=float(data_config.get("length", 3.0)),
            direction=tuple(
                float(value)
                for value in data_config.get("direction", [1.0, 0.5, 0.25])
            ),
            center=tuple(
                float(value)
                for value in data_config.get("center", [0.0, 0.0, 0.0])
            ),
            noise=float(data_config.get("noise", 0.0)),
        )
    if name in {"planar_disk", "disk"}:
        return PlanarDisk(
            radius=float(data_config.get("radius", 1.2)),
            height=float(data_config.get("height", 0.0)),
            noise=float(data_config.get("noise", 0.0)),
        )
    if name in {"trefoil_knot", "trefoil"}:
        return TrefoilKnot(
            scale=float(data_config.get("scale", 0.55)),
            noise=float(data_config.get("noise", 0.0)),
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
    if name in {"model_generated", "learned_flow", "teacher", "distillation"}:
        return _build_model_generated_coupling(coupling_config)
    if name == "reflow" and coupling_config.get("checkpoint_path"):
        return _build_model_generated_coupling(coupling_config)
    if name in {"reflow", "reflow_placeholder"}:
        return ReflowCouplingPlaceholder(checkpoint_path=coupling_config.get("checkpoint_path"))
    raise ValueError(f"Unsupported coupling: {name}")


def build_path(config: dict[str, Any]):
    path_config = config.get("path", {})
    name = path_config.get("name", "linear").lower()
    if name in {"linear", "rectified"}:
        return LinearPath()
    if name in {"gaussian_diffusion", "diffusion", "stochastic_interpolant"}:
        return GaussianDiffusionPath(
            schedule=str(path_config.get("schedule", "trig")),
            sigma_min=float(path_config.get("sigma_min", 1e-4)),
        )
    if name in {"learned_acceleration", "acceleration", "quadratic_acceleration"}:
        source_dim = int(config.get("source", {}).get("dim", config.get("data", {}).get("dim", 2)))
        return LearnedAccelerationPath(
            dim=source_dim,
            basis=str(path_config.get("basis", "quadratic")),
            hidden_dim=int(path_config.get("hidden_dim", 128)),
            depth=int(path_config.get("depth", 3)),
            activation=str(path_config.get("activation", "silu")),
            network=str(path_config.get("network", path_config.get("backbone", "mlp"))),
            image_shape=(
                tuple(int(value) for value in path_config["image_shape"])
                if "image_shape" in path_config
                else None
            ),
            base_channels=int(path_config.get("base_channels", 32)),
            zero_init_head=bool(path_config.get("zero_init_head", True)),
            eps=float(path_config.get("eps", 1e-8)),
        )
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
    conditioning_config = config.get("conditioning", {}) or {}
    conditioning_enabled = bool(conditioning_config.get("enabled", False))
    num_classes = (
        int(conditioning_config["num_classes"])
        if conditioning_enabled
        else None
    )
    if conditioning_enabled and num_classes < 1:
        raise ValueError("conditioning.num_classes must be positive.")
    class_embedding_dim = conditioning_config.get("embedding_dim")
    if class_embedding_dim is not None:
        class_embedding_dim = int(class_embedding_dim)
    name = model_config.get("name", "mlp").lower()
    if name == "mlp":
        return MLPVelocity(
            dim=dim,
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            depth=int(model_config.get("depth", 4)),
            activation=model_config.get("activation", "silu"),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 64)),
            num_classes=num_classes,
            class_embedding_dim=class_embedding_dim,
        )
    if name in {"direction_speed_mlp", "direction_only"}:
        if conditioning_enabled:
            raise ValueError("Class conditioning is not supported by direction-speed models.")
        return DirectionSpeedMLP(
            dim=dim,
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            depth=int(model_config.get("depth", 4)),
            activation=model_config.get("activation", "silu"),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 64)),
            direction_eps=float(model_config.get("direction_eps", 1e-8)),
        )
    if name in {"image_unet", "mnist_unet", "conv_unet"}:
        image_shape = tuple(int(value) for value in model_config.get("image_shape", [28, 28]))
        return ImageUNetVelocity(
            dim=dim,
            image_shape=image_shape,
            base_channels=int(model_config.get("base_channels", 32)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 128)),
            activation=model_config.get("activation", "silu"),
            zero_init_head=bool(model_config.get("zero_init_head", True)),
            num_classes=num_classes,
            class_embedding_dim=class_embedding_dim,
        )
    if name in {"direction_speed_image_unet", "direction_only_image_unet"}:
        if conditioning_enabled:
            raise ValueError("Class conditioning is not supported by direction-speed models.")
        image_shape = tuple(int(value) for value in model_config.get("image_shape", [28, 28]))
        return DirectionSpeedImageUNet(
            dim=dim,
            image_shape=image_shape,
            base_channels=int(model_config.get("base_channels", 32)),
            time_embedding_dim=int(model_config.get("time_embedding_dim", 128)),
            activation=model_config.get("activation", "silu"),
            direction_eps=float(model_config.get("direction_eps", 1e-8)),
            direction_zero_init_head=bool(model_config.get("direction_zero_init_head", False)),
            speed_zero_init_head=bool(model_config.get("speed_zero_init_head", True)),
        )
    raise ValueError(f"Unsupported model: {name}")


def build_solvers(config: dict[str, Any]) -> list[Solver]:
    solver_config = config.get("solvers", {})
    names = solver_config.get("names", ["euler"])
    return [_build_solver(name, solver_config) for name in names]


def _build_model_generated_coupling(coupling_config: dict[str, Any]) -> ModelGeneratedCoupling:
    checkpoint_path = coupling_config.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("model_generated coupling requires coupling.checkpoint_path.")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    teacher_config = checkpoint.get("config")
    if not isinstance(teacher_config, dict):
        raise ValueError("Teacher checkpoint must contain a config dictionary.")

    teacher_source = build_source(teacher_config)
    teacher_model = build_model(teacher_config, dim=teacher_source.dim)
    teacher_model.load_state_dict(checkpoint["model_state_dict"])
    teacher_model.eval()
    solver = _build_solver(str(coupling_config.get("solver", "rk4")), coupling_config)
    return ModelGeneratedCoupling(
        teacher_model=teacher_model,
        solver=solver,
        nfe=int(coupling_config.get("nfe", 64)),
        schedule=str(coupling_config.get("schedule", "uniform")),
    )


def _build_solver(name: str, solver_config: dict[str, Any]) -> Solver:
    normalized = name.lower()
    if normalized == "euler":
        return EulerSolver()
    if normalized == "heun":
        return HeunSolver()
    if normalized == "midpoint":
        return MidpointSolver()
    if normalized == "rk4":
        return RK4Solver()
    if normalized in {"dopri", "dopri5", "rk45"}:
        return ScipyDopri5Solver(
            rtol=float(solver_config.get("rtol", 1e-5)),
            atol=float(solver_config.get("atol", 1e-6)),
        )
    raise ValueError(f"Unsupported solver: {name}")


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
