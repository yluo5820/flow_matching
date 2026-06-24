"""Low-dimensional manifold and shell toy targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _resolve_device(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu" if device is None else device)


@dataclass
class SphericalShell:
    dim: int = 3
    radius: float = 1.0
    noise: float = 0.02
    name: str = "spherical_shell"

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        x = torch.randn(n, self.dim, device=device)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        x = self.radius * x
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": self.dim - 1,
            "radius": self.radius,
            "noise": self.noise,
        }


@dataclass
class NestedSphericalShells:
    radii: tuple[float, ...] = (0.7, 1.2, 1.7)
    noise: float = 0.02
    dim: int = 3
    name: str = "nested_spherical_shells"

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if not self.radii:
            raise ValueError("NestedSphericalShells requires at least one radius.")
        device = _resolve_device(device)
        radius_ids = torch.randint(0, len(self.radii), (n,), device=device)
        radii = torch.tensor(self.radii, dtype=torch.float32, device=device)[radius_ids]
        x = torch.randn(n, self.dim, device=device)
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        x = radii[:, None] * x
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": self.dim - 1,
            "radii": list(self.radii),
            "noise": self.noise,
        }


@dataclass
class SwissRoll:
    noise: float = 0.05
    scale: float = 1.0
    name: str = "swiss_roll"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        t = 1.5 * math.pi * (1.0 + 2.0 * torch.rand(n, device=device))
        height = 2.0 * torch.rand(n, device=device) - 1.0
        x = torch.stack([t * torch.cos(t), height, t * torch.sin(t)], dim=1)
        x = self.scale * x / (4.5 * math.pi)
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 2,
            "noise": self.noise,
            "scale": self.scale,
        }


@dataclass
class MultiSwissRoll:
    n_rolls: int = 3
    noise: float = 0.04
    scale: float = 0.75
    separation: float = 2.0
    name: str = "multi_swiss_roll"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if self.n_rolls < 1:
            raise ValueError("MultiSwissRoll requires at least one roll.")
        device = _resolve_device(device)
        component_ids = torch.randint(0, self.n_rolls, (n,), device=device)
        angles = 2 * math.pi * component_ids.float() / self.n_rolls

        t = 1.5 * math.pi * (1.0 + 2.0 * torch.rand(n, device=device))
        height = 2.0 * torch.rand(n, device=device) - 1.0
        base = torch.stack([t * torch.cos(t), height, t * torch.sin(t)], dim=1)
        base = self.scale * base / (4.5 * math.pi)

        cos_angle = torch.cos(angles)
        sin_angle = torch.sin(angles)
        x = cos_angle * base[:, 0] + sin_angle * base[:, 2]
        z = -sin_angle * base[:, 0] + cos_angle * base[:, 2]
        rotated = torch.stack([x, base[:, 1], z], dim=1)
        shifts = self.separation * torch.stack(
            [torch.cos(angles), torch.zeros_like(angles), torch.sin(angles)],
            dim=1,
        )
        samples = rotated + shifts
        if self.noise > 0:
            samples = samples + self.noise * torch.randn_like(samples)
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 2,
            "n_rolls": self.n_rolls,
            "noise": self.noise,
            "scale": self.scale,
            "separation": self.separation,
        }


@dataclass
class GaussianMixture3D:
    n_modes: int = 12
    radius: float = 2.0
    std: float = 0.08
    name: str = "gaussian_mixture_3d"
    dim: int = 3

    def centers(self, device: torch.device | str | None = None) -> torch.Tensor:
        if self.n_modes < 1:
            raise ValueError("GaussianMixture3D requires at least one mode.")
        device = _resolve_device(device)
        indices = torch.arange(self.n_modes, dtype=torch.float32, device=device)
        z = 1.0 - 2.0 * (indices + 0.5) / self.n_modes
        radius_xy = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        theta = golden_angle * indices
        centers = torch.stack(
            [radius_xy * torch.cos(theta), radius_xy * torch.sin(theta), z],
            dim=1,
        )
        return self.radius * centers

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        centers = self.centers(device)
        mode_ids = torch.randint(0, self.n_modes, (n,), device=device)
        return centers[mode_ids] + self.std * torch.randn(n, 3, device=device)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        centers = self.centers(x.device)
        diff = x[:, None, :] - centers[None, :, :]
        exponent = -0.5 * diff.square().sum(dim=-1) / (self.std**2)
        log_norm = -1.5 * math.log(2 * math.pi) - 3.0 * math.log(self.std)
        return torch.logsumexp(exponent + log_norm - math.log(self.n_modes), dim=1)

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 3,
            "n_modes": self.n_modes,
            "radius": self.radius,
            "std": self.std,
        }


@dataclass
class Torus:
    major_radius: float = 1.2
    minor_radius: float = 0.35
    noise: float = 0.02
    name: str = "torus"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        device = _resolve_device(device)
        return _sample_torus(
            n=n,
            device=device,
            major_radius=self.major_radius,
            minor_radius=self.minor_radius,
            noise=self.noise,
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 2,
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
            "noise": self.noise,
        }


@dataclass
class MultiTorus:
    n_tori: int = 3
    major_radius: float = 0.75
    minor_radius: float = 0.22
    separation: float = 2.2
    noise: float = 0.02
    name: str = "multi_torus"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if self.n_tori < 1:
            raise ValueError("MultiTorus requires at least one torus.")
        device = _resolve_device(device)
        component_ids = torch.randint(0, self.n_tori, (n,), device=device)
        angles = 2 * math.pi * component_ids.float() / self.n_tori
        samples = _sample_torus(
            n=n,
            device=device,
            major_radius=self.major_radius,
            minor_radius=self.minor_radius,
            noise=self.noise,
        )
        shifts = self.separation * torch.stack(
            [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)],
            dim=1,
        )
        return samples + shifts

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 2,
            "n_tori": self.n_tori,
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
            "separation": self.separation,
            "noise": self.noise,
        }


@dataclass
class HelixMixture:
    n_helixes: int = 4
    turns: float = 3.0
    radius: float = 0.35
    pitch: float = 1.8
    separation: float = 1.5
    noise: float = 0.03
    name: str = "helix_mixture"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if self.n_helixes < 1:
            raise ValueError("HelixMixture requires at least one helix.")
        device = _resolve_device(device)
        component_ids = torch.randint(0, self.n_helixes, (n,), device=device)
        component_angles = 2 * math.pi * component_ids.float() / self.n_helixes
        u = torch.rand(n, device=device)
        theta = 2 * math.pi * self.turns * u + component_angles
        base = torch.stack(
            [
                self.radius * torch.cos(theta),
                self.radius * torch.sin(theta),
                self.pitch * (u - 0.5),
            ],
            dim=1,
        )
        shifts = self.separation * torch.stack(
            [torch.cos(component_angles), torch.sin(component_angles), torch.zeros_like(u)],
            dim=1,
        )
        samples = base + shifts
        if self.noise > 0:
            samples = samples + self.noise * torch.randn_like(samples)
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 1,
            "n_helixes": self.n_helixes,
            "turns": self.turns,
            "radius": self.radius,
            "pitch": self.pitch,
            "separation": self.separation,
            "noise": self.noise,
        }


@dataclass
class MoebiusStrip:
    major_radius: float = 1.2
    half_width: float = 0.35
    noise: float = 0.0
    name: str = "moebius_strip"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if self.major_radius <= 0:
            raise ValueError("MoebiusStrip major_radius must be positive.")
        if self.half_width <= 0:
            raise ValueError("MoebiusStrip half_width must be positive.")
        device = _resolve_device(device)
        theta = 2 * math.pi * torch.rand(n, device=device)
        width = self.half_width * (2 * torch.rand(n, device=device) - 1)
        half_theta = 0.5 * theta
        radial = self.major_radius + width * torch.cos(half_theta)
        samples = torch.stack(
            [
                radial * torch.cos(theta),
                radial * torch.sin(theta),
                width * torch.sin(half_theta),
            ],
            dim=1,
        )
        if self.noise > 0:
            samples = samples + self.noise * torch.randn_like(samples)
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 2,
            "major_radius": self.major_radius,
            "half_width": self.half_width,
            "noise": self.noise,
        }


@dataclass
class LineSegment3D:
    length: float = 3.0
    direction: tuple[float, float, float] = (1.0, 0.5, 0.25)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise: float = 0.0
    name: str = "line_segment_3d"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if self.length <= 0:
            raise ValueError("LineSegment3D length must be positive.")
        if len(self.direction) != 3 or len(self.center) != 3:
            raise ValueError("LineSegment3D direction and center must have length 3.")
        device = _resolve_device(device)
        direction = torch.tensor(self.direction, dtype=torch.float32, device=device)
        norm = direction.norm()
        if float(norm) <= 0:
            raise ValueError("LineSegment3D direction must be nonzero.")
        direction = direction / norm
        center = torch.tensor(self.center, dtype=torch.float32, device=device)
        coordinate = self.length * (torch.rand(n, device=device) - 0.5)
        samples = center + coordinate[:, None] * direction
        if self.noise > 0:
            samples = samples + self.noise * torch.randn_like(samples)
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 1,
            "length": self.length,
            "direction": list(self.direction),
            "center": list(self.center),
            "noise": self.noise,
        }


@dataclass
class PlanarDisk:
    radius: float = 1.2
    height: float = 0.0
    noise: float = 0.0
    name: str = "planar_disk"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if self.radius <= 0:
            raise ValueError("PlanarDisk radius must be positive.")
        device = _resolve_device(device)
        radius = self.radius * torch.sqrt(torch.rand(n, device=device))
        angle = 2 * math.pi * torch.rand(n, device=device)
        samples = torch.stack(
            [
                radius * torch.cos(angle),
                radius * torch.sin(angle),
                torch.full_like(radius, self.height),
            ],
            dim=1,
        )
        if self.noise > 0:
            samples = samples + self.noise * torch.randn_like(samples)
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 2,
            "radius": self.radius,
            "height": self.height,
            "noise": self.noise,
        }


@dataclass
class TrefoilKnot:
    scale: float = 0.55
    noise: float = 0.0
    name: str = "trefoil_knot"
    dim: int = 3

    def sample(self, n: int, device: torch.device | str | None = None) -> torch.Tensor:
        if self.scale <= 0:
            raise ValueError("TrefoilKnot scale must be positive.")
        device = _resolve_device(device)
        theta = 2 * math.pi * torch.rand(n, device=device)
        samples = self.scale * torch.stack(
            [
                torch.sin(theta) + 2 * torch.sin(2 * theta),
                torch.cos(theta) - 2 * torch.cos(2 * theta),
                -torch.sin(3 * theta),
            ],
            dim=1,
        )
        if self.noise > 0:
            samples = samples + self.noise * torch.randn_like(samples)
        return samples

    def log_prob(self, x: torch.Tensor) -> torch.Tensor | None:
        return None

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "intrinsic_dim": 1,
            "scale": self.scale,
            "noise": self.noise,
        }


def _sample_torus(
    *,
    n: int,
    device: torch.device,
    major_radius: float,
    minor_radius: float,
    noise: float,
) -> torch.Tensor:
    theta = 2 * math.pi * torch.rand(n, device=device)
    phi = 2 * math.pi * torch.rand(n, device=device)
    tube = major_radius + minor_radius * torch.cos(phi)
    samples = torch.stack(
        [
            tube * torch.cos(theta),
            tube * torch.sin(theta),
            minor_radius * torch.sin(phi),
        ],
        dim=1,
    )
    if noise > 0:
        samples = samples + noise * torch.randn_like(samples)
    return samples
