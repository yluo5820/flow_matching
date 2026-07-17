"""Semantic latent factor spaces for synthetic image-formation benchmarks."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LatentSample:
    """Container for sampled latent states and column-oriented metadata."""

    values: Any
    metadata: dict[str, Any]


class LatentFactorSpace(ABC):
    """Base class for meaningful visual/image-formation factors."""

    name: str
    dim: int
    factor_names: list[str]
    factor_dims: list[int]

    @abstractmethod
    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        """Sample n latent states from this factor space."""

    @abstractmethod
    def tangent_basis(self, z: Any) -> Any:
        """Return dim tangent directions at z."""

    @abstractmethod
    def retract(self, z: Any, tangent_vec: Any, eps: float) -> Any:
        """Move from z along tangent_vec by eps while staying in the factor space."""

    @abstractmethod
    def distance(self, z1: Any, z2: Any) -> float:
        """Natural latent-space distance."""

    def coordinates(self, z: Any) -> dict[str, float]:
        """Human-readable coordinates for logging and plotting."""

        return {}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        """Factor-bin labels for color overlays and UI display."""

        return {}

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        """Stable labels for the tangent-basis directions at z."""

        return _axis_labels(str(self.name), int(self.dim))


@dataclass(frozen=True)
class AzimuthCircle(LatentFactorSpace):
    """One-dimensional camera azimuth orbit, represented in radians."""

    name: str = "azimuth"
    dim: int = 1
    factor_names: list[str] = None  # type: ignore[assignment]
    factor_dims: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_names", [self.name])
        object.__setattr__(self, "factor_dims", [self.dim])

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        rng = np.random.default_rng(seed)
        values = rng.uniform(0.0, 2.0 * math.pi, size=int(n)).astype(np.float32)
        return LatentSample(values=values, metadata=_columns(self, values))

    def tangent_basis(self, z: Any) -> np.ndarray:
        return np.asarray([[1.0]], dtype=np.float32)

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        return ["azimuth"]

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> float:
        step = float(np.asarray(tangent_vec).reshape(-1)[0])
        return float((float(z) + float(eps) * step) % (2.0 * math.pi))

    def distance(self, z1: Any, z2: Any) -> float:
        delta = abs(float(z1) - float(z2)) % (2.0 * math.pi)
        return float(min(delta, 2.0 * math.pi - delta))

    def coordinates(self, z: Any) -> dict[str, float]:
        angle = float(z) % (2.0 * math.pi)
        return {
            "azimuth_rad": angle,
            "azimuth_deg": math.degrees(angle),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _cyclic_bin(float(z), bins=num_bins)
        return {
            "label": f"azimuth_bin_{label_id:02d}",
            "label_id": str(label_id),
            "azimuth_bin": str(label_id),
        }


@dataclass(frozen=True)
class LookAtViewSphere(LatentFactorSpace):
    """Camera viewing direction on S^2 for a look-at camera with no roll."""

    name: str = "view_sphere"
    dim: int = 2
    factor_names: list[str] = None  # type: ignore[assignment]
    factor_dims: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_names", [self.name])
        object.__setattr__(self, "factor_dims", [self.dim])

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        rng = np.random.default_rng(seed)
        values = rng.normal(size=(int(n), 3)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return LatentSample(values=values, metadata=_columns(self, values))

    def tangent_basis(self, z: Any) -> np.ndarray:
        direction = _unit3(z)
        helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(direction, helper))) > 0.9:
            helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        first = _normalize(np.cross(direction, helper))
        second = _normalize(np.cross(direction, first))
        return np.stack([first, second]).astype(np.float32)

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        return [f"{self.name}_tangent_1", f"{self.name}_tangent_2"]

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> np.ndarray:
        return _normalize(_unit3(z) + float(eps) * np.asarray(tangent_vec, dtype=np.float32))

    def distance(self, z1: Any, z2: Any) -> float:
        cosine = float(np.dot(_unit3(z1), _unit3(z2)))
        return float(math.acos(min(1.0, max(-1.0, cosine))))

    def coordinates(self, z: Any) -> dict[str, float]:
        direction = _unit3(z)
        azimuth = math.atan2(float(direction[0]), -float(direction[1])) % (2.0 * math.pi)
        elevation = math.asin(float(direction[2]))
        return {
            "view_x": float(direction[0]),
            "view_y": float(direction[1]),
            "view_z": float(direction[2]),
            "view_azimuth_deg": math.degrees(azimuth),
            "view_elevation_deg": math.degrees(elevation),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        z_value = float(_unit3(z)[2])
        label_id = _linear_bin(z_value, (-1.0, 1.0), bins=num_bins)
        return {
            "label": f"view_z_bin_{label_id:02d}",
            "label_id": str(label_id),
            "view_z_bin": str(label_id),
        }


@dataclass(frozen=True)
class BoundedLookAtView(LatentFactorSpace):
    """Look-at camera direction within an elevation band, without roll."""

    elevation_bounds: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0)
    name: str = "bounded_look_at_view"
    dim: int = 2
    factor_names: tuple[str, ...] = ("camera_view",)
    factor_dims: tuple[int, ...] = (2,)

    def __post_init__(self) -> None:
        low, high = (float(value) for value in self.elevation_bounds)
        if not -math.pi / 2.0 < low < high < math.pi / 2.0:
            raise ValueError("elevation_bounds must lie inside (-pi/2, pi/2).")

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        rng = np.random.default_rng(seed)
        low, high = self.elevation_bounds
        values = np.column_stack(
            [
                rng.uniform(-math.pi, math.pi, int(n)),
                rng.uniform(math.sin(low), math.sin(high), int(n)),
            ]
        ).astype(np.float32)
        return LatentSample(values=values, metadata=_columns(self, values))

    def tangent_basis(self, z: Any) -> np.ndarray:
        del z
        return np.eye(2, dtype=np.float32)

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        del z
        return ["camera_azimuth", "camera_elevation"]

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> np.ndarray:
        value = np.asarray(z, dtype=np.float64) + eps * np.asarray(tangent_vec)
        value[0] = (value[0] + math.pi) % (2.0 * math.pi) - math.pi
        low, high = self.elevation_bounds
        value[1] = np.clip(value[1], math.sin(low), math.sin(high))
        return value.astype(np.float32)

    def distance(self, z1: Any, z2: Any) -> float:
        first = np.asarray(z1, dtype=np.float64)
        second = np.asarray(z2, dtype=np.float64)
        azimuth = abs(first[0] - second[0])
        azimuth = min(azimuth, 2.0 * math.pi - azimuth)
        return float(np.hypot(azimuth, first[1] - second[1]))

    def coordinates(self, z: Any) -> dict[str, float]:
        azimuth, sin_elevation = np.asarray(z, dtype=np.float64)
        return {
            "camera_azimuth": float(azimuth),
            "camera_elevation": float(math.asin(np.clip(sin_elevation, -1.0, 1.0))),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        coordinates = self.coordinates(z)
        azimuth_bin = _linear_bin(
            coordinates["camera_azimuth"],
            (-math.pi, math.pi),
            bins=num_bins,
        )
        elevation_bin = _linear_bin(
            coordinates["camera_elevation"],
            self.elevation_bounds,
            bins=num_bins,
        )
        label_id = azimuth_bin * num_bins + elevation_bin
        return {
            "label": f"bounded_view_bin_{label_id:04d}",
            "label_id": str(label_id),
            "camera_azimuth_bin": str(azimuth_bin),
            "camera_elevation_bin": str(elevation_bin),
        }


@dataclass(frozen=True)
class LightingDirectionSphere(LookAtViewSphere):
    """Directional light factor on S^2."""

    name: str = "light_sphere"

    def coordinates(self, z: Any) -> dict[str, float]:
        direction = _unit3(z)
        azimuth = math.atan2(float(direction[0]), -float(direction[1])) % (2.0 * math.pi)
        elevation = math.asin(float(direction[2]))
        return {
            "light_x": float(direction[0]),
            "light_y": float(direction[1]),
            "light_z": float(direction[2]),
            "light_azimuth_deg": math.degrees(azimuth),
            "light_elevation_deg": math.degrees(elevation),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        z_value = float(_unit3(z)[2])
        label_id = _linear_bin(z_value, (-1.0, 1.0), bins=num_bins)
        return {
            "label": f"light_z_bin_{label_id:02d}",
            "label_id": str(label_id),
            "light_z_bin": str(label_id),
        }

@dataclass(frozen=True)
class BoundedTranslation(LatentFactorSpace):
    """Bounded Euclidean translation box."""

    dim: int = 2
    bounds: tuple[tuple[float, float], ...] | tuple[float, float] = (-0.5, 0.5)
    name: str | None = None
    clip_retraction: bool = True
    factor_names: list[str] = None  # type: ignore[assignment]
    factor_dims: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        bounds = _bounds_array(self.bounds, dim=self.dim)
        object.__setattr__(self, "bounds", tuple((float(a), float(b)) for a, b in bounds))
        if self.name is None:
            object.__setattr__(self, "name", f"translation_{'xyz'[: self.dim]}")
        object.__setattr__(self, "factor_names", [str(self.name)])
        object.__setattr__(self, "factor_dims", [self.dim])

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        rng = np.random.default_rng(seed)
        lows = np.asarray([bound[0] for bound in self.bounds], dtype=np.float32)
        highs = np.asarray([bound[1] for bound in self.bounds], dtype=np.float32)
        values = rng.uniform(lows, highs, size=(int(n), self.dim)).astype(np.float32)
        return LatentSample(values=values, metadata=_columns(self, values))

    def tangent_basis(self, z: Any) -> np.ndarray:
        return np.eye(self.dim, dtype=np.float32)

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        return _axis_labels(str(self.name), int(self.dim))

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> np.ndarray:
        value = np.asarray(z, dtype=np.float32) + float(eps) * np.asarray(
            tangent_vec,
            dtype=np.float32,
        )
        if not self.clip_retraction:
            return value
        lows = np.asarray([bound[0] for bound in self.bounds], dtype=np.float32)
        highs = np.asarray([bound[1] for bound in self.bounds], dtype=np.float32)
        return np.clip(value, lows, highs).astype(np.float32)

    def distance(self, z1: Any, z2: Any) -> float:
        delta = np.asarray(z1, dtype=np.float32) - np.asarray(z2, dtype=np.float32)
        return float(np.linalg.norm(delta))

    def coordinates(self, z: Any) -> dict[str, float]:
        value = np.asarray(z, dtype=np.float32).reshape(-1)
        keys = ("translation_x", "translation_y", "translation_z")
        return {keys[index]: float(value[index]) for index in range(min(self.dim, 3))}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        value = np.asarray(z, dtype=np.float32).reshape(-1)
        bins_per_axis = max(1, int(round(num_bins ** (1.0 / max(1, self.dim)))))
        label_id = 0
        multiplier = 1
        result: dict[str, str] = {}
        for index in range(self.dim):
            axis_bin = _linear_bin(float(value[index]), self.bounds[index], bins=bins_per_axis)
            result[f"translation_{'xyz'[index]}_bin"] = str(axis_bin)
            label_id += multiplier * axis_bin
            multiplier *= bins_per_axis
        result["label"] = f"{self.name}_bin_{label_id:02d}"
        result["label_id"] = str(label_id)
        return result


@dataclass(frozen=True)
class CameraRadiusInterval(LatentFactorSpace):
    """One-dimensional bounded camera radius factor."""

    bounds: tuple[float, float] = (2.0, 8.0)
    name: str = "camera_radius"
    dim: int = 1
    factor_names: list[str] = None  # type: ignore[assignment]
    factor_dims: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _validate_range(self.bounds, "bounds")
        object.__setattr__(self, "factor_names", [self.name])
        object.__setattr__(self, "factor_dims", [self.dim])

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        rng = np.random.default_rng(seed)
        values = rng.uniform(self.bounds[0], self.bounds[1], size=int(n)).astype(np.float32)
        return LatentSample(values=values, metadata=_columns(self, values))

    def tangent_basis(self, z: Any) -> np.ndarray:
        return np.asarray([[1.0]], dtype=np.float32)

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        return [str(self.name)]

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> float:
        step = float(np.asarray(tangent_vec).reshape(-1)[0])
        return float(np.clip(float(z) + float(eps) * step, self.bounds[0], self.bounds[1]))

    def distance(self, z1: Any, z2: Any) -> float:
        return float(abs(float(z1) - float(z2)))

    def coordinates(self, z: Any) -> dict[str, float]:
        return {"camera_radius": float(z)}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"camera_radius_bin_{label_id:02d}",
            "label_id": str(label_id),
            "camera_radius_bin": str(label_id),
        }


@dataclass(frozen=True)
class CameraDepthTranslationInterval(LatentFactorSpace):
    """One-dimensional camera-frame depth translation around a base radius."""

    bounds: tuple[float, float] = (-1.5, 2.5)
    name: str = "camera_depth_translation"
    dim: int = 1
    factor_names: list[str] = None  # type: ignore[assignment]
    factor_dims: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _validate_range(self.bounds, "bounds")
        object.__setattr__(self, "factor_names", [self.name])
        object.__setattr__(self, "factor_dims", [self.dim])

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        rng = np.random.default_rng(seed)
        values = rng.uniform(self.bounds[0], self.bounds[1], size=int(n)).astype(np.float32)
        return LatentSample(values=values, metadata=_columns(self, values))

    def tangent_basis(self, z: Any) -> np.ndarray:
        return np.asarray([[1.0]], dtype=np.float32)

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> float:
        step = float(np.asarray(tangent_vec).reshape(-1)[0])
        return float(np.clip(float(z) + float(eps) * step, self.bounds[0], self.bounds[1]))

    def distance(self, z1: Any, z2: Any) -> float:
        return float(abs(float(z1) - float(z2)))

    def coordinates(self, z: Any) -> dict[str, float]:
        value = float(z)
        return {
            "camera_depth_delta": value,
            "translation_z": value,
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"camera_depth_bin_{label_id:02d}",
            "label_id": str(label_id),
            "camera_depth_bin": str(label_id),
            "translation_z_bin": str(label_id),
        }


@dataclass(frozen=True)
class ZoomInterval(CameraRadiusInterval):
    """One-dimensional bounded focal-length/zoom factor."""

    bounds: tuple[float, float] = (40.0, 100.0)
    name: str = "zoom"

    def coordinates(self, z: Any) -> dict[str, float]:
        return {"zoom": float(z), "focal_length": float(z)}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"zoom_bin_{label_id:02d}",
            "label_id": str(label_id),
            "zoom_bin": str(label_id),
        }


@dataclass(frozen=True)
class CameraRollInterval(CameraRadiusInterval):
    """Bounded roll angle around the camera optical axis, in radians."""

    bounds: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0)
    name: str = "camera_roll"

    def coordinates(self, z: Any) -> dict[str, float]:
        value = float(z)
        return {"camera_roll_rad": value, "camera_roll_deg": math.degrees(value)}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"camera_roll_bin_{label_id:02d}",
            "label_id": str(label_id),
            "camera_roll_bin": str(label_id),
        }


@dataclass(frozen=True)
class CameraLocalTranslation(BoundedTranslation):
    """Bounded translation in the current camera frame."""

    dim: int = 3
    bounds: tuple[tuple[float, float], ...] | tuple[float, float] = (
        (-0.6, 0.6),
        (-0.6, 0.6),
        (-1.0, 1.0),
    )
    name: str | None = "camera_translation_xyz"

    def coordinates(self, z: Any) -> dict[str, float]:
        value = np.asarray(z, dtype=np.float32).reshape(-1)
        return {
            "camera_translation_x": float(value[0]),
            "camera_translation_y": float(value[1]),
            "camera_translation_z": float(value[2]),
            "translation_x": float(value[0]),
            "translation_y": float(value[1]),
            "translation_z": float(value[2]),
        }


@dataclass(frozen=True)
class CameraLogFocalScaleInterval(CameraRadiusInterval):
    """Log focal-length scale around the renderer's base focal length."""

    bounds: tuple[float, float] = (-0.4, 0.4)
    name: str = "camera_log_focal_scale"

    def coordinates(self, z: Any) -> dict[str, float]:
        value = float(z)
        return {
            "camera_log_focal_scale": value,
            "camera_focal_scale": math.exp(value),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"camera_focal_scale_bin_{label_id:02d}",
            "label_id": str(label_id),
            "camera_focal_scale_bin": str(label_id),
        }


@dataclass(frozen=True)
class CameraLogAspectRatioInterval(CameraRadiusInterval):
    """Log camera aspect ratio, log(fx / fy)."""

    bounds: tuple[float, float] = (-0.25, 0.25)
    name: str = "camera_log_aspect_ratio"

    def coordinates(self, z: Any) -> dict[str, float]:
        value = float(z)
        return {
            "camera_log_aspect_ratio": value,
            "camera_aspect_ratio": math.exp(value),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"camera_aspect_ratio_bin_{label_id:02d}",
            "label_id": str(label_id),
            "camera_aspect_ratio_bin": str(label_id),
        }


@dataclass(frozen=True)
class CameraPrincipalPointOffset(BoundedTranslation):
    """Principal-point offset in image pixels relative to image center."""

    dim: int = 2
    bounds: tuple[tuple[float, float], ...] | tuple[float, float] = (-8.0, 8.0)
    name: str | None = "camera_principal_point"

    def coordinates(self, z: Any) -> dict[str, float]:
        value = np.asarray(z, dtype=np.float32).reshape(-1)
        return {
            "camera_principal_point_offset_x": float(value[0]),
            "camera_principal_point_offset_y": float(value[1]),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        value = np.asarray(z, dtype=np.float32).reshape(-1)
        bins_per_axis = max(1, int(round(num_bins ** 0.5)))
        x_bin = _linear_bin(float(value[0]), self.bounds[0], bins=bins_per_axis)
        y_bin = _linear_bin(float(value[1]), self.bounds[1], bins=bins_per_axis)
        label_id = x_bin + bins_per_axis * y_bin
        return {
            "label": f"camera_principal_point_bin_{label_id:02d}",
            "label_id": str(label_id),
            "camera_principal_point_x_bin": str(x_bin),
            "camera_principal_point_y_bin": str(y_bin),
        }


@dataclass(frozen=True)
class CameraSkewInterval(CameraRadiusInterval):
    """Camera skew as a dimensionless fraction of focal length."""

    bounds: tuple[float, float] = (-0.15, 0.15)
    name: str = "camera_skew"

    def coordinates(self, z: Any) -> dict[str, float]:
        return {"camera_skew_ratio": float(z)}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"camera_skew_bin_{label_id:02d}",
            "label_id": str(label_id),
            "camera_skew_bin": str(label_id),
        }


@dataclass(frozen=True)
class LightLogEnergyInterval(CameraRadiusInterval):
    """Log scale for directional light energy around the renderer default."""

    bounds: tuple[float, float] = (-0.8, 0.8)
    name: str = "light_log_energy"

    def coordinates(self, z: Any) -> dict[str, float]:
        value = float(z)
        return {
            "light_log_energy": value,
            "light_energy_scale": math.exp(value),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"light_energy_bin_{label_id:02d}",
            "label_id": str(label_id),
            "light_energy_bin": str(label_id),
        }


@dataclass(frozen=True)
class AmbientLightInterval(CameraRadiusInterval):
    """Ambient illumination coefficient."""

    bounds: tuple[float, float] = (0.05, 0.75)
    name: str = "ambient_light"

    def coordinates(self, z: Any) -> dict[str, float]:
        return {"ambient_light": float(z)}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"ambient_light_bin_{label_id:02d}",
            "label_id": str(label_id),
            "ambient_light_bin": str(label_id),
        }


@dataclass(frozen=True)
class DiffuseLightInterval(CameraRadiusInterval):
    """Diffuse Lambertian illumination coefficient."""

    bounds: tuple[float, float] = (0.05, 1.10)
    name: str = "diffuse_light"

    def coordinates(self, z: Any) -> dict[str, float]:
        return {"diffuse_light": float(z)}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"diffuse_light_bin_{label_id:02d}",
            "label_id": str(label_id),
            "diffuse_light_bin": str(label_id),
        }


@dataclass(frozen=True)
class ImageLogExposureInterval(CameraRadiusInterval):
    """Log post-render image exposure scale."""

    bounds: tuple[float, float] = (-0.6, 0.4)
    name: str = "image_log_exposure"

    def coordinates(self, z: Any) -> dict[str, float]:
        value = float(z)
        return {
            "image_log_exposure": value,
            "image_exposure_scale": math.exp(value),
        }

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        label_id = _linear_bin(float(z), self.bounds, bins=num_bins)
        return {
            "label": f"image_exposure_bin_{label_id:02d}",
            "label_id": str(label_id),
            "image_exposure_bin": str(label_id),
        }


@dataclass(frozen=True)
class OrientationSO3(LatentFactorSpace):
    """Optional SO(3) object-orientation factor under a fixed/constrained camera."""

    name: str = "object_orientation_so3"
    dim: int = 3
    factor_names: list[str] = None  # type: ignore[assignment]
    factor_dims: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "factor_names", [self.name])
        object.__setattr__(self, "factor_dims", [self.dim])

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        try:
            from scipy.stats import special_ortho_group
        except ImportError as exc:  # pragma: no cover - scipy is a project dependency
            raise RuntimeError("OrientationSO3 sampling requires scipy.") from exc
        rng = np.random.default_rng(seed)
        values = special_ortho_group.rvs(3, size=int(n), random_state=rng)
        values = np.asarray(values, dtype=np.float32).reshape(int(n), 3, 3)
        return LatentSample(values=values, metadata=_columns(self, values))

    def tangent_basis(self, z: Any) -> np.ndarray:
        return np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        return [f"{self.name}_{axis}" for axis in ("x", "y", "z")]

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> np.ndarray:
        try:
            from scipy.linalg import expm
        except ImportError as exc:  # pragma: no cover - scipy is a project dependency
            raise RuntimeError("OrientationSO3 retraction requires scipy.") from exc
        rotation = np.asarray(z, dtype=np.float32).reshape(3, 3)
        update = expm(float(eps) * np.asarray(tangent_vec, dtype=np.float32))
        return np.asarray(rotation @ update, dtype=np.float32)

    def distance(self, z1: Any, z2: Any) -> float:
        relative = np.asarray(z1, dtype=np.float32).T @ np.asarray(z2, dtype=np.float32)
        cosine = (float(np.trace(relative)) - 1.0) / 2.0
        return float(math.acos(min(1.0, max(-1.0, cosine))))

    def coordinates(self, z: Any) -> dict[str, float]:
        rotation = np.asarray(z, dtype=np.float32).reshape(3, 3)
        cosine = (float(np.trace(rotation)) - 1.0) / 2.0
        angle = math.acos(min(1.0, max(-1.0, cosine)))
        return {"rotation_angle_deg": math.degrees(angle)}

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        angle = self.coordinates(z)["rotation_angle_deg"]
        label_id = _linear_bin(angle, (0.0, 180.0), bins=num_bins)
        return {
            "label": f"so3_angle_bin_{label_id:02d}",
            "label_id": str(label_id),
            "so3_angle_bin": str(label_id),
        }


class ProductFactorSpace(LatentFactorSpace):
    """Cartesian product of independently sampled semantic factor spaces."""

    def __init__(self, factors: list[LatentFactorSpace], name: str | None = None):
        if not factors:
            raise ValueError("ProductFactorSpace requires at least one factor.")
        self.factors = list(factors)
        self.factor_keys = _unique_names([factor.name for factor in self.factors])
        self.name = name or "product__" + "__".join(self.factor_keys)
        self.dim = int(sum(factor.dim for factor in self.factors))
        self.factor_names = list(self.factor_keys)
        self.factor_dims = [int(factor.dim) for factor in self.factors]
        self.factor_slices: dict[str, slice] = {}
        start = 0
        for key, factor in zip(self.factor_keys, self.factors, strict=True):
            stop = start + int(factor.dim)
            self.factor_slices[key] = slice(start, stop)
            start = stop

    def sample(self, n: int, seed: int | None = None) -> LatentSample:
        rng = np.random.default_rng(seed)
        component_samples = [
            factor.sample(int(n), seed=int(rng.integers(0, np.iinfo(np.int32).max)))
            for factor in self.factors
        ]
        values = []
        for row_id in range(int(n)):
            state = {}
            for key, sample in zip(self.factor_keys, component_samples, strict=True):
                state[key] = _value_at(sample.values, row_id)
            values.append(state)
        metadata: dict[str, Any] = {}
        for key, factor, sample in zip(
            self.factor_keys,
            self.factors,
            component_samples,
            strict=True,
        ):
            for column, column_values in sample.metadata.items():
                target = column if column not in metadata else f"{key}_{column}"
                metadata[target] = column_values
            metadata[f"{key}_dim"] = [int(factor.dim)] * int(n)
        return LatentSample(values=values, metadata=metadata)

    def tangent_basis(self, z: Any) -> list[dict[str, Any]]:
        basis: list[dict[str, Any]] = []
        for key, factor in zip(self.factor_keys, self.factors, strict=True):
            for tangent in factor.tangent_basis(z[key]):
                basis.append({key: tangent})
        return basis

    def tangent_labels(self, z: Any | None = None) -> list[str]:
        labels: list[tuple[str, str]] = []
        for key, factor in zip(self.factor_keys, self.factors, strict=True):
            component_z = z[key] if z is not None else None
            labels.extend((key, label) for label in factor.tangent_labels(component_z))
        counts: dict[str, int] = {}
        for _, label in labels:
            counts[label] = counts.get(label, 0) + 1
        resolved = [
            f"{key}_{label}" if counts[label] > 1 else label
            for key, label in labels
        ]
        return _unique_names(resolved)

    def retract(self, z: Any, tangent_vec: Any, eps: float) -> dict[str, Any]:
        result = dict(z)
        for key, tangent in dict(tangent_vec).items():
            factor = self._factor_for_key(key)
            result[key] = factor.retract(z[key], tangent, eps)
        return result

    def distance(self, z1: Any, z2: Any) -> float:
        total = 0.0
        for key, factor in zip(self.factor_keys, self.factors, strict=True):
            total += factor.distance(z1[key], z2[key]) ** 2
        return float(math.sqrt(total))

    def coordinates(self, z: Any) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, factor in zip(self.factor_keys, self.factors, strict=True):
            for column, value in factor.coordinates(z[key]).items():
                target = column if column not in result else f"{key}_{column}"
                result[target] = float(value)
        return result

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        result: dict[str, str] = {}
        label_parts = []
        for key, factor in zip(self.factor_keys, self.factors, strict=True):
            bins = factor.bins(z[key], num_bins=num_bins)
            label_parts.append(str(bins.get("label", key)))
            for column, value in bins.items():
                if column in {"label", "label_id"}:
                    result[f"{key}_{column}"] = str(value)
                else:
                    target = column if column not in result else f"{key}_{column}"
                    result[target] = str(value)
        result["label"] = "__".join(label_parts)
        result["label_id"] = str(abs(hash(result["label"])) % 1_000_000)
        return result

    def _factor_for_key(self, key: str) -> LatentFactorSpace:
        for current, factor in zip(self.factor_keys, self.factors, strict=True):
            if current == key:
                return factor
        raise KeyError(key)


class CameraSE3Factor(ProductFactorSpace):
    """Look-at view sphere, bounded camera roll, and local camera translation."""

    def __init__(
        self,
        *,
        roll_bounds: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0),
        translation_bounds: (
            tuple[tuple[float, float], ...] | tuple[float, float]
        ) = ((-0.6, 0.6), (-0.6, 0.6), (-1.0, 1.0)),
        name: str = "camera_se3",
    ) -> None:
        super().__init__(
            [
                LookAtViewSphere(),
                CameraRollInterval(bounds=roll_bounds),
                CameraLocalTranslation(bounds=translation_bounds),
            ],
            name=name,
        )


class CameraIntrinsicsFactor(ProductFactorSpace):
    """Five-DoF pinhole intrinsic matrix factor K."""

    def __init__(
        self,
        *,
        focal_log_bounds: tuple[float, float] = (-0.4, 0.4),
        aspect_log_bounds: tuple[float, float] = (-0.25, 0.25),
        principal_point_bounds: (
            tuple[tuple[float, float], ...] | tuple[float, float]
        ) = (-8.0, 8.0),
        skew_bounds: tuple[float, float] = (-0.15, 0.15),
        name: str = "camera_intrinsics_k",
    ) -> None:
        super().__init__(
            [
                CameraLogFocalScaleInterval(bounds=focal_log_bounds),
                CameraLogAspectRatioInterval(bounds=aspect_log_bounds),
                CameraPrincipalPointOffset(bounds=principal_point_bounds),
                CameraSkewInterval(bounds=skew_bounds),
            ],
            name=name,
        )


class FullCameraFactor(ProductFactorSpace):
    """Flat 11-DoF camera factor: view, roll, translation, and K intrinsics."""

    def __init__(
        self,
        *,
        roll_bounds: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0),
        translation_bounds: (
            tuple[tuple[float, float], ...] | tuple[float, float]
        ) = ((-0.6, 0.6), (-0.6, 0.6), (-1.0, 1.0)),
        focal_log_bounds: tuple[float, float] = (-0.4, 0.4),
        aspect_log_bounds: tuple[float, float] = (-0.25, 0.25),
        principal_point_bounds: (
            tuple[tuple[float, float], ...] | tuple[float, float]
        ) = (-8.0, 8.0),
        skew_bounds: tuple[float, float] = (-0.15, 0.15),
        name: str = "full_camera",
    ) -> None:
        super().__init__(
            [
                LookAtViewSphere(),
                CameraRollInterval(bounds=roll_bounds),
                CameraLocalTranslation(bounds=translation_bounds),
                CameraLogFocalScaleInterval(bounds=focal_log_bounds),
                CameraLogAspectRatioInterval(bounds=aspect_log_bounds),
                CameraPrincipalPointOffset(bounds=principal_point_bounds),
                CameraSkewInterval(bounds=skew_bounds),
            ],
            name=name,
        )


class IlluminationFactor(ProductFactorSpace):
    """Light direction plus local illumination coefficients."""

    def __init__(
        self,
        *,
        energy_log_bounds: tuple[float, float] = (-0.8, 0.8),
        ambient_bounds: tuple[float, float] = (0.05, 0.75),
        diffuse_bounds: tuple[float, float] = (0.05, 1.10),
        name: str = "illumination",
    ) -> None:
        super().__init__(
            [
                LightingDirectionSphere(),
                LightLogEnergyInterval(bounds=energy_log_bounds),
                AmbientLightInterval(bounds=ambient_bounds),
                DiffuseLightInterval(bounds=diffuse_bounds),
            ],
            name=name,
        )


class PhotometryFactor(ProductFactorSpace):
    """Lighting strength and image-response controls with fixed light direction."""

    def __init__(
        self,
        *,
        energy_log_bounds: tuple[float, float] = (-0.8, 0.8),
        ambient_bounds: tuple[float, float] = (0.05, 0.75),
        diffuse_bounds: tuple[float, float] = (0.05, 1.10),
        exposure_log_bounds: tuple[float, float] = (-0.6, 0.4),
        name: str = "photometry",
    ) -> None:
        super().__init__(
            [
                LightLogEnergyInterval(bounds=energy_log_bounds),
                AmbientLightInterval(bounds=ambient_bounds),
                DiffuseLightInterval(bounds=diffuse_bounds),
                ImageLogExposureInterval(bounds=exposure_log_bounds),
            ],
            name=name,
        )


class FullAppearanceFactor(ProductFactorSpace):
    """Light direction, illumination strength, and image exposure."""

    def __init__(
        self,
        *,
        energy_log_bounds: tuple[float, float] = (-0.8, 0.8),
        ambient_bounds: tuple[float, float] = (0.05, 0.75),
        diffuse_bounds: tuple[float, float] = (0.05, 1.10),
        exposure_log_bounds: tuple[float, float] = (-0.6, 0.4),
        name: str = "full_appearance",
    ) -> None:
        super().__init__(
            [
                LightingDirectionSphere(),
                LightLogEnergyInterval(bounds=energy_log_bounds),
                AmbientLightInterval(bounds=ambient_bounds),
                DiffuseLightInterval(bounds=diffuse_bounds),
                ImageLogExposureInterval(bounds=exposure_log_bounds),
            ],
            name=name,
        )


def sample_values(sample: LatentSample) -> list[Any]:
    """Return sampled values as a Python list regardless of representation."""

    values = sample.values
    if isinstance(values, list):
        return values
    array = np.asarray(values)
    return [_value_at(array, index) for index in range(len(array))]


def _columns(space: LatentFactorSpace, values: Any) -> dict[str, list[Any]]:
    rows = [_value_at(values, index) for index in range(len(values))]
    metadata: dict[str, list[Any]] = {}
    for row in rows:
        row_values = {**space.coordinates(row), **space.bins(row)}
        for key, value in row_values.items():
            metadata.setdefault(key, []).append(value)
    return metadata


def _value_at(values: Any, index: int) -> Any:
    if isinstance(values, list):
        return values[index]
    array = np.asarray(values)
    value = array[index]
    if np.isscalar(value) or getattr(value, "shape", ()) == ():
        return float(value)
    return np.asarray(value, dtype=np.float32)


def _unit3(value: Any) -> np.ndarray:
    return _normalize(np.asarray(value, dtype=np.float32).reshape(3))


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return np.asarray(value, dtype=np.float32) / norm


def _bounds_array(
    bounds: tuple[tuple[float, float], ...] | tuple[float, float],
    *,
    dim: int,
) -> list[tuple[float, float]]:
    if len(bounds) == 2 and all(isinstance(item, int | float) for item in bounds):
        low, high = float(bounds[0]), float(bounds[1])  # type: ignore[index]
        _validate_range((low, high), "bounds")
        return [(low, high)] * int(dim)
    if len(bounds) != dim:
        raise ValueError(f"bounds must contain 2 values or {dim} ranges.")
    result = []
    for index, item in enumerate(bounds):
        if not isinstance(item, tuple | list) or len(item) != 2:
            raise ValueError(f"bounds[{index}] must be a pair.")
        pair = (float(item[0]), float(item[1]))
        _validate_range(pair, f"bounds[{index}]")
        result.append(pair)
    return result


def _validate_range(value: tuple[float, float], name: str) -> None:
    if float(value[1]) <= float(value[0]):
        raise ValueError(f"{name} must be increasing.")


def _linear_bin(value: float, bounds: tuple[float, float], *, bins: int) -> int:
    low, high = float(bounds[0]), float(bounds[1])
    fraction = (float(value) - low) / max(high - low, 1.0e-12)
    clipped = min(1.0, max(0.0, fraction))
    return min(int(bins) - 1, int(math.floor(clipped * int(bins))))


def _cyclic_bin(value: float, *, bins: int) -> int:
    fraction = (float(value) % (2.0 * math.pi)) / (2.0 * math.pi)
    return min(int(bins) - 1, int(math.floor(fraction * int(bins))))


def _axis_labels(name: str, dim: int) -> list[str]:
    if int(dim) == 1:
        return [str(name)]
    suffixes = ["x", "y", "z", "w", "v", "u"]
    if int(dim) > len(suffixes):
        suffixes.extend(str(index) for index in range(len(suffixes), int(dim)))
    base = str(name)
    for compact_suffix in ("xyz", "xy"):
        marker = f"_{compact_suffix}"
        if base.endswith(marker) and int(dim) <= len(compact_suffix):
            base = base[: -len(marker)]
            break
    return [f"{base}_{suffixes[index]}" for index in range(int(dim))]


def _unique_names(names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result = []
    for name in names:
        count = counts.get(name, 0)
        counts[name] = count + 1
        result.append(name if count == 0 else f"{name}_{count + 1}")
    return result
