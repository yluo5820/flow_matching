"""Render-map adapters from semantic latent factors to synthetic images."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fm_lab.geometry_explorer.latent_factors import (
    AmbientLightInterval,
    AzimuthCircle,
    BoundedLookAtView,
    BoundedTranslation,
    CameraDepthTranslationInterval,
    CameraLocalTranslation,
    CameraLogAspectRatioInterval,
    CameraLogFocalScaleInterval,
    CameraPrincipalPointOffset,
    CameraRadiusInterval,
    CameraRollInterval,
    CameraSkewInterval,
    DiffuseLightInterval,
    ImageLogExposureInterval,
    LatentFactorSpace,
    LightingDirectionSphere,
    LightLogEnergyInterval,
    LookAtViewSphere,
    OrientationSO3,
    ProductFactorSpace,
    ZoomInterval,
)
from fm_lab.geometry_explorer.synthetic_objects import (
    SUPPORTED_RENDER_MODES,
    SyntheticObjectSpec,
    SyntheticRenderConfig,
    _camera_frame,
    _camera_frame_from_position,
    _CameraFrame,
    _validated_rgb_triplet,
    render_synthetic_object,
)
from fm_lab.utils.config import ConfigError


@dataclass(frozen=True)
class RenderConfig:
    image_size: int = 64
    render_mode: str = "colored"
    background: str | tuple[float, float, float] = "white"
    antialias: bool = True
    normalize_pixels: bool = True
    object_config: Any | None = None
    camera_config: Any | None = None
    light_config: Any | None = None


@dataclass
class _RenderControls:
    azimuth_rad: float = 0.0
    camera_direction: np.ndarray | None = None
    camera_radius: float = 4.0
    camera_roll_rad: float = 0.0
    focal_length: float = 70.0
    aspect_ratio: float = 1.0
    principal_point_offset: np.ndarray | None = None
    skew_ratio: float = 0.0
    camera_plane_translation: np.ndarray | None = None
    object_translation: np.ndarray | None = None
    rotation_matrix: np.ndarray | None = None
    light_position: np.ndarray | None = None
    light_energy: float = 400.0
    ambient: float = 0.35
    diffuse: float = 0.70
    image_exposure: float = 1.0


class RenderMap:
    """Renderer-induced map F_O: Z -> R^{HWC} for a fixed object/config."""

    def __init__(
        self,
        factor_space: LatentFactorSpace,
        object_name: str,
        config: RenderConfig | None = None,
    ) -> None:
        self.factor_space = factor_space
        self.object_name = object_name
        self.config = config or RenderConfig()
        self.object_spec = _object_spec(object_name, self.config.object_config)
        self.camera_config = dict(self.config.camera_config or {})
        self.light_config = dict(self.config.light_config or {})
        render_mode = str(self.config.render_mode).lower()
        if render_mode not in SUPPORTED_RENDER_MODES:
            supported = ", ".join(sorted(SUPPORTED_RENDER_MODES))
            raise ConfigError(f"Unsupported render_mode {render_mode!r}: {supported}.")
        self.synthetic_render = SyntheticRenderConfig(
            image_size=int(self.config.image_size),
            elevation=float(self.camera_config.get("elevation", 20.0)),
            camera_distance=float(self.camera_config.get("radius", 4.0)),
            focal_length=float(self.camera_config.get("focal_length", 70.0)),
            background=_background_triplet(self.config.background),
            light_position=tuple(
                float(value)
                for value in self.light_config.get("position", (3.0, -4.0, 5.0))
            ),
            light_energy=float(self.light_config.get("energy", 400.0)),
            ambient=float(self.light_config.get("ambient", 0.35)),
            diffuse=float(self.light_config.get("diffuse", 0.70)),
            supersample=3 if self.config.antialias else 1,
        )

    @property
    def render_mode(self) -> str:
        return str(self.config.render_mode).lower()

    def render(self, z: Any) -> np.ndarray:
        controls = self._controls(z)
        render_config = replace(
            self.synthetic_render,
            light_energy=float(controls.light_energy),
            ambient=float(controls.ambient),
            diffuse=float(controls.diffuse),
        )
        image = render_synthetic_object(
            object_spec=self.object_spec,
            render=render_config,
            azimuth_deg=math.degrees(controls.azimuth_rad),
            rotation_matrix=controls.rotation_matrix,
            object_translation=controls.object_translation,
            camera_frame=self._camera_frame(controls),
            light_position=controls.light_position,
            focal_length=controls.focal_length,
            aspect_ratio=controls.aspect_ratio,
            principal_point_offset=controls.principal_point_offset,
            skew=controls.skew_ratio,
            render_mode=self.render_mode,
        )
        image = np.clip(
            np.asarray(image, dtype=np.float32) * float(controls.image_exposure),
            0.0,
            1.0,
        )
        if self.config.normalize_pixels:
            return np.asarray(image, dtype=np.float32)
        return np.rint(np.asarray(image, dtype=np.float32) * 255.0).astype(np.float32)

    def render_flat(self, z: Any) -> np.ndarray:
        return self.render(z).reshape(-1).astype(np.float32)

    def render_batch(self, zs: list[Any] | tuple[Any, ...], batch_size: int = 128) -> np.ndarray:
        del batch_size
        return np.asarray([self.render_flat(z) for z in zs], dtype=np.float32)

    def coordinates(self, z: Any) -> dict[str, float]:
        controls = self._controls(z)
        coordinates = dict(self.factor_space.coordinates(z))
        coordinates.update(
            {
                "true_latent_dim": float(self.factor_space.dim),
                "camera_radius": float(controls.camera_radius),
                "camera_roll_rad": float(controls.camera_roll_rad),
                "camera_roll_deg": math.degrees(float(controls.camera_roll_rad)),
                "focal_length": float(controls.focal_length),
                "camera_aspect_ratio": float(controls.aspect_ratio),
                "camera_skew_ratio": float(controls.skew_ratio),
                "light_x": float(controls.light_position[0]),
                "light_y": float(controls.light_position[1]),
                "light_z": float(controls.light_position[2]),
                "light_energy": float(controls.light_energy),
                "ambient": float(controls.ambient),
                "diffuse": float(controls.diffuse),
                "image_exposure": float(controls.image_exposure),
            }
        )
        if controls.principal_point_offset is not None:
            point = controls.principal_point_offset
            coordinates.update(
                {
                    "camera_principal_point_offset_x": float(point[0]),
                    "camera_principal_point_offset_y": float(point[1]),
                }
            )
        if controls.camera_plane_translation is not None:
            value = controls.camera_plane_translation
            coordinates.update(
                {
                    "camera_plane_translation_x": float(value[0]),
                    "camera_plane_translation_y": float(value[1]),
                    "camera_plane_translation_z": float(value[2]),
                }
            )
        return coordinates

    def bins(self, z: Any, num_bins: int = 36) -> dict[str, str]:
        return self.factor_space.bins(z, num_bins=num_bins)

    def _controls(self, z: Any) -> _RenderControls:
        controls = _RenderControls(
            camera_radius=float(self.synthetic_render.camera_distance),
            focal_length=float(self.synthetic_render.focal_length),
            light_position=np.asarray(self.synthetic_render.light_position, dtype=np.float32),
            light_energy=float(self.synthetic_render.light_energy),
            ambient=float(self.synthetic_render.ambient),
            diffuse=float(self.synthetic_render.diffuse),
        )
        self._apply_factor(self.factor_space, z, controls)
        return controls

    def _apply_factor(
        self,
        factor: LatentFactorSpace,
        z: Any,
        controls: _RenderControls,
    ) -> None:
        if isinstance(factor, ProductFactorSpace):
            for key, component in zip(factor.factor_keys, factor.factors, strict=True):
                self._apply_factor(component, z[key], controls)
            return
        if isinstance(factor, AzimuthCircle):
            controls.azimuth_rad = float(z)
            return
        if isinstance(factor, LightingDirectionSphere):
            distance = float(self.light_config.get("distance", 6.0))
            controls.light_position = _unit3(z) * distance
            return
        if isinstance(factor, LightLogEnergyInterval):
            controls.light_energy = float(self.synthetic_render.light_energy) * math.exp(
                float(z)
            )
            return
        if isinstance(factor, AmbientLightInterval):
            controls.ambient = float(z)
            return
        if isinstance(factor, DiffuseLightInterval):
            controls.diffuse = float(z)
            return
        if isinstance(factor, ImageLogExposureInterval):
            controls.image_exposure = math.exp(float(z))
            return
        if isinstance(factor, BoundedLookAtView):
            azimuth, sin_elevation = np.asarray(z, dtype=np.float64)
            elevation = math.asin(float(np.clip(sin_elevation, -1.0, 1.0)))
            cos_elevation = math.cos(elevation)
            controls.camera_direction = np.asarray(
                [
                    cos_elevation * math.cos(float(azimuth)),
                    cos_elevation * math.sin(float(azimuth)),
                    math.sin(elevation),
                ],
                dtype=np.float64,
            )
            return
        if isinstance(factor, LookAtViewSphere):
            controls.camera_direction = _unit3(z)
            return
        if isinstance(factor, CameraRollInterval):
            controls.camera_roll_rad = float(z)
            return
        if isinstance(factor, CameraLogFocalScaleInterval):
            controls.focal_length = float(self.synthetic_render.focal_length) * math.exp(float(z))
            return
        if isinstance(factor, CameraLogAspectRatioInterval):
            controls.aspect_ratio = math.exp(float(z))
            return
        if isinstance(factor, CameraPrincipalPointOffset):
            controls.principal_point_offset = np.asarray(z, dtype=np.float32).reshape(2)
            return
        if isinstance(factor, CameraSkewInterval):
            controls.skew_ratio = float(z)
            return
        if isinstance(factor, ZoomInterval):
            controls.focal_length = float(z)
            return
        if isinstance(factor, CameraDepthTranslationInterval):
            controls.camera_radius += float(z)
            return
        if isinstance(factor, CameraRadiusInterval):
            controls.camera_radius = float(z)
            return
        if isinstance(factor, CameraLocalTranslation | BoundedTranslation):
            self._apply_translation(factor, z, controls)
            return
        if isinstance(factor, OrientationSO3):
            controls.rotation_matrix = np.asarray(z, dtype=np.float32).reshape(3, 3)
            return

    def _apply_translation(
        self,
        factor: BoundedTranslation,
        z: Any,
        controls: _RenderControls,
    ) -> None:
        values = np.asarray(z, dtype=np.float32).reshape(-1)
        padded = np.zeros(3, dtype=np.float32)
        padded[: min(3, len(values))] = values[:3]
        target = str(
            self.camera_config.get(
                "translation_target",
                self.camera_config.get("translation_mode", "camera_plane"),
            )
        )
        if str(factor.name).startswith("object_") or target == "object":
            controls.object_translation = padded
            return
        controls.camera_plane_translation = padded

    def _camera_frame(self, controls: _RenderControls) -> _CameraFrame:
        if controls.camera_direction is not None:
            base = _camera_frame_from_position(
                _unit3(controls.camera_direction) * float(controls.camera_radius)
            )
        else:
            base = _camera_frame(
                azimuth_deg=math.degrees(float(controls.azimuth_rad)),
                elevation_deg=float(self.synthetic_render.elevation),
                distance=float(controls.camera_radius),
            )
        base = _roll_camera_frame(base, float(controls.camera_roll_rad))
        if controls.camera_plane_translation is None:
            return base
        shift = np.asarray(controls.camera_plane_translation, dtype=np.float32)
        depth = max(1.0e-6, float(controls.camera_radius) + float(shift[2]))
        position = -base.forward * depth + base.right * float(shift[0]) + base.up * float(shift[1])
        return _CameraFrame(
            position=position.astype(np.float32),
            right=base.right,
            up=base.up,
            forward=base.forward,
        )


def _roll_camera_frame(frame: _CameraFrame, roll_rad: float) -> _CameraFrame:
    if math.isclose(float(roll_rad), 0.0, abs_tol=1.0e-12):
        return frame
    cosine = math.cos(float(roll_rad))
    sine = math.sin(float(roll_rad))
    right = cosine * frame.right + sine * frame.up
    up = -sine * frame.right + cosine * frame.up
    return _CameraFrame(
        position=frame.position,
        right=_unit3(right),
        up=_unit3(up),
        forward=frame.forward,
    )


def _object_spec(object_name: str, raw: Any | None) -> SyntheticObjectSpec:
    values = dict(raw or {})
    aliases = {
        "cube": "marked_cube",
        "marked_cube": "marked_cube",
        "abstract_statue": "abstract_statue",
        "offset_monument": "offset_monument",
        "stepped_monument": "stepped_monument",
        "crooked_arch": "crooked_arch",
        "three_arm_vane": "three_arm_vane",
    }
    kind = aliases.get(str(object_name), str(object_name))
    return SyntheticObjectSpec(
        kind=str(values.get("kind", kind)),
        scale=float(values.get("scale", 1.35)),
        base_color=(
            None
            if values.get("base_color") is None
            else _validated_rgb_triplet(
                values["base_color"],
                name="object.base_color",
            )
        ),
        marker=bool(values.get("marker", kind == "marked_cube")),
        marker_face=str(values.get("marker_face", "negative_y")),
        marker_size=float(values.get("marker_size", 0.28)),
        marker_offset=tuple(values.get("marker_offset", (0.28, 0.24))),  # type: ignore[arg-type]
    )


def _background_triplet(value: str | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(value, str):
        colors = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "gray": (0.5, 0.5, 0.5),
            "grey": (0.5, 0.5, 0.5),
        }
        if value.lower() not in colors:
            raise ConfigError(f"Unsupported background color {value!r}.")
        return colors[value.lower()]
    if len(value) != 3:
        raise ConfigError("background must be a named color or RGB triplet.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _unit3(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("Expected nonzero 3D vector.")
    return vector / norm
