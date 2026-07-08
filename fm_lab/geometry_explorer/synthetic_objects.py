"""Lightweight analytic synthetic object dataset generation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.image_diagnostics.save_utils import write_parquet
from fm_lab.utils.config import ConfigError, save_config
from fm_lab.utils.logging import write_json

SUPPORTED_OBJECT_KINDS = {"marked_cube", "abstract_statue", "offset_monument"}
TRANSLATION_POSE_MODES = {"translation_xy", "translation_z", "translation_xyz"}
SUPPORTED_POSE_MODES = {"azimuth", "so3", "sphere", *TRANSLATION_POSE_MODES}


@dataclass(frozen=True)
class SyntheticObjectSpec:
    kind: str = "marked_cube"
    scale: float = 1.35
    marker: bool = True
    marker_face: str = "negative_y"
    marker_size: float = 0.28
    marker_offset: tuple[float, float] = (0.28, 0.24)


@dataclass(frozen=True)
class SyntheticRenderConfig:
    backend: str = "analytic"
    image_size: int = 64
    azimuth_start: float = 0.0
    azimuth_stop: float = 180.0
    azimuth_steps: int = 181
    elevation: float = 20.0
    camera_distance: float = 4.0
    focal_length: float = 70.0
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    light_position: tuple[float, float, float] = (3.0, -4.0, 5.0)
    light_energy: float = 400.0
    ambient: float = 0.35
    diffuse: float = 0.70
    supersample: int = 3
    azimuth_bins: int = 12


@dataclass(frozen=True)
class SyntheticPoseConfig:
    mode: str = "azimuth"
    samples: int | None = None
    orientation_bins: int = 36
    translation_bins: int = 36
    translation_x_range: tuple[float, float] = (-0.9, 0.9)
    translation_y_range: tuple[float, float] = (-0.9, 0.9)
    depth_range: tuple[float, float] = (2.0, 8.0)


@dataclass(frozen=True)
class SyntheticOutputConfig:
    value_range: tuple[float, float] = (0.0, 1.0)
    save_pngs: bool = True


@dataclass(frozen=True)
class SyntheticObjectBuildConfig:
    family: str = "synthetic_object"
    variant: str = "cube_pose_180"
    base: str = "analytic"
    split: str = "pose_sweep"
    seed: int = 42
    object_spec: SyntheticObjectSpec = field(default_factory=SyntheticObjectSpec)
    render: SyntheticRenderConfig = field(default_factory=SyntheticRenderConfig)
    pose: SyntheticPoseConfig = field(default_factory=SyntheticPoseConfig)
    output: SyntheticOutputConfig = field(default_factory=SyntheticOutputConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def variant_id(self) -> str:
        return f"{self.family}/{self.variant}"


def load_synthetic_object_config(path: str | Path) -> SyntheticObjectBuildConfig:
    from fm_lab.utils.config import load_config

    return synthetic_object_config_from_dict(load_config(path))


def synthetic_object_config_from_dict(raw: dict[str, Any]) -> SyntheticObjectBuildConfig:
    """Parse a synthetic-object dataset config."""

    values = dict(raw)
    family = str(values.get("family", "synthetic_object")).lower()
    if family != "synthetic_object":
        raise ConfigError("Synthetic object configs must set family: synthetic_object.")

    object_values = _section(values, "object")
    render_values = _section(values, "render")
    pose_values = _section(values, "pose")
    output_values = _section(values, "output")
    object_spec = SyntheticObjectSpec(
        kind=str(object_values.get("kind", "marked_cube")),
        scale=float(object_values.get("scale", 1.35)),
        marker=bool(object_values.get("marker", True)),
        marker_face=str(object_values.get("marker_face", "negative_y")),
        marker_size=float(object_values.get("marker_size", 0.28)),
        marker_offset=_float_pair(
            object_values.get("marker_offset", (0.28, 0.24)),
            name="object.marker_offset",
        ),
    )
    render = SyntheticRenderConfig(
        backend=str(render_values.get("backend", "analytic")),
        image_size=int(render_values.get("image_size", 64)),
        azimuth_start=float(render_values.get("azimuth_start", 0.0)),
        azimuth_stop=float(render_values.get("azimuth_stop", 180.0)),
        azimuth_steps=int(render_values.get("azimuth_steps", 181)),
        elevation=float(render_values.get("elevation", 20.0)),
        camera_distance=float(render_values.get("camera_distance", 4.0)),
        focal_length=float(render_values.get("focal_length", 70.0)),
        background=_float_triplet(
            render_values.get("background", (1.0, 1.0, 1.0)),
            name="render.background",
        ),
        light_position=_float_triplet(
            render_values.get("light_position", (3.0, -4.0, 5.0)),
            name="render.light_position",
        ),
        light_energy=float(render_values.get("light_energy", 400.0)),
        ambient=float(render_values.get("ambient", 0.35)),
        diffuse=float(render_values.get("diffuse", 0.70)),
        supersample=int(render_values.get("supersample", 3)),
        azimuth_bins=int(render_values.get("azimuth_bins", 12)),
    )
    pose = SyntheticPoseConfig(
        mode=str(pose_values.get("mode", "azimuth")).lower(),
        samples=(
            None
            if pose_values.get("samples") is None
            else int(pose_values["samples"])
        ),
        orientation_bins=int(pose_values.get("orientation_bins", 36)),
        translation_bins=int(pose_values.get("translation_bins", 36)),
        translation_x_range=_float_pair(
            pose_values.get("translation_x_range", (-0.9, 0.9)),
            name="pose.translation_x_range",
        ),
        translation_y_range=_float_pair(
            pose_values.get("translation_y_range", (-0.9, 0.9)),
            name="pose.translation_y_range",
        ),
        depth_range=_float_pair(
            pose_values.get("depth_range", (2.0, 8.0)),
            name="pose.depth_range",
        ),
    )
    output = SyntheticOutputConfig(
        value_range=_float_pair(
            output_values.get("value_range", (0.0, 1.0)),
            name="output.value_range",
        ),
        save_pngs=bool(output_values.get("save_pngs", True)),
    )
    config = SyntheticObjectBuildConfig(
        family=family,
        variant=str(values.get("variant", "cube_pose_180")),
        base=str(values.get("base", "analytic")),
        split=str(values.get("split", "pose_sweep")),
        seed=int(values.get("seed", 42)),
        object_spec=object_spec,
        render=render,
        pose=pose,
        output=output,
        raw=values,
    )
    validate_synthetic_object_config(config)
    return config


def validate_synthetic_object_config(config: SyntheticObjectBuildConfig) -> None:
    if config.object_spec.kind not in SUPPORTED_OBJECT_KINDS:
        supported = ", ".join(sorted(SUPPORTED_OBJECT_KINDS))
        raise ConfigError(f"Unsupported object.kind {config.object_spec.kind!r}: {supported}.")
    if config.object_spec.marker and config.object_spec.marker_face != "negative_y":
        raise ConfigError("Synthetic object MVP supports object.marker_face: negative_y.")
    if config.render.backend != "analytic":
        raise ConfigError("Synthetic object MVP supports render.backend: analytic.")
    if config.pose.mode not in SUPPORTED_POSE_MODES:
        supported = ", ".join(sorted(SUPPORTED_POSE_MODES))
        raise ConfigError(f"Unsupported pose.mode {config.pose.mode!r}: {supported}.")
    if config.pose.mode in {"so3", "sphere", *TRANSLATION_POSE_MODES}:
        if config.pose.samples is None or config.pose.samples < 1:
            raise ConfigError(
                f"pose.samples must be positive for {config.pose.mode} sampling."
            )
    if config.pose.mode in {"so3", "sphere"}:
        if config.pose.orientation_bins < 1:
            raise ConfigError("pose.orientation_bins must be positive.")
    if config.pose.mode in TRANSLATION_POSE_MODES:
        if config.pose.translation_bins < 1:
            raise ConfigError("pose.translation_bins must be positive.")
        _validate_increasing_range(
            config.pose.translation_x_range,
            name="pose.translation_x_range",
        )
        _validate_increasing_range(
            config.pose.translation_y_range,
            name="pose.translation_y_range",
        )
        _validate_increasing_range(config.pose.depth_range, name="pose.depth_range")
        if config.pose.depth_range[0] <= 0.0:
            raise ConfigError("pose.depth_range values must be positive.")
    if config.render.image_size < 8:
        raise ConfigError("render.image_size must be at least 8.")
    if config.render.azimuth_steps < 1:
        raise ConfigError("render.azimuth_steps must be positive.")
    if config.render.camera_distance <= 0:
        raise ConfigError("render.camera_distance must be positive.")
    if config.render.focal_length <= 0:
        raise ConfigError("render.focal_length must be positive.")
    if config.render.supersample < 1:
        raise ConfigError("render.supersample must be positive.")
    if config.render.azimuth_bins < 1:
        raise ConfigError("render.azimuth_bins must be positive.")
    if any(value < 0.0 or value > 1.0 for value in config.render.background):
        raise ConfigError("render.background values must be in [0, 1].")
    low, high = config.output.value_range
    if high <= low:
        raise ConfigError("output.value_range must be increasing.")


def build_synthetic_object_dataset(
    config: SyntheticObjectBuildConfig,
    *,
    workspace: str | Path = DEFAULT_WORKSPACE,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Render, save, and register one synthetic object dataset variant."""

    registry = GeometryRegistry(workspace)
    output_dir = registry.workspace / "datasets" / config.family / config.variant
    image_dir = output_dir / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.output.save_pngs:
        image_dir.mkdir(parents=True, exist_ok=True)

    pose_samples = _pose_samples(config)
    images: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for pose in pose_samples:
        image = render_marked_cube(
            object_spec=config.object_spec,
            render=config.render,
            azimuth_deg=float(pose.azimuth_deg),
            rotation_matrix=pose.rotation_matrix,
            camera_frame=_camera_frame_for_pose(pose, config.render),
        )
        images.append(image)
        image_path = ""
        if config.output.save_pngs:
            path = _image_path(image_dir, pose)
            _save_float_image(image, path)
            image_path = str(path.resolve())
        rows.append(
            _metadata_row(
                pose=pose,
                image_path=image_path,
                config=config,
            )
        )

    image_array = np.asarray(images, dtype=np.float32)
    vectors = image_array.reshape(len(image_array), -1)
    metadata = pd.DataFrame(rows)
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    dataset_path = write_parquet(metadata, output_dir / "dataset_index.parquet")
    data_path = output_dir / "data.npy"
    labels_path = output_dir / "labels.npy"
    np.save(data_path, vectors)
    np.save(labels_path, labels)
    save_config(_config_raw(config), output_dir / "config_used.yaml")

    label_counts = _label_counts(metadata)
    manifest = {
        "variant_id": config.variant_id,
        "family": config.family,
        "variant": config.variant,
        "base": config.base,
        "split": config.split,
        "pose_mode": config.pose.mode,
        "rows": int(len(metadata)),
        "label_counts": label_counts,
        "image_shape": [config.render.image_size, config.render.image_size, 3],
        "value_range": list(config.output.value_range),
        "dataset_path": str(dataset_path),
        "data_path": str(data_path),
        "labels_path": str(labels_path),
        "config_path": str(config_path or output_dir / "config_used.yaml"),
    }
    write_json(manifest, output_dir / "manifest.json")
    registry.register_dataset_variant(
        variant_id=config.variant_id,
        family=config.family,
        variant=config.variant,
        base=config.base,
        split=config.split,
        dataset_path=dataset_path,
        data_path=data_path,
        labels_path=labels_path,
        config_path=config_path or output_dir / "config_used.yaml",
        row_count=len(metadata),
        label_counts=label_counts,
        image_shape=(config.render.image_size, config.render.image_size, 3),
        value_range=config.output.value_range,
    )
    return {
        "variant_id": config.variant_id,
        "output_dir": output_dir,
        "dataset_path": dataset_path,
        "data_path": data_path,
        "labels_path": labels_path,
        "rows": len(metadata),
        "label_counts": label_counts,
    }


def render_marked_cube(
    *,
    object_spec: SyntheticObjectSpec | None = None,
    render: SyntheticRenderConfig | None = None,
    azimuth_deg: float = 0.0,
    rotation_matrix: np.ndarray | None = None,
    camera_position: np.ndarray | None = None,
    camera_frame: _CameraFrame | None = None,
) -> np.ndarray:
    """Render one analytic RGB synthetic object image in [0, 1]."""

    spec = object_spec or SyntheticObjectSpec()
    cfg = render or SyntheticRenderConfig()
    scale = cfg.supersample
    canvas_size = cfg.image_size * scale
    background = _rgb255(cfg.background)
    image = Image.new("RGB", (canvas_size, canvas_size), background)
    draw = ImageDraw.Draw(image)
    if camera_frame is not None:
        camera = camera_frame
    elif camera_position is not None:
        camera = _camera_frame_from_position(camera_position)
    else:
        camera = _camera_frame(
            azimuth_deg=azimuth_deg,
            elevation_deg=cfg.elevation,
            distance=cfg.camera_distance,
        )
    faces = _rotated_faces(_object_faces(spec), rotation_matrix)
    visible_faces = []
    for face in faces:
        if _is_face_visible(face, camera.position):
            projected, depth = _project_points(
                face.vertices,
                camera=camera,
                focal_px=cfg.focal_length * scale,
                image_size=canvas_size,
            )
            if projected is None:
                continue
            visible_faces.append((depth, face, projected))

    for _, face, projected in sorted(visible_faces, key=lambda item: item[0], reverse=True):
        color = _lit_color(face, cfg)
        draw.polygon([tuple(point) for point in projected], fill=color)
        if spec.kind == "marked_cube" and spec.marker and face.name == spec.marker_face:
            marker = _marker_polygon(face, spec)
            projected_marker, _ = _project_points(
                marker,
                camera=camera,
                focal_px=cfg.focal_length * scale,
                image_size=canvas_size,
            )
            if projected_marker is not None:
                draw.polygon([tuple(point) for point in projected_marker], fill=(20, 20, 20))

    if scale > 1:
        image = image.resize(
            (cfg.image_size, cfg.image_size),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(image, dtype=np.float32) / 255.0


@dataclass(frozen=True)
class _CameraFrame:
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray


@dataclass(frozen=True)
class _CubeFace:
    name: str
    vertices: np.ndarray
    normal: np.ndarray
    base_color: tuple[float, float, float]

    @property
    def center(self) -> np.ndarray:
        return np.mean(self.vertices, axis=0)


@dataclass(frozen=True)
class _PoseSample:
    row_id: int
    mode: str
    label: str
    label_id: int
    azimuth_deg: float
    rotation_matrix: np.ndarray | None = None
    rotation_angle_deg: float | None = None
    rotation_axis: tuple[float, float, float] | None = None
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    camera_direction: tuple[float, float, float] | None = None
    camera_azimuth_deg: float | None = None
    camera_elevation_deg: float | None = None
    translation_x: float | None = None
    translation_y: float | None = None
    translation_z: float | None = None
    camera_depth: float | None = None
    camera_position: tuple[float, float, float] | None = None
    object_center_u: float | None = None
    object_center_v: float | None = None
    translation_x_bin: int | None = None
    translation_y_bin: int | None = None
    translation_depth_bin: int | None = None


def _pose_samples(config: SyntheticObjectBuildConfig) -> list[_PoseSample]:
    if config.pose.mode == "so3":
        return _so3_pose_samples(config)
    if config.pose.mode == "sphere":
        return _sphere_pose_samples(config)
    if config.pose.mode in TRANSLATION_POSE_MODES:
        return _translation_pose_samples(config)
    azimuth_values = np.linspace(
        config.render.azimuth_start,
        config.render.azimuth_stop,
        num=config.render.azimuth_steps,
        dtype=np.float32,
    )
    return [
        _PoseSample(
            row_id=row_id,
            mode="azimuth",
            label=f"azimuth_bin_{_azimuth_bin(float(azimuth), config.render):02d}",
            label_id=_azimuth_bin(float(azimuth), config.render),
            azimuth_deg=float(azimuth),
        )
        for row_id, azimuth in enumerate(azimuth_values)
    ]


def _sphere_pose_samples(config: SyntheticObjectBuildConfig) -> list[_PoseSample]:
    assert config.pose.samples is not None
    rng = np.random.default_rng(config.seed)
    z_values = rng.uniform(-1.0, 1.0, size=config.pose.samples)
    phi_values = rng.uniform(0.0, 2.0 * math.pi, size=config.pose.samples)
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z_values * z_values))
    directions = np.column_stack(
        [
            radius_xy * np.sin(phi_values),
            -radius_xy * np.cos(phi_values),
            z_values,
        ]
    ).astype(np.float32)
    samples: list[_PoseSample] = []
    for row_id, (direction, phi, z_value) in enumerate(
        zip(directions, phi_values, z_values, strict=True)
    ):
        label_id = _sphere_z_bin(float(z_value), bins=config.pose.orientation_bins)
        azimuth_deg = math.degrees(float(phi)) % 360.0
        samples.append(
            _PoseSample(
                row_id=row_id,
                mode="sphere",
                label=f"sphere_z_bin_{label_id:02d}",
                label_id=label_id,
                azimuth_deg=azimuth_deg,
                camera_direction=(
                    float(direction[0]),
                    float(direction[1]),
                    float(direction[2]),
                ),
                camera_azimuth_deg=azimuth_deg,
                camera_elevation_deg=math.degrees(math.asin(float(z_value))),
            )
        )
    return samples


def _so3_pose_samples(config: SyntheticObjectBuildConfig) -> list[_PoseSample]:
    assert config.pose.samples is not None
    rotations = _sample_so3(config.pose.samples, seed=config.seed)
    samples: list[_PoseSample] = []
    for row_id, rotation in enumerate(rotations):
        angle_deg = _rotation_angle_deg(rotation)
        label_id = _haar_orientation_bin(angle_deg, bins=config.pose.orientation_bins)
        samples.append(
            _PoseSample(
                row_id=row_id,
                mode="so3",
                label=f"so3_haar_bin_{label_id:02d}",
                label_id=label_id,
                azimuth_deg=0.0,
                rotation_matrix=rotation,
                rotation_angle_deg=angle_deg,
                rotation_axis=_rotation_axis(rotation, angle_deg=angle_deg),
                quaternion_wxyz=_rotation_quaternion_wxyz(rotation),
            )
        )
    return samples


def _translation_pose_samples(config: SyntheticObjectBuildConfig) -> list[_PoseSample]:
    assert config.pose.samples is not None
    rng = np.random.default_rng(config.seed)
    samples: list[_PoseSample] = []
    mode = config.pose.mode
    x_values = np.zeros(config.pose.samples, dtype=np.float32)
    y_values = np.zeros(config.pose.samples, dtype=np.float32)
    depth_values = np.full(
        config.pose.samples,
        float(config.render.camera_distance),
        dtype=np.float32,
    )
    if mode in {"translation_xy", "translation_xyz"}:
        x_values = rng.uniform(
            config.pose.translation_x_range[0],
            config.pose.translation_x_range[1],
            size=config.pose.samples,
        ).astype(np.float32)
        y_values = rng.uniform(
            config.pose.translation_y_range[0],
            config.pose.translation_y_range[1],
            size=config.pose.samples,
        ).astype(np.float32)
    if mode == "translation_z":
        depth_values = np.linspace(
            config.pose.depth_range[0],
            config.pose.depth_range[1],
            num=config.pose.samples,
            dtype=np.float32,
        )
    elif mode == "translation_xyz":
        depth_values = rng.uniform(
            config.pose.depth_range[0],
            config.pose.depth_range[1],
            size=config.pose.samples,
        ).astype(np.float32)

    xy_bins = _grid_2d_dimensions(config.pose.translation_bins)
    xyz_bins = _grid_3d_dimensions(config.pose.translation_bins)
    for row_id, (x_value, y_value, depth_value) in enumerate(
        zip(x_values, y_values, depth_values, strict=True)
    ):
        translation_x = float(x_value)
        translation_y = float(y_value)
        camera_depth = float(depth_value)
        x_bin = _range_bin(
            translation_x,
            config.pose.translation_x_range,
            bins=xy_bins[0] if mode == "translation_xy" else xyz_bins[0],
        )
        y_bin = _range_bin(
            translation_y,
            config.pose.translation_y_range,
            bins=xy_bins[1] if mode == "translation_xy" else xyz_bins[1],
        )
        depth_bin = _range_bin(
            camera_depth,
            config.pose.depth_range,
            bins=config.pose.translation_bins if mode == "translation_z" else xyz_bins[2],
        )
        label_id = _translation_label_id(
            mode=mode,
            x_bin=x_bin,
            y_bin=y_bin,
            depth_bin=depth_bin,
            xy_bins=xy_bins,
            xyz_bins=xyz_bins,
        )
        u_value, v_value = _translation_center_projection(
            translation_x=translation_x,
            translation_y=translation_y,
            camera_depth=camera_depth,
            render=config.render,
        )
        samples.append(
            _PoseSample(
                row_id=row_id,
                mode=mode,
                label=f"{mode}_bin_{label_id:02d}",
                label_id=label_id,
                azimuth_deg=0.0,
                translation_x=translation_x,
                translation_y=translation_y,
                translation_z=camera_depth - float(config.render.camera_distance),
                camera_depth=camera_depth,
                camera_position=_translation_camera_position(
                    translation_x=translation_x,
                    translation_y=translation_y,
                    camera_depth=camera_depth,
                    render=config.render,
                ),
                object_center_u=u_value,
                object_center_v=v_value,
                translation_x_bin=x_bin,
                translation_y_bin=y_bin,
                translation_depth_bin=depth_bin,
            )
        )
    return samples


def _sample_so3(samples: int, *, seed: int) -> np.ndarray:
    try:
        from scipy.stats import special_ortho_group
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise ConfigError("SO(3) sampling requires scipy.") from exc
    rng = np.random.default_rng(seed)
    rotations = special_ortho_group.rvs(3, size=samples, random_state=rng)
    return np.asarray(rotations, dtype=np.float32).reshape(samples, 3, 3)


def _camera_frame_for_pose(
    pose: _PoseSample,
    render: SyntheticRenderConfig,
) -> _CameraFrame | None:
    if pose.camera_direction is not None:
        direction = np.asarray(pose.camera_direction, dtype=np.float32)
        return _camera_frame_from_position(direction * float(render.camera_distance))
    if pose.mode in TRANSLATION_POSE_MODES:
        return _translation_camera_frame(pose, render)
    return None


def _camera_frame(
    *,
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
) -> _CameraFrame:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    position = np.asarray(
        [
            distance * math.cos(elevation) * math.sin(azimuth),
            -distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.sin(elevation),
        ],
        dtype=np.float32,
    )
    return _camera_frame_from_position(position)


def _camera_frame_from_position(position: np.ndarray) -> _CameraFrame:
    position = np.asarray(position, dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    forward = _normalize(target - position)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = _normalize(np.cross(forward, world_up))
    if np.linalg.norm(right) < 1.0e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    up = _normalize(np.cross(right, forward))
    return _CameraFrame(position=position, right=right, up=up, forward=forward)


def _base_translation_camera_frame(render: SyntheticRenderConfig) -> _CameraFrame:
    return _camera_frame(
        azimuth_deg=0.0,
        elevation_deg=render.elevation,
        distance=render.camera_distance,
    )


def _translation_camera_frame(
    pose: _PoseSample,
    render: SyntheticRenderConfig,
) -> _CameraFrame:
    if pose.translation_x is None or pose.translation_y is None or pose.camera_depth is None:
        raise ConfigError("Translation pose is missing camera translation values.")
    base = _base_translation_camera_frame(render)
    position = _translation_camera_position_from_base(
        base=base,
        translation_x=float(pose.translation_x),
        translation_y=float(pose.translation_y),
        camera_depth=float(pose.camera_depth),
    )
    return _CameraFrame(
        position=position,
        right=base.right,
        up=base.up,
        forward=base.forward,
    )


def _translation_camera_position(
    *,
    translation_x: float,
    translation_y: float,
    camera_depth: float,
    render: SyntheticRenderConfig,
) -> tuple[float, float, float]:
    base = _base_translation_camera_frame(render)
    position = _translation_camera_position_from_base(
        base=base,
        translation_x=translation_x,
        translation_y=translation_y,
        camera_depth=camera_depth,
    )
    return (float(position[0]), float(position[1]), float(position[2]))


def _translation_camera_position_from_base(
    *,
    base: _CameraFrame,
    translation_x: float,
    translation_y: float,
    camera_depth: float,
) -> np.ndarray:
    return (
        -base.forward * float(camera_depth)
        + base.right * float(translation_x)
        + base.up * float(translation_y)
    ).astype(np.float32)


def _translation_center_projection(
    *,
    translation_x: float,
    translation_y: float,
    camera_depth: float,
    render: SyntheticRenderConfig,
) -> tuple[float, float]:
    center = (float(render.image_size) - 1.0) / 2.0
    u_value = center - float(render.focal_length) * float(translation_x) / camera_depth
    v_value = center + float(render.focal_length) * float(translation_y) / camera_depth
    return (float(u_value), float(v_value))


def _cube_faces(scale: float) -> tuple[_CubeFace, ...]:
    return _box_faces(
        center=(0.0, 0.0, 0.0),
        size=(float(scale), float(scale), float(scale)),
        colors=(
            (0.88, 0.22, 0.18),
            (0.20, 0.66, 0.30),
            (0.18, 0.35, 0.88),
            (0.95, 0.78, 0.18),
            (0.16, 0.72, 0.78),
            (0.76, 0.30, 0.78),
        ),
        prefix="",
    )


def _object_faces(spec: SyntheticObjectSpec) -> tuple[_CubeFace, ...]:
    if spec.kind == "marked_cube":
        return _cube_faces(spec.scale)
    if spec.kind == "abstract_statue":
        return _abstract_statue_faces(spec.scale)
    if spec.kind == "offset_monument":
        return _offset_monument_faces(spec.scale)
    raise ConfigError(f"Unsupported synthetic object kind: {spec.kind}.")


def _rotated_faces(
    faces: tuple[_CubeFace, ...],
    rotation_matrix: np.ndarray | None,
) -> tuple[_CubeFace, ...]:
    if rotation_matrix is None:
        return faces
    rotation = np.asarray(rotation_matrix, dtype=np.float32).reshape(3, 3)
    return tuple(
        _CubeFace(
            name=face.name,
            vertices=face.vertices @ rotation.T,
            normal=_normalize(face.normal @ rotation.T),
            base_color=face.base_color,
        )
        for face in faces
    )


def _abstract_statue_faces(scale: float) -> tuple[_CubeFace, ...]:
    parts = (
        ((0.00, 0.00, -0.72), (1.12, 0.88, 0.28), _material_ramp((0.50, 0.54, 0.58))),
        ((-0.10, 0.00, -0.34), (0.54, 0.42, 0.66), _material_ramp((0.20, 0.52, 0.48))),
        ((0.12, -0.06, 0.10), (0.46, 0.36, 0.56), _material_ramp((0.76, 0.38, 0.24))),
        ((-0.08, -0.08, 0.55), (0.32, 0.28, 0.32), _material_ramp((0.78, 0.62, 0.30))),
        ((0.48, -0.04, 0.18), (0.20, 0.22, 0.78), _material_ramp((0.28, 0.58, 0.30))),
        ((-0.46, 0.18, -0.06), (0.18, 0.28, 0.58), _material_ramp((0.50, 0.36, 0.68))),
        ((-0.08, -0.29, 0.56), (0.11, 0.12, 0.09), _material_ramp((0.12, 0.12, 0.12))),
    )
    return _parts_to_faces(parts, scale=scale)


def _offset_monument_faces(scale: float) -> tuple[_CubeFace, ...]:
    parts = (
        ((0.00, 0.00, -0.78), (1.18, 0.76, 0.24), _material_ramp((0.42, 0.45, 0.48))),
        ((-0.18, 0.04, -0.14), (0.38, 0.34, 1.36), _material_ramp((0.22, 0.42, 0.70))),
        ((-0.03, -0.13, 0.64), (0.58, 0.30, 0.22), _material_ramp((0.78, 0.68, 0.26))),
        ((0.34, 0.02, -0.24), (0.50, 0.22, 0.22), _material_ramp((0.72, 0.30, 0.22))),
        ((-0.53, -0.18, 0.08), (0.22, 0.24, 0.34), _material_ramp((0.28, 0.60, 0.44))),
        ((0.05, 0.36, 0.14), (0.22, 0.20, 0.74), _material_ramp((0.58, 0.34, 0.68))),
        ((0.27, -0.24, 0.28), (0.18, 0.16, 0.22), _material_ramp((0.14, 0.14, 0.16))),
    )
    return _parts_to_faces(parts, scale=scale)


def _parts_to_faces(
    parts: tuple[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[tuple[float, float, float], ...],
        ],
        ...,
    ],
    *,
    scale: float,
) -> tuple[_CubeFace, ...]:
    faces: list[_CubeFace] = []
    for index, (center, size, colors) in enumerate(parts):
        faces.extend(
            _box_faces(
                center=tuple(float(value) * scale for value in center),
                size=tuple(float(value) * scale for value in size),
                colors=colors,
                prefix=f"part_{index}",
            )
        )
    return tuple(faces)


def _box_faces(
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    colors: tuple[tuple[float, float, float], ...],
    prefix: str,
) -> tuple[_CubeFace, ...]:
    cx, cy, cz = center
    hx, hy, hz = (float(value) / 2.0 for value in size)
    vertices = np.asarray(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )
    return (
        _face(
            _face_name(prefix, "negative_y"),
            vertices[[0, 1, 5, 4]],
            (0.0, -1.0, 0.0),
            colors[0],
        ),
        _face(_face_name(prefix, "positive_x"), vertices[[1, 2, 6, 5]], (1.0, 0.0, 0.0), colors[1]),
        _face(_face_name(prefix, "positive_y"), vertices[[2, 3, 7, 6]], (0.0, 1.0, 0.0), colors[2]),
        _face(
            _face_name(prefix, "negative_x"),
            vertices[[3, 0, 4, 7]],
            (-1.0, 0.0, 0.0),
            colors[3],
        ),
        _face(_face_name(prefix, "positive_z"), vertices[[4, 5, 6, 7]], (0.0, 0.0, 1.0), colors[4]),
        _face(
            _face_name(prefix, "negative_z"),
            vertices[[3, 2, 1, 0]],
            (0.0, 0.0, -1.0),
            colors[5],
        ),
    )


def _face(
    name: str,
    vertices: np.ndarray,
    normal: tuple[float, float, float],
    base_color: tuple[float, float, float],
) -> _CubeFace:
    return _CubeFace(
        name=name,
        vertices=np.asarray(vertices, dtype=np.float32),
        normal=np.asarray(normal, dtype=np.float32),
        base_color=base_color,
    )


def _face_name(prefix: str, suffix: str) -> str:
    return suffix if not prefix else f"{prefix}:{suffix}"


def _material_ramp(
    base_color: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    base = np.asarray(base_color, dtype=np.float32)
    multipliers = (0.95, 1.04, 0.82, 1.12, 1.08, 0.72)
    return tuple(
        tuple(float(value) for value in np.clip(base * multiplier, 0.0, 1.0))
        for multiplier in multipliers
    )


def _is_face_visible(face: _CubeFace, camera_position: np.ndarray) -> bool:
    view_direction = camera_position - face.center
    return float(np.dot(face.normal, view_direction)) > 0.0


def _project_points(
    points: np.ndarray,
    *,
    camera: _CameraFrame,
    focal_px: float,
    image_size: int,
) -> tuple[np.ndarray | None, float]:
    relative = np.asarray(points, dtype=np.float32) - camera.position
    x = relative @ camera.right
    y = relative @ camera.up
    z = relative @ camera.forward
    if np.any(z <= 1.0e-4):
        return None, 0.0
    center = (image_size - 1) / 2.0
    projected = np.column_stack(
        [
            center + focal_px * x / z,
            center - focal_px * y / z,
        ],
    )
    return projected.astype(np.float32), float(np.mean(z))


def _lit_color(face: _CubeFace, render: SyntheticRenderConfig) -> tuple[int, int, int]:
    light_position = np.asarray(render.light_position, dtype=np.float32)
    light_direction = _normalize(light_position - face.center)
    lambert = max(0.0, float(np.dot(face.normal, light_direction)))
    energy_scale = max(0.0, render.light_energy / 400.0)
    shade = np.clip(render.ambient + render.diffuse * energy_scale * lambert, 0.0, 1.25)
    color = np.clip(np.asarray(face.base_color, dtype=np.float32) * shade, 0.0, 1.0)
    return tuple(int(round(float(channel) * 255.0)) for channel in color)


def _marker_polygon(face: _CubeFace, spec: SyntheticObjectSpec) -> np.ndarray:
    if face.name != "negative_y":
        return np.empty((0, 3), dtype=np.float32)
    normal = face.normal
    u_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    v_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    s = spec.scale / 2.0
    center = face.center + normal * 1.0e-3
    center = center + u_axis * spec.marker_offset[0] * s + v_axis * spec.marker_offset[1] * s
    half = spec.marker_size * s / 2.0
    return np.asarray(
        [
            center - u_axis * half - v_axis * half,
            center + u_axis * half - v_axis * half,
            center + u_axis * half + v_axis * half,
            center - u_axis * half + v_axis * half,
        ],
        dtype=np.float32,
    )


def _metadata_row(
    *,
    pose: _PoseSample,
    image_path: str,
    config: SyntheticObjectBuildConfig,
) -> dict[str, Any]:
    render = config.render
    object_spec = config.object_spec
    row = {
        "row_id": int(pose.row_id),
        "image_path": image_path,
        "dataset": config.family,
        "split": config.split,
        "label": pose.label,
        "label_id": int(pose.label_id),
        "family": config.family,
        "variant_id": config.variant_id,
        "variant": config.variant,
        "base_variant": config.base,
        "prompt_id": object_spec.kind,
        "prompt": _prompt(object_spec.kind, pose),
        "tags": ["synthetic", object_spec.kind, config.pose.mode],
        "source_index": int(pose.row_id),
        "sample_type": "synthetic_render",
        "pose_mode": pose.mode,
        "object_kind": object_spec.kind,
        "object_label": object_spec.kind,
        "azimuth_deg": float(pose.azimuth_deg),
        "azimuth_bin": int(_azimuth_bin(float(pose.azimuth_deg), render)),
        "rotation_angle_deg": _optional_float(pose.rotation_angle_deg),
        "rotation_angle_bin": (
            None
            if pose.rotation_angle_deg is None
            else int(_orientation_bin(pose.rotation_angle_deg, bins=config.pose.orientation_bins))
        ),
        "so3_haar_bin": (
            None
            if pose.rotation_angle_deg is None
            else int(
                _haar_orientation_bin(
                    pose.rotation_angle_deg,
                    bins=config.pose.orientation_bins,
                )
            )
        ),
        "elevation_deg": float(render.elevation),
        "camera_distance": float(render.camera_distance),
        "focal_length": float(render.focal_length),
        "light_x": float(render.light_position[0]),
        "light_y": float(render.light_position[1]),
        "light_z": float(render.light_position[2]),
        "light_energy": float(render.light_energy),
        "ambient": float(render.ambient),
        "diffuse": float(render.diffuse),
        "background_r": float(render.background[0]),
        "background_g": float(render.background[1]),
        "background_b": float(render.background[2]),
    }
    if pose.camera_direction is not None:
        row["camera_direction_x"] = float(pose.camera_direction[0])
        row["camera_direction_y"] = float(pose.camera_direction[1])
        row["camera_direction_z"] = float(pose.camera_direction[2])
        row["camera_azimuth_deg"] = _optional_float(pose.camera_azimuth_deg)
        row["camera_elevation_deg"] = _optional_float(pose.camera_elevation_deg)
        row["sphere_z_bin"] = int(pose.label_id)
    if pose.camera_position is not None:
        row["camera_position_x"] = float(pose.camera_position[0])
        row["camera_position_y"] = float(pose.camera_position[1])
        row["camera_position_z"] = float(pose.camera_position[2])
    if pose.translation_x is not None:
        row["translation_x"] = float(pose.translation_x)
        row["translation_y"] = float(pose.translation_y or 0.0)
        row["translation_z"] = float(pose.translation_z or 0.0)
        row["translation_xy_radius"] = float(
            math.hypot(float(pose.translation_x), float(pose.translation_y or 0.0))
        )
        row["translation_x_bin"] = int(pose.translation_x_bin or 0)
        row["translation_y_bin"] = int(pose.translation_y_bin or 0)
        row["translation_depth_bin"] = int(pose.translation_depth_bin or 0)
        row["camera_depth"] = float(pose.camera_depth or render.camera_distance)
        row["object_center_u"] = _optional_float(pose.object_center_u)
        row["object_center_v"] = _optional_float(pose.object_center_v)
        if pose.object_center_u is not None:
            row["object_center_u_norm"] = float(
                pose.object_center_u / max(1, render.image_size - 1)
            )
        if pose.object_center_v is not None:
            row["object_center_v_norm"] = float(
                pose.object_center_v / max(1, render.image_size - 1)
            )
    if pose.rotation_matrix is not None:
        rotation = np.asarray(pose.rotation_matrix, dtype=np.float32).reshape(3, 3)
        for row_index in range(3):
            for column_index in range(3):
                row[f"rotation_r{row_index}{column_index}"] = float(
                    rotation[row_index, column_index]
                )
    if pose.rotation_axis is not None:
        row["rotation_axis_x"] = float(pose.rotation_axis[0])
        row["rotation_axis_y"] = float(pose.rotation_axis[1])
        row["rotation_axis_z"] = float(pose.rotation_axis[2])
    if pose.quaternion_wxyz is not None:
        row["quat_w"] = float(pose.quaternion_wxyz[0])
        row["quat_x"] = float(pose.quaternion_wxyz[1])
        row["quat_y"] = float(pose.quaternion_wxyz[2])
        row["quat_z"] = float(pose.quaternion_wxyz[3])
    return row


def _prompt(object_kind: str, pose: _PoseSample) -> str:
    if pose.mode == "so3":
        return f"{object_kind} under Haar SO(3) orientation sample {pose.row_id}"
    if pose.mode == "sphere":
        return f"{object_kind} from uniform sphere camera sample {pose.row_id}"
    if pose.mode in TRANSLATION_POSE_MODES:
        return f"{object_kind} under camera translation sample {pose.row_id}"
    return f"{object_kind} at azimuth {pose.azimuth_deg:.3f} degrees"


def _image_path(image_dir: Path, pose: _PoseSample) -> Path:
    if pose.mode == "so3":
        return image_dir / f"{pose.row_id:05d}_so3.png"
    if pose.mode == "sphere":
        return image_dir / f"{pose.row_id:05d}_sphere.png"
    if pose.mode in TRANSLATION_POSE_MODES:
        return image_dir / f"{pose.row_id:05d}_{pose.mode}.png"
    return image_dir / f"{pose.row_id:05d}_azimuth_{float(pose.azimuth_deg):07.3f}.png"


def _azimuth_bin(azimuth_deg: float, render: SyntheticRenderConfig) -> int:
    if render.azimuth_steps == 1 or render.azimuth_stop == render.azimuth_start:
        return 0
    low = min(render.azimuth_start, render.azimuth_stop)
    high = max(render.azimuth_start, render.azimuth_stop)
    fraction = (float(azimuth_deg) - low) / max(high - low, 1.0e-8)
    clipped = min(1.0, max(0.0, fraction))
    return min(render.azimuth_bins - 1, int(math.floor(clipped * render.azimuth_bins)))


def _orientation_bin(angle_deg: float, *, bins: int) -> int:
    fraction = float(angle_deg) / 180.0
    clipped = min(1.0, max(0.0, fraction))
    return min(bins - 1, int(math.floor(clipped * bins)))


def _sphere_z_bin(z_value: float, *, bins: int) -> int:
    fraction = (float(z_value) + 1.0) / 2.0
    clipped = min(1.0, max(0.0, fraction))
    return min(bins - 1, int(math.floor(clipped * bins)))


def _range_bin(value: float, value_range: tuple[float, float], *, bins: int) -> int:
    low, high = value_range
    fraction = (float(value) - low) / max(high - low, 1.0e-8)
    clipped = min(1.0, max(0.0, fraction))
    return min(bins - 1, int(math.floor(clipped * bins)))


def _grid_2d_dimensions(total_bins: int) -> tuple[int, int]:
    x_bins = max(1, int(round(math.sqrt(total_bins))))
    while total_bins % x_bins != 0 and x_bins > 1:
        x_bins -= 1
    return (x_bins, max(1, total_bins // x_bins))


def _grid_3d_dimensions(total_bins: int) -> tuple[int, int, int]:
    best = (1, 1, max(1, total_bins))
    best_score = float("inf")
    for x_bins in range(1, total_bins + 1):
        if total_bins % x_bins != 0:
            continue
        remaining = total_bins // x_bins
        for y_bins in range(1, remaining + 1):
            if remaining % y_bins != 0:
                continue
            z_bins = remaining // y_bins
            dims = tuple(sorted((x_bins, y_bins, z_bins)))
            score = float(dims[-1] - dims[0])
            if score < best_score:
                best = dims
                best_score = score
    return best


def _translation_label_id(
    *,
    mode: str,
    x_bin: int,
    y_bin: int,
    depth_bin: int,
    xy_bins: tuple[int, int],
    xyz_bins: tuple[int, int, int],
) -> int:
    if mode == "translation_xy":
        return int(y_bin * xy_bins[0] + x_bin)
    if mode == "translation_z":
        return int(depth_bin)
    if mode == "translation_xyz":
        x_bins, y_bins, _ = xyz_bins
        return int((depth_bin * y_bins + y_bin) * x_bins + x_bin)
    raise ConfigError(f"Unsupported translation pose mode: {mode}.")


def _haar_orientation_bin(angle_deg: float, *, bins: int) -> int:
    angle = math.radians(float(angle_deg))
    # SO(3) Haar measure induces rotation-angle CDF F(theta) = (theta - sin(theta)) / pi.
    fraction = (angle - math.sin(angle)) / math.pi
    clipped = min(1.0, max(0.0, fraction))
    return min(bins - 1, int(math.floor(clipped * bins)))


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    cosine = min(1.0, max(-1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _rotation_axis(
    rotation: np.ndarray,
    *,
    angle_deg: float,
) -> tuple[float, float, float]:
    angle = math.radians(angle_deg)
    if abs(math.sin(angle)) < 1.0e-6:
        return (0.0, 0.0, 0.0)
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float32,
    )
    axis = axis / (2.0 * math.sin(angle))
    axis = _normalize(axis)
    return (float(axis[0]), float(axis[1]), float(axis[2]))


def _rotation_quaternion_wxyz(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray([w, x, y, z], dtype=np.float32)
    quaternion = _normalize(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _optional_float(value: float | None) -> float | None:
    return None if value is None else float(value)


def _save_float_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _label_counts(metadata: pd.DataFrame) -> dict[str, int]:
    counts = metadata["label"].astype(str).value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _config_raw(config: SyntheticObjectBuildConfig) -> dict[str, Any]:
    if config.raw:
        return dict(config.raw)
    return {
        "family": config.family,
        "variant": config.variant,
        "base": config.base,
        "split": config.split,
        "seed": config.seed,
        "object": {
            "kind": config.object_spec.kind,
            "scale": config.object_spec.scale,
            "marker": config.object_spec.marker,
            "marker_face": config.object_spec.marker_face,
            "marker_size": config.object_spec.marker_size,
            "marker_offset": list(config.object_spec.marker_offset),
        },
        "render": {
            "backend": config.render.backend,
            "image_size": config.render.image_size,
            "azimuth_start": config.render.azimuth_start,
            "azimuth_stop": config.render.azimuth_stop,
            "azimuth_steps": config.render.azimuth_steps,
            "elevation": config.render.elevation,
            "camera_distance": config.render.camera_distance,
            "focal_length": config.render.focal_length,
            "background": list(config.render.background),
            "light_position": list(config.render.light_position),
            "light_energy": config.render.light_energy,
            "ambient": config.render.ambient,
            "diffuse": config.render.diffuse,
            "supersample": config.render.supersample,
            "azimuth_bins": config.render.azimuth_bins,
        },
        "pose": {
            "mode": config.pose.mode,
            "samples": config.pose.samples,
            "orientation_bins": config.pose.orientation_bins,
            "translation_bins": config.pose.translation_bins,
            "translation_x_range": list(config.pose.translation_x_range),
            "translation_y_range": list(config.pose.translation_y_range),
            "depth_range": list(config.pose.depth_range),
        },
        "output": {
            "value_range": list(config.output.value_range),
            "save_pngs": config.output.save_pngs,
        },
    }


def _rgb255(values: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(int(round(float(channel) * 255.0)) for channel in values)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return np.asarray(vector, dtype=np.float32) / norm


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Config section {name!r} must be a mapping.")
    return dict(value)


def _float_triplet(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ConfigError(f"{name} must contain exactly three numeric values.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _float_pair(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigError(f"{name} must contain exactly two numeric values.")
    return (float(value[0]), float(value[1]))


def _validate_increasing_range(value: tuple[float, float], *, name: str) -> None:
    if value[1] <= value[0]:
        raise ConfigError(f"{name} must be increasing.")
