from __future__ import annotations

import numpy as np

from fm_lab.geometry_explorer.latent_factors import (
    AmbientLightInterval,
    AzimuthCircle,
    BoundedLookAtView,
    BoundedTranslation,
    CameraDepthTranslationInterval,
    CameraIntrinsicsFactor,
    CameraLocalTranslation,
    CameraLogAspectRatioInterval,
    CameraLogFocalScaleInterval,
    CameraPrincipalPointOffset,
    CameraRadiusInterval,
    CameraRollInterval,
    CameraSE3Factor,
    CameraSkewInterval,
    DiffuseLightInterval,
    FullAppearanceFactor,
    FullCameraFactor,
    IlluminationFactor,
    ImageLogExposureInterval,
    LatentFactorSpace,
    LightingDirectionSphere,
    LightLogEnergyInterval,
    LookAtViewSphere,
    PhotometryFactor,
    ProductFactorSpace,
    ZoomInterval,
    sample_values,
)
from fm_lab.geometry_explorer.latent_pixel_diagnostics import (
    analyze_latent_pixel_diagnostics,
)
from fm_lab.geometry_explorer.product_structure import (
    analyze_product_structure,
    analyze_product_structure_at_point,
    factor_coupling_score,
    product_metric_error,
)
from fm_lab.geometry_explorer.pullback_metric import (
    analyze_pullback_metrics,
    estimate_pullback_metric,
)
from fm_lab.geometry_explorer.render_maps import RenderConfig, RenderMap


class _TranslationIdentityMap:
    object_name = "analytic"
    render_mode = "analytic"

    def render_flat(self, z: np.ndarray) -> np.ndarray:
        value = np.asarray(z, dtype=np.float32)
        return np.asarray([value[0], value[1], 0.0, 0.0], dtype=np.float32)


class _IndependentProductMap:
    object_name = "analytic"
    render_mode = "analytic"

    def render_flat(self, z: dict[str, np.ndarray]) -> np.ndarray:
        return np.asarray([float(z["a"][0]), float(z["b"][0])], dtype=np.float32)


class _EntangledProductMap:
    object_name = "analytic"
    render_mode = "analytic"

    def render_flat(self, z: dict[str, np.ndarray]) -> np.ndarray:
        a_value = float(z["a"][0])
        b_value = float(z["b"][0])
        return np.asarray([a_value, b_value, a_value + b_value], dtype=np.float32)


class _ConstantBatchMap:
    def render_batch(self, zs: list[np.ndarray], batch_size: int = 128) -> np.ndarray:
        del batch_size
        return np.zeros((len(zs), 4), dtype=np.float32)


def test_primitive_factor_spaces_sample_retract_and_distance() -> None:
    factors: list[LatentFactorSpace] = [
        AzimuthCircle(),
        LookAtViewSphere(),
        LightingDirectionSphere(),
        BoundedTranslation(dim=2, bounds=(-0.5, 0.5)),
        CameraDepthTranslationInterval(bounds=(-0.5, 0.5)),
        CameraRadiusInterval(bounds=(2.0, 4.0)),
        ZoomInterval(bounds=(40.0, 80.0)),
        CameraRollInterval(bounds=(-0.25, 0.25)),
        CameraLocalTranslation(bounds=(-0.5, 0.5)),
        CameraLogFocalScaleInterval(bounds=(-0.2, 0.2)),
        CameraLogAspectRatioInterval(bounds=(-0.2, 0.2)),
        CameraPrincipalPointOffset(bounds=(-2.0, 2.0)),
        CameraSkewInterval(bounds=(-0.1, 0.1)),
        LightLogEnergyInterval(bounds=(-0.2, 0.2)),
        AmbientLightInterval(bounds=(0.1, 0.5)),
        DiffuseLightInterval(bounds=(0.2, 0.8)),
        ImageLogExposureInterval(bounds=(-0.2, 0.2)),
        CameraSE3Factor(),
        CameraIntrinsicsFactor(),
        FullCameraFactor(),
        IlluminationFactor(),
        PhotometryFactor(),
        FullAppearanceFactor(),
    ]

    for factor in factors:
        sample = factor.sample(5, seed=3)
        values = sample_values(sample)
        assert len(values) == 5
        assert len(factor.tangent_basis(values[0])) == factor.dim
        assert len(factor.tangent_labels(values[0])) == factor.dim
        moved = factor.retract(values[0], factor.tangent_basis(values[0])[0], 1.0e-3)
        distance = factor.distance(values[0], moved)
        assert np.isfinite(distance)
        assert distance >= 0.0
        assert np.isclose(
            factor.distance(values[0], values[1]),
            factor.distance(values[1], values[0]),
        )


def test_bounded_look_at_view_samples_area_uniform_band_and_retracts() -> None:
    elevation_bounds = (-np.pi / 6, np.pi / 3)
    factor = BoundedLookAtView(elevation_bounds=elevation_bounds)
    values = np.asarray(sample_values(factor.sample(20_000, seed=7)))
    sin_bounds = np.sin(elevation_bounds)
    expected_quartiles = sin_bounds[0] + (sin_bounds[1] - sin_bounds[0]) * np.asarray(
        [0.25, 0.75]
    )

    assert values.shape == (20_000, 2)
    assert np.all(values[:, 0] >= -np.pi)
    assert np.all(values[:, 0] < np.pi)
    assert np.all(values[:, 1] >= sin_bounds[0])
    assert np.all(values[:, 1] <= sin_bounds[1])
    assert abs(float(values[:, 1].mean()) - float(np.mean(sin_bounds))) < 0.01
    assert np.allclose(
        np.quantile(values[:, 1], [0.25, 0.75]),
        expected_quartiles,
        atol=0.015,
    )
    assert factor.dim == 2
    assert factor.tangent_labels(values[0]) == [
        "camera_azimuth",
        "camera_elevation",
    ]

    retracted = factor.retract(
        np.asarray([np.pi - 0.1, 0.45], dtype=np.float32),
        np.asarray([1.0, 3.0], dtype=np.float32),
        eps=0.2,
    )
    assert np.isclose(retracted[0], -np.pi + 0.1)
    assert np.isclose(retracted[1], sin_bounds[1])


def test_render_map_applies_bounded_look_at_view() -> None:
    factor = BoundedLookAtView(elevation_bounds=(-np.pi / 6, np.pi / 6))
    render_map = RenderMap(
        factor,
        object_name="offset_monument",
        config=RenderConfig(image_size=32, render_mode="silhouette"),
    )
    front = render_map.render(np.asarray([0.0, 0.0], dtype=np.float32))
    side = render_map.render(np.asarray([np.pi / 2, 0.0], dtype=np.float32))
    high = render_map.render(np.asarray([0.0, 0.5], dtype=np.float32))

    assert np.mean(np.abs(front - side)) > 0.01
    assert np.mean(np.abs(front - high)) > 0.01


def test_product_factor_space_preserves_block_structure_metadata() -> None:
    product = ProductFactorSpace(
        [
            LookAtViewSphere(),
            BoundedTranslation(dim=2, bounds=(-0.2, 0.2)),
        ]
    )
    sample = product.sample(4, seed=7)
    values = sample_values(sample)

    assert product.dim == 4
    assert product.factor_dims == [2, 2]
    assert product.factor_slices["view_sphere"] == slice(0, 2)
    assert product.factor_slices["translation_xy"] == slice(2, 4)
    assert set(values[0]) == {"view_sphere", "translation_xy"}
    assert len(product.tangent_basis(values[0])) == 4
    assert product.tangent_labels(values[0]) == [
        "view_sphere_tangent_1",
        "view_sphere_tangent_2",
        "translation_x",
        "translation_y",
    ]


def test_pullback_metric_matches_analytic_identity_map() -> None:
    factor = BoundedTranslation(dim=2, bounds=(-10.0, 10.0), clip_retraction=False)
    z_value = np.asarray([0.25, -0.5], dtype=np.float32)

    result = estimate_pullback_metric(
        _TranslationIdentityMap(),
        factor,
        z_value,
        eps=1.0e-4,
        normalize_by_num_pixels=False,
    )

    assert np.allclose(result.G, np.eye(2), atol=1.0e-3)
    assert result.estimated_rank == 2
    assert np.allclose(result.eigenvalues, [1.0, 1.0], atol=1.0e-3)
    assert np.allclose(result.tangent_correlation, np.eye(2), atol=1.0e-3)
    assert result.tangent_labels == ["translation_x", "translation_y"]
    assert result.max_abs_offdiag_tangent_correlation < 1.0e-3


def test_product_coupling_is_zero_for_independent_map_and_nonzero_when_entangled(
) -> None:
    product = ProductFactorSpace(
        [
            BoundedTranslation(dim=1, bounds=(-1.0, 1.0), name="a"),
            BoundedTranslation(dim=1, bounds=(-1.0, 1.0), name="b"),
        ]
    )
    z_value = {
        "a": np.asarray([0.2], dtype=np.float32),
        "b": np.asarray([-0.1], dtype=np.float32),
    }

    independent = analyze_product_structure_at_point(
        _IndependentProductMap(),
        product,
        z_value,
        eps=1.0e-4,
    )
    entangled = analyze_product_structure_at_point(
        _EntangledProductMap(),
        product,
        z_value,
        eps=1.0e-4,
    )

    assert product_metric_error(independent.G, product) < 1.0e-6
    assert factor_coupling_score(
        independent.G[0:1, 0:1],
        independent.G[1:2, 1:2],
        independent.G[0:1, 1:2],
    ) < 1.0e-6
    assert independent.cross_factor_tangent_correlations["a__b"] < 1.0e-6
    assert entangled.pairwise_couplings["a__b"] > 0.1
    assert entangled.cross_factor_tangent_correlations["a__b"] > 0.1
    assert entangled.product_error > 0.1


def test_labeled_pullback_summaries_report_entangled_tangent_pairs() -> None:
    product = ProductFactorSpace(
        [
            BoundedTranslation(dim=1, bounds=(-1.0, 1.0), name="a"),
            BoundedTranslation(dim=1, bounds=(-1.0, 1.0), name="b"),
        ]
    )
    z_value = {
        "a": np.asarray([0.2], dtype=np.float32),
        "b": np.asarray([-0.1], dtype=np.float32),
    }

    pullback = analyze_pullback_metrics(
        _EntangledProductMap(),
        product,
        [z_value],
        eps=1.0e-4,
        normalize_by_num_pixels=False,
    )
    product_summary = analyze_product_structure(
        _EntangledProductMap(),
        product,
        [z_value],
        eps=1.0e-4,
    )

    assert pullback.tangent_labels == ["a", "b"]
    assert pullback.top_tangent_collapse_pairs[0]["left"] == "a"
    assert pullback.top_tangent_collapse_pairs[0]["right"] == "b"
    assert pullback.top_tangent_collapse_pairs[0]["correlation"] > 0.45
    assert pullback.tangent_energy[0]["label"] == "a"
    assert product_summary.tangent_labels == ["a", "b"]
    assert product_summary.top_tangent_collapse_pairs[0]["left"] == "a"
    assert product_summary.top_tangent_collapse_pairs[0]["right"] == "b"


def test_latent_pixel_diagnostics_allow_constant_rendered_images() -> None:
    factor = BoundedTranslation(
        dim=1,
        bounds=(-1.0, 1.0),
        name="inactive_factor",
    )
    values = sample_values(factor.sample(16, seed=9))

    summary = analyze_latent_pixel_diagnostics(
        _ConstantBatchMap(),
        factor,
        values,
        max_samples=16,
        pair_count=100,
        ks=(3,),
        seed=9,
    )

    assert np.isnan(summary.spearman_distance_corr)
    assert np.isnan(summary.pearson_distance_corr)
    assert np.isfinite(summary.knn_overlap_by_k[3])


def test_render_map_modes_and_pullback_metric_are_finite() -> None:
    factor = AzimuthCircle()
    z_value = sample_values(factor.sample(1, seed=11))[0]
    colored = RenderMap(
        factor,
        object_name="abstract_statue",
        config=RenderConfig(image_size=24, render_mode="colored", antialias=False),
    )
    silhouette = RenderMap(
        factor,
        object_name="abstract_statue",
        config=RenderConfig(image_size=24, render_mode="silhouette", antialias=False),
    )

    colored_image = colored.render(z_value)
    silhouette_image = silhouette.render(z_value)
    metric = estimate_pullback_metric(colored, factor, z_value, eps=2.0e-3)

    assert colored_image.shape == (24, 24, 3)
    assert silhouette_image.shape == (24, 24, 3)
    assert not np.allclose(colored_image, silhouette_image)
    assert metric.G.shape == (1, 1)
    assert np.isfinite(metric.trace)
    assert metric.trace >= 0.0


def test_view_sphere_translation_xyz_is_relative_to_camera_frame() -> None:
    factor = ProductFactorSpace(
        [
            LookAtViewSphere(),
            BoundedTranslation(dim=3, bounds=((-0.5, 0.5), (-0.5, 0.5), (-1.0, 1.0))),
        ]
    )
    render_map = RenderMap(
        factor,
        object_name="abstract_statue",
        config=RenderConfig(
            image_size=24,
            render_mode="silhouette",
            antialias=False,
            camera_config={"radius": 4.0, "translation_target": "camera_plane"},
        ),
    )
    translation = np.asarray([0.2, -0.1, 0.3], dtype=np.float32)
    first = {
        "view_sphere": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
        "translation_xyz": translation,
    }
    second = {
        "view_sphere": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "translation_xyz": translation,
    }

    first_frame = render_map._camera_frame(render_map._controls(first))
    second_frame = render_map._camera_frame(render_map._controls(second))
    first_base = render_map._camera_frame(
        render_map._controls({**first, "translation_xyz": np.zeros(3, dtype=np.float32)})
    )
    second_base = render_map._camera_frame(
        render_map._controls({**second, "translation_xyz": np.zeros(3, dtype=np.float32)})
    )
    first_delta = first_frame.position - first_base.position
    second_delta = second_frame.position - second_base.position

    assert not np.allclose(first_delta, second_delta)
    assert np.isclose(float(np.dot(first_delta, first_base.right)), translation[0])
    assert np.isclose(float(np.dot(first_delta, first_base.up)), translation[1])
    assert np.isclose(float(np.dot(first_delta, -first_base.forward)), translation[2])
    assert np.isclose(float(np.dot(second_delta, second_base.right)), translation[0])
    assert np.isclose(float(np.dot(second_delta, second_base.up)), translation[1])
    assert np.isclose(float(np.dot(second_delta, -second_base.forward)), translation[2])


def test_azimuth_depth_product_changes_radius_in_azimuth_frame() -> None:
    factor = ProductFactorSpace(
        [
            AzimuthCircle(),
            CameraDepthTranslationInterval(bounds=(-1.0, 1.0)),
        ],
        name="azimuth_depth_product",
    )
    render_map = RenderMap(
        factor,
        object_name="abstract_statue",
        config=RenderConfig(
            image_size=24,
            render_mode="silhouette",
            antialias=False,
            camera_config={"radius": 4.0},
        ),
    )
    z_value = {
        "azimuth": np.float32(np.pi / 3.0),
        "camera_depth_translation": np.float32(0.75),
    }

    controls = render_map._controls(z_value)
    frame = render_map._camera_frame(controls)

    assert np.isclose(controls.camera_radius, 4.75)
    assert np.isclose(np.linalg.norm(frame.position), 4.75, atol=1.0e-5)
    assert factor.dim == 2


def test_full_camera_factor_is_flat_eleven_dimensional_product() -> None:
    factor = FullCameraFactor()
    sample = factor.sample(3, seed=17)
    values = sample_values(sample)

    assert factor.dim == 11
    assert factor.factor_names == [
        "view_sphere",
        "camera_roll",
        "camera_translation_xyz",
        "camera_log_focal_scale",
        "camera_log_aspect_ratio",
        "camera_principal_point",
        "camera_skew",
    ]
    assert factor.factor_dims == [2, 1, 3, 1, 1, 2, 1]
    assert len(factor.tangent_basis(values[0])) == 11
    assert factor.tangent_labels(values[0]) == [
        "view_sphere_tangent_1",
        "view_sphere_tangent_2",
        "camera_roll",
        "camera_translation_x",
        "camera_translation_y",
        "camera_translation_z",
        "camera_log_focal_scale",
        "camera_log_aspect_ratio",
        "camera_principal_point_x",
        "camera_principal_point_y",
        "camera_skew",
    ]


def test_camera_roll_rotates_local_translation_axes_without_moving_base_position() -> None:
    factor = ProductFactorSpace(
        [
            LookAtViewSphere(),
            CameraRollInterval(bounds=(-np.pi, np.pi)),
            CameraLocalTranslation(bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))),
        ]
    )
    render_map = RenderMap(
        factor,
        object_name="abstract_statue",
        config=RenderConfig(
            image_size=24,
            render_mode="silhouette",
            antialias=False,
            camera_config={"radius": 4.0},
        ),
    )
    direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
    no_roll = {
        "view_sphere": direction,
        "camera_roll": np.float32(0.0),
        "camera_translation_xyz": np.zeros(3, dtype=np.float32),
    }
    rolled = {
        **no_roll,
        "camera_roll": np.float32(np.pi / 2.0),
    }
    translated = {
        **rolled,
        "camera_translation_xyz": np.asarray([0.25, 0.0, 0.0], dtype=np.float32),
    }

    no_roll_frame = render_map._camera_frame(render_map._controls(no_roll))
    rolled_frame = render_map._camera_frame(render_map._controls(rolled))
    translated_frame = render_map._camera_frame(render_map._controls(translated))

    assert np.allclose(rolled_frame.position, no_roll_frame.position)
    assert np.allclose(rolled_frame.forward, no_roll_frame.forward)
    assert np.isclose(float(np.dot(rolled_frame.right, no_roll_frame.up)), 1.0)
    assert np.isclose(float(np.dot(rolled_frame.up, -no_roll_frame.right)), 1.0)
    assert np.isclose(
        float(np.dot(translated_frame.position - rolled_frame.position, rolled_frame.right)),
        0.25,
    )


def test_camera_intrinsics_factor_updates_projection_controls() -> None:
    factor = CameraIntrinsicsFactor(
        focal_log_bounds=(-1.0, 1.0),
        aspect_log_bounds=(-1.0, 1.0),
        principal_point_bounds=((-4.0, 4.0), (-4.0, 4.0)),
        skew_bounds=(-0.2, 0.2),
    )
    render_map = RenderMap(
        factor,
        object_name="abstract_statue",
        config=RenderConfig(
            image_size=24,
            render_mode="silhouette",
            antialias=False,
            camera_config={"focal_length": 70.0},
        ),
    )
    z_value = {
        "camera_log_focal_scale": np.float32(np.log(1.2)),
        "camera_log_aspect_ratio": np.float32(np.log(1.5)),
        "camera_principal_point": np.asarray([2.0, -3.0], dtype=np.float32),
        "camera_skew": np.float32(0.1),
    }
    controls = render_map._controls(z_value)
    image = render_map.render(z_value)

    assert np.isclose(controls.focal_length, 84.0)
    assert np.isclose(controls.aspect_ratio, 1.5)
    assert np.allclose(controls.principal_point_offset, [2.0, -3.0])
    assert np.isclose(controls.skew_ratio, 0.1)
    assert image.shape == (24, 24, 3)
    assert np.isfinite(image).all()


def test_illumination_and_photometry_factors_update_render_controls() -> None:
    factor = ProductFactorSpace(
        [
            LightingDirectionSphere(),
            LightLogEnergyInterval(bounds=(-1.0, 1.0)),
            AmbientLightInterval(bounds=(0.05, 0.75)),
            DiffuseLightInterval(bounds=(0.05, 1.10)),
            ImageLogExposureInterval(bounds=(-0.6, 0.4)),
        ],
        name="test_full_appearance",
    )
    render_map = RenderMap(
        factor,
        object_name="abstract_statue",
        config=RenderConfig(
            image_size=24,
            render_mode="colored",
            background="gray",
            antialias=False,
            light_config={
                "distance": 6.0,
                "energy": 400.0,
                "ambient": 0.35,
                "diffuse": 0.70,
            },
        ),
    )
    z_value = {
        "light_sphere": np.asarray([1.0, -1.0, 1.0], dtype=np.float32),
        "light_log_energy": np.float32(np.log(1.5)),
        "ambient_light": np.float32(0.55),
        "diffuse_light": np.float32(0.90),
        "image_log_exposure": np.float32(np.log(0.8)),
    }

    controls = render_map._controls(z_value)
    image = render_map.render(z_value)
    baseline = render_map.render(
        {
            **z_value,
            "light_log_energy": np.float32(0.0),
            "ambient_light": np.float32(0.35),
            "diffuse_light": np.float32(0.70),
            "image_log_exposure": np.float32(0.0),
        }
    )

    assert np.isclose(np.linalg.norm(controls.light_position), 6.0, atol=1.0e-5)
    assert np.isclose(controls.light_energy, 600.0)
    assert np.isclose(controls.ambient, 0.55)
    assert np.isclose(controls.diffuse, 0.90)
    assert np.isclose(controls.image_exposure, 0.8)
    assert image.shape == (24, 24, 3)
    assert np.isfinite(image).all()
    assert not np.allclose(image, baseline)


def test_named_appearance_product_factors_have_expected_axes() -> None:
    illumination = IlluminationFactor()
    photometry = PhotometryFactor()
    full = FullAppearanceFactor()

    assert illumination.dim == 5
    assert illumination.tangent_labels(None) == [
        "light_sphere_tangent_1",
        "light_sphere_tangent_2",
        "light_log_energy",
        "ambient_light",
        "diffuse_light",
    ]
    assert photometry.dim == 4
    assert photometry.tangent_labels(None) == [
        "light_log_energy",
        "ambient_light",
        "diffuse_light",
        "image_log_exposure",
    ]
    assert full.dim == 6
    assert full.tangent_labels(None) == [
        "light_sphere_tangent_1",
        "light_sphere_tangent_2",
        "light_log_energy",
        "ambient_light",
        "diffuse_light",
        "image_log_exposure",
    ]
