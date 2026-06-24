import torch

from fm_lab.data import (
    GaussianMixture3D,
    HelixMixture,
    LineSegment3D,
    MoebiusStrip,
    MultiSwissRoll,
    MultiTorus,
    NestedSphericalShells,
    PlanarDisk,
    SphericalShell,
    SwissRoll,
    Torus,
    TrefoilKnot,
)
from fm_lab.diagnostics import radial_deviation, radial_tangent_velocity_2d
from fm_lab.sources import SphericalShellSource


def test_shell_and_swiss_roll_shapes() -> None:
    assert SphericalShell(dim=5).sample(16).shape == (16, 5)
    assert SwissRoll().sample(16).shape == (16, 3)
    assert SphericalShellSource(dim=4).sample(16).shape == (16, 4)


def test_harder_3d_toy_shapes() -> None:
    distributions = [
        GaussianMixture3D(n_modes=6),
        MultiSwissRoll(n_rolls=2),
        Torus(),
        MultiTorus(n_tori=2),
        HelixMixture(n_helixes=3),
        MoebiusStrip(),
        LineSegment3D(),
        PlanarDisk(),
        TrefoilKnot(),
        NestedSphericalShells(radii=(0.5, 1.0), noise=0.0),
    ]

    for distribution in distributions:
        samples = distribution.sample(32)

        assert samples.shape == (32, 3)
        assert torch.isfinite(samples).all()
        assert distribution.metadata()["dim"] == 3


def test_exact_manifold_targets_respect_intrinsic_geometry() -> None:
    sphere = SphericalShell(dim=3, radius=1.3, noise=0.0).sample(256)
    assert torch.allclose(sphere.norm(dim=1), torch.full((256,), 1.3), atol=1e-5)

    line_distribution = LineSegment3D(
        length=2.0,
        direction=(1.0, 2.0, -1.0),
        center=(0.5, -0.25, 0.75),
        noise=0.0,
    )
    line = line_distribution.sample(256)
    direction = torch.tensor(line_distribution.direction)
    direction = direction / direction.norm()
    centered = line - torch.tensor(line_distribution.center)
    residual = centered - (centered @ direction)[:, None] * direction
    assert residual.norm(dim=1).max() < 1e-5

    disk = PlanarDisk(radius=1.1, height=0.4, noise=0.0).sample(256)
    assert torch.allclose(disk[:, 2], torch.full((256,), 0.4))
    assert disk[:, :2].norm(dim=1).max() <= 1.1


def test_manifold_metadata_reports_intrinsic_dimension() -> None:
    assert SphericalShell(dim=3).metadata()["intrinsic_dim"] == 2
    assert MoebiusStrip().metadata()["intrinsic_dim"] == 2
    assert LineSegment3D().metadata()["intrinsic_dim"] == 1
    assert PlanarDisk().metadata()["intrinsic_dim"] == 2
    assert TrefoilKnot().metadata()["intrinsic_dim"] == 1


def test_gaussian_mixture_3d_log_prob_shape() -> None:
    distribution = GaussianMixture3D(n_modes=4)
    samples = distribution.sample(8)

    log_prob = distribution.log_prob(samples)

    assert log_prob is not None
    assert log_prob.shape == (8,)


def test_nested_shell_samples_land_on_configured_radii_without_noise() -> None:
    distribution = NestedSphericalShells(radii=(0.5, 1.0), noise=0.0)

    samples = distribution.sample(64)
    distances = (samples.norm(dim=1)[:, None] - torch.tensor([0.5, 1.0])[None, :]).abs()

    assert distances.min(dim=1).values.max() < 1e-5


def test_radial_deviation_is_zero_on_configured_radius() -> None:
    x = torch.tensor([[1.0, 0.0], [0.0, -1.0]])

    result = radial_deviation(x, radii=(1.0,))

    assert result["radial_deviation_mean"] == 0.0


def test_radial_tangent_decomposition_2d() -> None:
    x = torch.tensor([[1.0, 0.0]])
    velocity = torch.tensor([[0.0, 2.0]])

    result = radial_tangent_velocity_2d(x, velocity)

    assert result["radial_velocity_abs_mean"] == 0.0
    assert result["tangent_velocity_abs_mean"] == 2.0
