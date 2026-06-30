import torch

from fm_lab.models import (
    DirectionSpeedImageUNet,
    DirectionSpeedMLP,
    ImageUNetVelocity,
)


def test_direction_speed_mlp_directions_are_unit_norm() -> None:
    model = DirectionSpeedMLP(dim=3, hidden_dim=16, depth=2)
    source_label = torch.randn(8, 3)

    direction = model.direction(source_label)

    assert direction.shape == (8, 3)
    assert torch.allclose(direction.norm(dim=1), torch.ones(8), atol=1e-5)


def test_direction_speed_mlp_velocity_is_parallel_to_direction() -> None:
    model = DirectionSpeedMLP(dim=3, hidden_dim=16, depth=2)
    source_label = torch.randn(8, 3)
    x = torch.randn(8, 3)
    t = torch.linspace(0.1, 0.9, 8)

    direction = model.direction(source_label)
    velocity = model(x, t, context={"source_label": source_label})
    projection = (velocity * direction).sum(dim=1, keepdim=True) * direction

    assert velocity.shape == (8, 3)
    assert torch.allclose(velocity, projection, atol=1e-5)


def test_image_unet_velocity_preserves_flattened_image_shape() -> None:
    model = ImageUNetVelocity(
        dim=28 * 28,
        image_shape=(28, 28),
        base_channels=8,
        time_embedding_dim=16,
    )
    x = torch.randn(4, 28 * 28)
    t = torch.linspace(0.0, 1.0, 4)

    velocity = model(x, t)

    assert velocity.shape == x.shape


def test_image_unet_velocity_supports_hwc_rgb_images() -> None:
    model = ImageUNetVelocity(
        dim=32 * 32 * 3,
        image_shape=(32, 32, 3),
        base_channels=8,
        time_embedding_dim=16,
    )
    x = torch.randn(4, 32 * 32 * 3)
    t = torch.linspace(0.0, 1.0, 4)

    velocity = model(x, t)

    assert velocity.shape == x.shape


def test_direction_speed_image_unet_velocity_is_parallel_to_direction() -> None:
    model = DirectionSpeedImageUNet(
        dim=28 * 28,
        image_shape=(28, 28),
        base_channels=8,
        time_embedding_dim=16,
    )
    source_label = torch.randn(2, 28 * 28)
    x = torch.randn(2, 28 * 28)
    t = torch.linspace(0.1, 0.9, 2)

    direction = model.direction(source_label)
    velocity = model(x, t, context={"source_label": source_label})
    projection = (velocity * direction).sum(dim=1, keepdim=True) * direction

    assert direction.shape == (2, 28 * 28)
    assert torch.allclose(direction.norm(dim=1), torch.ones(2), atol=1e-5)
    assert velocity.shape == (2, 28 * 28)
    assert torch.allclose(velocity, projection, atol=1e-5)
