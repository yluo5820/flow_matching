import torch

from fm_lab.models import DirectionSpeedMLP


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
