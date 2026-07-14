import pytest
import torch

from fm_lab.paths import (
    ConvertibleFlowPath,
    LinearPath,
    PredictionKind,
    normalize_prediction_kind,
)


def test_prediction_kind_aliases_normalize_to_canonical_values() -> None:
    assert normalize_prediction_kind("epsilon") is PredictionKind.SOURCE
    assert normalize_prediction_kind("x") is PredictionKind.TARGET
    assert normalize_prediction_kind("v") is PredictionKind.VELOCITY
    with pytest.raises(ValueError, match="source, target, or velocity"):
        normalize_prediction_kind("score")


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("source", PredictionKind.SOURCE),
        ("epsilon", PredictionKind.SOURCE),
        ("noise", PredictionKind.SOURCE),
        ("target", PredictionKind.TARGET),
        ("x", PredictionKind.TARGET),
        ("x1", PredictionKind.TARGET),
        ("clean", PredictionKind.TARGET),
        ("velocity", PredictionKind.VELOCITY),
        ("v", PredictionKind.VELOCITY),
        ("field", PredictionKind.VELOCITY),
    ],
)
def test_all_prediction_kind_aliases_are_supported(
    alias: str,
    expected: PredictionKind,
) -> None:
    assert normalize_prediction_kind(alias) is expected
    assert normalize_prediction_kind(alias.upper()) is expected
    assert normalize_prediction_kind(expected) is expected


def test_linear_prediction_converts_target_to_source_and_velocity() -> None:
    path = LinearPath()
    source = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    target = torch.tensor([[3.0, 1.0], [-0.5, 1.25]])
    t = torch.tensor([0.25, 0.75])
    xt = path.sample_xt(source, target, t)
    prediction = path.prediction_state(xt, t).prediction(
        target, PredictionKind.TARGET
    )

    assert torch.allclose(prediction.as_source(), source)
    assert torch.allclose(prediction.as_target(), target)
    assert torch.allclose(prediction.as_velocity(), target - source)


def test_linear_prediction_broadcasts_time_over_image_tensors() -> None:
    path = LinearPath()
    source = torch.randn(2, 3, 4, 4)
    velocity = torch.randn_like(source)
    t = torch.tensor([0.2, 0.8])
    xt = source + t[:, None, None, None] * velocity

    prediction = path.prediction_state(xt, t).prediction(velocity, "velocity")

    assert torch.allclose(prediction.as_source(), source)
    assert torch.allclose(
        prediction.as_target(),
        source + velocity,
    )


def test_linear_prediction_endpoint_conversions_are_finite() -> None:
    path = LinearPath()
    source = torch.randn(2, 1, 2, 2)
    target = torch.randn_like(source)
    t = torch.tensor([0.0, 1.0])
    xt = path.sample_xt(source, target, t)
    state = path.prediction_state(xt, t)

    from_source = state.prediction(source, PredictionKind.SOURCE)
    from_target = state.prediction(target, PredictionKind.TARGET)

    assert torch.isfinite(from_source.as_velocity()).all()
    assert torch.isfinite(from_source.as_target()).all()
    assert torch.isfinite(from_target.as_velocity()).all()
    assert torch.isfinite(from_target.as_source()).all()


def test_linear_velocity_conversion_is_exact_at_endpoints() -> None:
    path = LinearPath()
    source = torch.randn(2, 3)
    target = torch.randn_like(source)
    velocity = target - source
    t = torch.tensor([0.0, 1.0])
    xt = path.sample_xt(source, target, t)
    prediction = path.prediction_state(xt, t).prediction(velocity, "v")

    assert torch.allclose(prediction.as_source(), source)
    assert torch.allclose(prediction.as_target(), target)


def test_linear_prediction_conversion_preserves_gradients() -> None:
    path = LinearPath()
    xt = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3, requires_grad=True)
    state = path.prediction_state(xt, torch.tensor([0.25, 0.75]))

    loss = state.prediction(target, "target").as_velocity().square().sum()
    loss.backward()

    assert xt.grad is not None
    assert target.grad is not None
    assert torch.isfinite(xt.grad).all()
    assert torch.isfinite(target.grad).all()


def test_linear_prediction_state_validates_inputs() -> None:
    path = LinearPath()
    xt = torch.randn(2, 3)

    with pytest.raises(ValueError, match="min_denom must be positive"):
        path.prediction_state(xt, torch.rand(2), min_denom=0)
    with pytest.raises(ValueError, match="t must broadcast against xt"):
        path.prediction_state(xt, torch.rand(4))
    with pytest.raises(ValueError, match="prediction value must match xt shape"):
        path.prediction_state(xt, torch.rand(2)).prediction(
            torch.randn(2, 4), "source"
        )


def test_velocity_only_path_does_not_claim_prediction_conversion() -> None:
    class VelocityOnlyPath:
        name = "velocity-only"

        def sample_xt(self, x0, x1, t, **kwargs):
            return x0

        def target_velocity(self, x0, x1, t, **kwargs):
            return x1 - x0

    path = VelocityOnlyPath()

    assert not isinstance(path, ConvertibleFlowPath)
    with pytest.raises(AttributeError, match="prediction_state"):
        path.prediction_state(torch.zeros(1, 2), torch.zeros(1))
