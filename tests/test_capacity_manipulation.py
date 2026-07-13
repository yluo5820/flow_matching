import torch

import fm_lab.models as models


def test_low_rank_conv_ratio_sets_rank_and_starts_as_base_convolution() -> None:
    low_rank_conv = getattr(models, "SwitchableLowRankConv2d", None)
    assert low_rank_conv is not None

    layer = low_rank_conv(
        8,
        12,
        kernel_size=3,
        padding=1,
        rank_ratio=0.25,
        adapter_scale=0.5,
    )
    inputs = torch.randn(2, 8, 5, 5)

    assert layer.rank == 2
    assert torch.equal(layer(inputs, use_adapter=True), layer(inputs, use_adapter=False))


def test_low_rank_conv_switch_applies_scaled_factorized_weight() -> None:
    layer = models.SwitchableLowRankConv2d(
        1,
        1,
        kernel_size=1,
        rank=1,
        adapter_scale=0.5,
        bias=False,
    )
    with torch.no_grad():
        layer.weight.zero_()
        layer.adapter_a.fill_(2.0)
        layer.adapter_b.fill_(3.0)
    inputs = torch.ones(1, 1, 2, 2)

    assert torch.equal(layer(inputs, use_adapter=False), torch.zeros_like(inputs))
    assert torch.equal(layer(inputs, use_adapter=True), torch.full_like(inputs, 3.0))


def test_low_rank_conv_switch_controls_adapter_gradients() -> None:
    enabled = models.SwitchableLowRankConv2d(2, 2, kernel_size=1, rank=1, bias=False)
    disabled = models.SwitchableLowRankConv2d(2, 2, kernel_size=1, rank=1, bias=False)
    inputs = torch.randn(1, 2, 3, 3)

    enabled(inputs, use_adapter=True).square().sum().backward()
    disabled(inputs, use_adapter=False).square().sum().backward()

    assert enabled.adapter_b.grad is not None
    assert torch.count_nonzero(enabled.adapter_b.grad) > 0
    assert disabled.adapter_a.grad is None
    assert disabled.adapter_b.grad is None
