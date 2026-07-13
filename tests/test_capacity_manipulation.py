import torch

import fm_lab.models as models
from fm_lab.experiments.factory import build_model


def _cm_model_config() -> dict:
    return {
        "model": {
            "name": "ddpm_unet",
            "image_shape": [3, 8, 8],
            "base_channels": 32,
            "channel_multipliers": [1, 2],
            "attention_levels": [1],
            "num_res_blocks": 1,
            "dropout": 0.0,
            "capacity": {
                "enabled": True,
                "rank_ratio": 0.25,
                "adapter_scale": 0.5,
                "parts": ["up"],
            },
        },
        "conditioning": {"enabled": True, "num_classes": 10},
        "diffusion": {"timesteps": 1000},
    }


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


def test_cm_factory_places_low_rank_capacity_only_in_selected_unet_parts() -> None:
    model = build_model(_cm_model_config(), dim=3 * 8 * 8)
    adapter_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, models.SwitchableLowRankConv2d)
    ]

    assert adapter_names
    assert all(name.startswith("up_blocks") for name in adapter_names)
    assert model.capacity_metadata() == {
        "enabled": True,
        "rank": 0,
        "rank_ratio": 0.25,
        "adapter_scale": 0.5,
        "parts": ["up"],
        "adapter_layers": len(adapter_names),
    }


def test_cm_unet_context_switches_reserved_capacity_without_changing_base_branch() -> None:
    torch.manual_seed(4)
    model = build_model(_cm_model_config(), dim=3 * 8 * 8).eval()
    with torch.no_grad():
        model.output_conv.weight.normal_(std=0.02)
        model.output_conv.bias.zero_()
    inputs = torch.randn(2, 3 * 8 * 8)
    timesteps = torch.tensor([10, 20])
    labels = torch.tensor([1, 2])
    full_context = {"class_labels": labels, "use_capacity": True}
    base_context = {"class_labels": labels, "use_capacity": False}

    initial_full = model(inputs, timesteps, context=full_context)
    initial_base = model(inputs, timesteps, context=base_context)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, models.SwitchableLowRankConv2d):
                module.adapter_b.normal_(std=0.2)
    changed_full = model(inputs, timesteps, context=full_context)
    unchanged_base = model(inputs, timesteps, context=base_context)

    assert torch.equal(initial_full, initial_base)
    assert not torch.equal(changed_full, initial_full)
    assert torch.equal(unchanged_base, initial_base)
