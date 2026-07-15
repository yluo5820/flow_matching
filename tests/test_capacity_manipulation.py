import pytest
import torch

import fm_lab.models as models
from fm_lab.experiments.factory import build_model
from fm_lab.models.capacity import (
    CapacityConfig,
    apply_capacity_conv,
    use_capacity_from_context,
)
from fm_lab.paths import LinearPath
from fm_lab.training.losses import build_objective


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
    }


def _cm_image_model_config() -> dict:
    return {
        "model": {
            "name": "image_unet",
            "image_shape": [3, 8, 8],
            "base_channels": 32,
            "time_embedding_dim": 128,
            "activation": "silu",
            "zero_init_head": True,
            "capacity": {
                "enabled": True,
                "rank_ratio": 0.25,
                "adapter_scale": 0.5,
                "parts": ["up"],
            },
        },
        "conditioning": {"enabled": True, "num_classes": 10},
    }


def test_capacity_config_builds_only_selected_switchable_convolutions() -> None:
    capacity = CapacityConfig.build(
        rank=0,
        rank_ratio=0.25,
        adapter_scale=0.5,
        parts=["up"],
    )
    up = capacity.conv("up", 8, 12, 3, padding=1)
    down = capacity.conv("down", 8, 12, 3, padding=1)
    inputs = torch.randn(2, 8, 5, 5)

    assert isinstance(up, models.SwitchableLowRankConv2d)
    assert isinstance(down, torch.nn.Conv2d)
    assert not isinstance(down, models.SwitchableLowRankConv2d)
    assert torch.equal(
        apply_capacity_conv(up, inputs, use_capacity=True),
        apply_capacity_conv(up, inputs, use_capacity=False),
    )
    assert use_capacity_from_context({"use_capacity": False}) is False
    assert use_capacity_from_context({}) is True


def test_capacity_config_rejects_unknown_model_parts() -> None:
    with pytest.raises(ValueError, match="Unsupported capacity parts"):
        CapacityConfig.build(
            rank=1,
            rank_ratio=0.0,
            adapter_scale=1.0,
            parts=["unknown"],
        )


def test_image_unet_factory_places_cm_capacity_only_in_up_blocks() -> None:
    model = build_model(_cm_image_model_config(), dim=3 * 8 * 8)
    adapter_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, models.SwitchableLowRankConv2d)
    ]

    assert adapter_names == [
        "up1_block.conv1",
        "up1_block.conv2",
        "up0_block.conv1",
        "up0_block.conv2",
    ]
    assert model.capacity_metadata() == {
        "enabled": True,
        "rank": 0,
        "rank_ratio": 0.25,
        "adapter_scale": 0.5,
        "parts": ["up"],
        "adapter_layers": 4,
    }


def test_image_unet_capacity_switch_preserves_base_branch() -> None:
    torch.manual_seed(4)
    model = build_model(_cm_image_model_config(), dim=3 * 8 * 8).eval()
    with torch.no_grad():
        model.output_block[-1].weight.normal_(std=0.02)
        model.output_block[-1].bias.zero_()
    inputs = torch.randn(2, 3 * 8 * 8)
    timesteps = torch.tensor([0.1, 0.2])
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


def test_cm_objective_accepts_capacity_enabled_image_unet() -> None:
    model = build_model(_cm_image_model_config(), dim=3 * 8 * 8)
    objective = build_objective(
        {
            "name": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
            "modifiers": [
                {
                    "name": "cm",
                    "consistency_weight": 1.0,
                    "diversity_weight": 0.2,
                    "comparison_space": "target",
                },
            ],
        },
        class_counts=[100, 50, 25, 12, 6, 3, 2, 1, 1, 1],
    )
    labels = torch.tensor([0, 9])

    loss, metrics = objective(
        model=model,
        path=LinearPath(),
        x0=torch.randn(2, 3 * 8 * 8),
        x1=torch.randn(2, 3 * 8 * 8),
        t=torch.tensor([0.1, 0.9]),
        class_labels=labels,
        original_class_labels=labels,
    )

    assert torch.isfinite(loss)
    assert "cm.loss" in metrics


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


def test_capacity_adapter_initialization_preserves_global_rng_for_base_layers() -> None:
    torch.manual_seed(17)
    baseline_first = torch.nn.Conv2d(4, 6, kernel_size=3, padding=1)
    baseline_second = torch.nn.Conv2d(6, 8, kernel_size=3, padding=1)

    torch.manual_seed(17)
    capacity_first = models.SwitchableLowRankConv2d(
        4,
        6,
        kernel_size=3,
        padding=1,
        rank=2,
    )
    capacity_second = torch.nn.Conv2d(6, 8, kernel_size=3, padding=1)

    assert torch.equal(capacity_first.weight, baseline_first.weight)
    assert torch.equal(capacity_first.bias, baseline_first.bias)
    assert torch.equal(capacity_second.weight, baseline_second.weight)
    assert torch.equal(capacity_second.bias, baseline_second.bias)


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
    timesteps = torch.tensor([0.1, 0.2])
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
