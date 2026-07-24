import pytest
import torch

import fm_lab.models as models
from fm_lab.experiments.factory import build_model
from fm_lab.models.capacity import (
    CapacityConfig,
    apply_capacity_conv,
    use_capacity_from_context,
)


def _capacity_model_config() -> dict:
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


def _capacity_image_model_config(*, parts: list[str] | None = None) -> dict:
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
                "parts": parts or ["up"],
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


def test_image_unet_factory_places_capacity_only_in_up_blocks() -> None:
    model = build_model(_capacity_image_model_config(), dim=3 * 8 * 8)
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
    linear_adapter_names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, models.SwitchableLowRankLinear)
    ]
    assert linear_adapter_names == []
    assert model.capacity_metadata() == {
        "enabled": True,
        "rank": 0,
        "rank_ratio": 0.25,
        "adapter_scale": 0.5,
        "parts": ["up"],
        "adapter_layers": 4,
        "adapter_conv_layers": 4,
        "adapter_linear_layers": 0,
    }


def test_image_unet_full_capacity_covers_every_model_section() -> None:
    parts = ["conditioning", "head", "down", "middle", "up", "tail"]
    model = build_model(_capacity_image_model_config(parts=parts), dim=3 * 8 * 8)
    adapter_names = {
        name
        for name, module in model.named_modules()
        if isinstance(
            module,
            (models.SwitchableLowRankConv2d, models.SwitchableLowRankLinear),
        )
    }

    assert {
        "time_mlp.0",
        "time_mlp.2",
        "class_projection",
        "input_block.conv1",
        "down1.0",
        "down1_block.conv1",
        "middle.conv1",
        "up1_block.conv1",
        "output_block.2",
    } <= adapter_names
    assert model.capacity_metadata()["parts"] == sorted(parts)


def test_image_unet_full_capacity_preserves_same_seed_shared_parameters() -> None:
    parts = ["conditioning", "head", "down", "middle", "up", "tail"]
    baseline_config = _capacity_image_model_config()
    baseline_config["model"]["capacity"]["enabled"] = False

    torch.manual_seed(23)
    baseline = build_model(baseline_config, dim=3 * 8 * 8)
    torch.manual_seed(23)
    capacity = build_model(_capacity_image_model_config(parts=parts), dim=3 * 8 * 8)

    capacity_state = capacity.state_dict()
    shared_capacity_state = {
        name: value
        for name, value in capacity_state.items()
        if not name.endswith(("adapter_a", "adapter_b"))
    }
    baseline_state = baseline.state_dict()

    assert shared_capacity_state.keys() == baseline_state.keys()
    assert all(
        torch.equal(shared_capacity_state[name], baseline_state[name])
        for name in baseline_state
    )


def test_image_unet_capacity_switch_preserves_base_branch() -> None:
    torch.manual_seed(4)
    model = build_model(_capacity_image_model_config(), dim=3 * 8 * 8).eval()
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
            if isinstance(
                module,
                (models.SwitchableLowRankConv2d, models.SwitchableLowRankLinear),
            ):
                module.adapter_b.normal_(std=0.2)
    changed_full = model(inputs, timesteps, context=full_context)
    unchanged_base = model(inputs, timesteps, context=base_context)

    assert torch.equal(initial_full, initial_base)
    assert not torch.equal(changed_full, initial_full)
    assert torch.equal(unchanged_base, initial_base)


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


def test_low_rank_conv_uses_canonical_flattened_kernel_factors() -> None:
    layer = models.SwitchableLowRankConv2d(4, 6, kernel_size=3, rank=2)

    assert layer.adapter_a.shape == (2 * 3, 4 * 3)
    assert layer.adapter_b.shape == (6 * 3, 2 * 3)


def test_low_rank_linear_switch_applies_factorized_weight() -> None:
    layer = models.SwitchableLowRankLinear(2, 1, rank=1, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
        layer.adapter_a.fill_(2.0)
        layer.adapter_b.fill_(3.0)
    inputs = torch.ones(1, 2)

    assert torch.equal(layer(inputs, use_adapter=False), torch.zeros(1, 1))
    assert torch.equal(layer(inputs, use_adapter=True), torch.full((1, 1), 12.0))


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


def test_low_rank_conv_switch_applies_adapter_scale() -> None:
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


def test_ddpm_factory_places_low_rank_capacity_only_in_selected_unet_parts() -> None:
    model = build_model(_capacity_model_config(), dim=3 * 8 * 8)
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


def test_ddpm_unet_context_switches_adapter_without_changing_base_branch() -> None:
    torch.manual_seed(4)
    model = build_model(_capacity_model_config(), dim=3 * 8 * 8).eval()
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
