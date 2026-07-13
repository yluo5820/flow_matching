import pytest
import torch

from fm_lab.experiments.factory import build_model


def _config() -> dict:
    return {
        "model": {
            "name": "ddpm_unet",
            "image_shape": [3, 32, 32],
            "base_channels": 32,
            "channel_multipliers": [1, 2],
            "attention_levels": [1],
            "num_res_blocks": 1,
            "dropout": 0.1,
        },
        "conditioning": {"enabled": True, "num_classes": 10},
        "diffusion": {"timesteps": 1000},
    }


def test_factory_builds_class_conditional_ddpm_unet() -> None:
    model = build_model(_config(), dim=3 * 32 * 32)

    x = torch.randn(2, 3 * 32 * 32)
    output = model(
        x,
        torch.tensor([0, 999]),
        context={"class_labels": torch.tensor([2, -1])},
    )

    assert output.shape == x.shape
    assert model.is_class_conditional
    assert model.num_classes == 10
    assert model.num_timesteps == 1000


def test_ddpm_unet_rejects_incompatible_flat_dimension() -> None:
    with pytest.raises(ValueError, match="does not match image_shape"):
        build_model(_config(), dim=28 * 28)


def test_ddpm_unet_requires_conditioning() -> None:
    config = _config()
    config["conditioning"] = {"enabled": False}

    with pytest.raises(ValueError, match="requires class conditioning"):
        build_model(config, dim=3 * 32 * 32)
