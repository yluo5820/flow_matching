from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_intervention import (
    active_lora_modules,
    probe_imbdiff_cm_intervention,
    reversible_expert_intervention,
)
from fm_lab.diagnostics.imbdiff_cm_knowledge import (
    build_imbdiff_cm_knowledge_manifest,
)
from fm_lab.diagnostics.imbdiff_cm_probe import RestoredImbDiffCMCheckpoint
from fm_lab.integrations.official_imbdiff_cm import (
    OfficialImbDiffCMUNet,
    OfficialImbDiffObjective,
)


def _tiny_model() -> OfficialImbDiffCMUNet:
    model = OfficialImbDiffCMUNet(
        dim=3 * 4 * 4,
        image_shape=(3, 4, 4),
        timesteps=8,
        base_channels=32,
        channel_multipliers=(1,),
        attention_levels=(),
        num_res_blocks=1,
        dropout=0.0,
        num_classes=3,
        rank_ratio=0.1,
        capacity_parts=("up",),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.fill_(0.01)
    return model


def _tiny_objective() -> OfficialImbDiffObjective:
    return OfficialImbDiffObjective(
        class_counts=(9, 3, 1),
        method="pure_cm",
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=False,
        image_shape=(3, 4, 4),
    )


def test_probe_inputs_match_probe_terms() -> None:
    model = _tiny_model().eval()
    objective = _tiny_objective()
    clean = torch.linspace(-1.0, 1.0, 3 * 3 * 4 * 4).reshape(3, 3, 4, 4)
    labels = torch.tensor([0, 1, 2])
    timesteps = torch.tensor([1, 3, 6])
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(13))

    noisy, target = objective.probe_inputs(
        model=model,
        clean=clean,
        labels=labels,
        timesteps=timesteps,
        noise=noise,
    )
    terms = objective.probe_terms(
        model=model,
        clean=clean,
        labels=labels,
        timesteps=timesteps,
        noise=noise,
    )

    torch.testing.assert_close(noisy, terms.noisy)
    torch.testing.assert_close(target, terms.target)


def test_spectrum_random_intervention_restores_factors_bit_exactly() -> None:
    model = _tiny_model().eval()
    modules = active_lora_modules(model)
    originals = {
        name: (
            module.lora_A.detach().clone(),
            module.lora_B.detach().clone(),
        )
        for name, module in modules
    }
    original_products = {
        name: (module.lora_B @ module.lora_A).detach().float() for name, module in modules
    }

    with reversible_expert_intervention(
        model,
        mode="spectrum_random",
        seed=17,
    ) as intervention:
        for name, module in modules:
            random_product = (module.lora_B @ module.lora_A).detach().float()
            torch.testing.assert_close(
                torch.linalg.svdvals(random_product),
                torch.linalg.svdvals(original_products[name]),
                rtol=1e-5,
                atol=1e-6,
            )
            assert not torch.allclose(random_product, original_products[name])

    assert intervention["restoration_verified"] is True
    for name, module in modules:
        original_a, original_b = originals[name]
        assert torch.equal(module.lora_A, original_a)
        assert torch.equal(module.lora_B, original_b)


def test_intervention_probe_emits_paired_tail_controls_and_restores_model() -> None:
    model = _tiny_model().eval()
    objective = _tiny_objective()
    restored = RestoredImbDiffCMCheckpoint(
        model=model,
        objective=objective,
        config={},
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_step=60,
        method="pure_cm",
        weights="ema",
    )
    manifest = build_imbdiff_cm_knowledge_manifest(
        np.array([0, 0, 1, 1, 2, 2]),
        np.arange(6) + 20,
        timesteps=(2, 6),
        samples_per_class=2,
        seed=23,
        fine_to_coarse=(0, 1, 1),
    )
    clean = torch.linspace(-1.0, 1.0, 6 * 3 * 4 * 4).reshape(6, 3, 4, 4)
    originals = {
        name: (
            module.lora_A.detach().clone(),
            module.lora_B.detach().clone(),
        )
        for name, module in active_lora_modules(model)
    }

    (
        summary,
        effect_rows,
        random_rows,
        group_rows,
        class_rows,
        intervention_manifest,
    ) = probe_imbdiff_cm_intervention(
        restored,
        clean_images=clean,
        manifest=manifest,
        class_counts=(9, 3, 1),
        batch_size=3,
        random_repeats=2,
        bootstrap_repeats=50,
        seed=29,
        mixed_precision="off",
    )

    assert len(effect_rows) == 2 * 6
    assert len(random_rows) == 2 * 2 * 6
    assert len(group_rows) == 3 * 4
    assert len(class_rows) == 3
    assert len(summary["tail_selectivity"]) == 3 * 3
    assert summary["zero_validation_max_abs"] < 1e-6
    assert summary["max_random_spectrum_relative_error"] < 1e-5
    assert summary["restoration_verified"] is True
    assert intervention_manifest["restoration_verified"] is True
    assert {row["frequency_group"] for row in effect_rows} == {"many", "medium", "few"}
    assert {row["random_repeat"] for row in random_rows} == {0, 1}
    for timestep in (2, 6):
        learned_rms = np.sqrt(
            np.mean(
                [
                    row["learned_delta_rms"] ** 2
                    for row in effect_rows
                    if row["timestep"] == timestep
                ]
            )
        )
        for repeat in (0, 1):
            matched_rms = np.sqrt(
                np.mean(
                    [
                        row["response_matched_random_delta_rms"] ** 2
                        for row in random_rows
                        if row["timestep"] == timestep and row["random_repeat"] == repeat
                    ]
                )
            )
            assert np.isclose(learned_rms, matched_rms, rtol=1e-5)
    json.dumps(summary, allow_nan=False)
    json.dumps(intervention_manifest, allow_nan=False)
    for name, module in active_lora_modules(model):
        original_a, original_b = originals[name]
        assert torch.equal(module.lora_A, original_a)
        assert torch.equal(module.lora_B, original_b)
