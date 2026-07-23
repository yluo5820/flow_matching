from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_probe import RestoredImbDiffCMCheckpoint
from fm_lab.diagnostics.imbdiff_cm_sampling import (
    endpoint_response_scales,
    matched_sampling_inputs,
    quality_contrasts,
    sample_matched_cm_interventions,
)
from fm_lab.experiments.run_imbdiff_cm_sampling_intervention import (
    load_response_scales,
)
from fm_lab.integrations.official_imbdiff_cm import (
    OfficialImbDiffCMUNet,
    OfficialImbDiffObjective,
)


def _tiny_restored() -> RestoredImbDiffCMCheckpoint:
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
    objective = OfficialImbDiffObjective(
        class_counts=(9, 3, 1),
        method="pure_cm",
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=False,
        image_shape=(3, 4, 4),
    )
    return RestoredImbDiffCMCheckpoint(
        model=model.eval(),
        objective=objective,
        config={
            "sampling": {
                "sampler": "ddim",
                "ddim_skip": 2,
                "classifier_free_guidance": {"enabled": False},
            },
            "diffusion": {
                "timesteps": 8,
                "beta_start": 1e-4,
                "beta_end": 1e-2,
                "variance": "fixed_large",
            },
        },
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_step=60,
        method="pure_cm",
        weights="ema",
    )


def test_matched_sampling_inputs_are_balanced_and_deterministic() -> None:
    labels_a, noise_a = matched_sampling_inputs(
        num_classes=3,
        samples_per_class=2,
        image_shape=(3, 4, 4),
        seed=17,
    )
    labels_b, noise_b = matched_sampling_inputs(
        num_classes=3,
        samples_per_class=2,
        image_shape=(3, 4, 4),
        seed=17,
    )

    assert labels_a.tolist() == [0, 0, 1, 1, 2, 2]
    assert torch.equal(labels_a, labels_b)
    assert torch.equal(noise_a, noise_b)


def test_sampling_interventions_reuse_inputs_and_restore_factors(monkeypatch) -> None:
    restored = _tiny_restored()
    originals = {
        name: (module.lora_A.detach().clone(), module.lora_B.detach().clone())
        for name, module in restored.model.named_modules()
        if module.__class__.__name__ == "Conv2d_LoRA" and module.r > 0
    }

    def fake_sampler(*, model, initial_noise, class_labels, **_):
        timesteps = torch.zeros(len(class_labels), dtype=torch.long)
        return model(initial_noise, timesteps, y=class_labels, use_cm=True)

    monkeypatch.setattr(
        "fm_lab.diagnostics.imbdiff_cm_sampling.sample_official_imbdiff",
        fake_sampler,
    )
    payload, manifest = sample_matched_cm_interventions(
        restored,
        samples_per_class=2,
        batch_size=3,
        random_repeats=2,
        seed=19,
        response_scales={0: 0.5, 1: 0.75},
        mixed_precision="off",
    )

    assert set(payload) == {
        "learned",
        "general",
        "random_00",
        "random_01",
        "labels",
        "initial_noise",
    }
    assert manifest["restoration_verified"] is True
    assert manifest["input_seed"] == 19
    assert manifest["intervention_seed"] == 19
    assert manifest["conditions"][1]["response_scale"] == 0.5
    assert manifest["conditions"][2]["response_scale"] == 0.75
    assert not torch.equal(payload["learned"], payload["general"])
    for name, module in restored.model.named_modules():
        if name in originals:
            original_a, original_b = originals[name]
            assert torch.equal(module.lora_A, original_a)
            assert torch.equal(module.lora_B, original_b)


def test_endpoint_response_scales_match_global_endpoint_rms() -> None:
    general = torch.zeros(2, 1, 2, 2)
    learned = torch.full_like(general, 2.0)
    random_00 = torch.full_like(general, 4.0)
    random_01 = torch.full_like(general, 1.0)

    scales, audit = endpoint_response_scales(
        {
            "learned": learned,
            "general": general,
            "random_00": random_00,
            "random_01": random_01,
        },
        base_scales={0: 0.8, 1: 0.6},
    )

    assert scales == {0: 0.4, 1: 1.2}
    assert audit["learned_endpoint_rms"] == 2.0


def test_response_scale_loader_deduplicates_probe_rows(tmp_path: Path) -> None:
    path = tmp_path / "effects.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("random_repeat", "timestep", "response_match_scale"),
        )
        writer.writeheader()
        for _ in range(3):
            writer.writerow({"random_repeat": 0, "timestep": 100, "response_match_scale": 0.4})
            writer.writerow({"random_repeat": 0, "timestep": 900, "response_match_scale": 0.8})
            writer.writerow({"random_repeat": 1, "timestep": 100, "response_match_scale": 0.5})
            writer.writerow({"random_repeat": 1, "timestep": 900, "response_match_scale": 0.7})

    scales = load_response_scales(path, random_repeats=2)

    assert scales.keys() == {0, 1}
    assert np.isclose(scales[0], 0.6)
    assert np.isclose(scales[1], 0.6)


def test_quality_contrasts_report_tail_selectivity() -> None:
    subset_offsets = [-0.1, 0.0, 0.1]

    def estimates(value):
        return [value + offset for offset in subset_offsets]

    metrics = {
        "learned": {
            "kid": 1.0,
            "kid_subset_estimates": estimates(1.0),
            "groups": {
                "many": {"kid": 1.0, "kid_subset_estimates": estimates(1.0)},
                "medium": {"kid": 1.0, "kid_subset_estimates": estimates(1.0)},
                "few": {"kid": 1.0, "kid_subset_estimates": estimates(1.0)},
            },
        },
        "general": {
            "kid": 1.4,
            "kid_subset_estimates": estimates(1.4),
            "groups": {
                "many": {"kid": 1.1, "kid_subset_estimates": estimates(1.1)},
                "medium": {"kid": 1.3, "kid_subset_estimates": estimates(1.3)},
                "few": {"kid": 1.8, "kid_subset_estimates": estimates(1.8)},
            },
        },
        "random_00": {
            "kid": 1.6,
            "kid_subset_estimates": estimates(1.6),
            "groups": {
                "many": {"kid": 1.2, "kid_subset_estimates": estimates(1.2)},
                "medium": {"kid": 1.5, "kid_subset_estimates": estimates(1.5)},
                "few": {"kid": 2.0, "kid_subset_estimates": estimates(2.0)},
            },
        },
        "random_01": {
            "kid": 1.8,
            "kid_subset_estimates": estimates(1.8),
            "groups": {
                "many": {"kid": 1.4, "kid_subset_estimates": estimates(1.4)},
                "medium": {"kid": 1.7, "kid_subset_estimates": estimates(1.7)},
                "few": {"kid": 2.2, "kid_subset_estimates": estimates(2.2)},
            },
        },
    }

    contrasts = quality_contrasts(metrics)

    assert np.isclose(contrasts["overall"]["kid"]["learned_gain_vs_general"], 0.4)
    assert np.isclose(
        contrasts["tail_selectivity"]["kid"]["learned_gain_vs_general_few_minus_many"],
        0.7,
    )
    uncertainty = contrasts["paired_kid_subset_uncertainty"]["overall"]
    assert np.isclose(uncertainty["learned_gain_vs_general"]["mean"], 0.4)
    assert uncertainty["learned_gain_vs_general"]["fraction_positive"] == 1.0
