from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from fm_lab.diagnostics.imbdiff_cm_probe import (
    ImbDiffCMProbeManifest,
    RestoredImbDiffCMCheckpoint,
    build_imbdiff_cm_probe_manifest,
    materialize_probe_noise,
    probe_imbdiff_cm_checkpoint,
    radial_spectral_fractions,
    summarize_gradient_components,
)
from fm_lab.integrations.official_imbdiff_cm import (
    OfficialImbDiffCMUNet,
    OfficialImbDiffObjective,
)


def _tiny_model() -> OfficialImbDiffCMUNet:
    return OfficialImbDiffCMUNet(
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


def test_probe_terms_match_released_pure_cm_trainer_for_fixed_draws() -> None:
    model = _tiny_model().eval()
    objective = OfficialImbDiffObjective(
        class_counts=(9, 3, 1),
        method="pure_cm",
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=False,
        consistency_weight=1.0,
        diversity_weight=0.2,
        image_shape=(3, 4, 4),
    )
    trainer = objective._trainer_for(model, torch.device("cpu"))
    clean = torch.linspace(-1.0, 1.0, 3 * 3 * 4 * 4).reshape(3, 3, 4, 4)
    labels = torch.tensor([0, 1, 2])

    torch.manual_seed(41)
    timesteps = torch.randint(8, size=(3,))
    noise = torch.randn_like(clean)
    terms = objective.probe_terms(
        model=model,
        clean=clean,
        labels=labels,
        timesteps=timesteps,
        noise=noise,
    )

    torch.manual_seed(41)
    released_loss = trainer(clean, labels)

    torch.testing.assert_close(
        terms.total_per_sample.mean(),
        released_loss,
        rtol=1e-7,
        atol=1e-7,
    )
    torch.testing.assert_close(
        terms.total_per_sample,
        terms.base_per_sample
        + objective.consistency_weight * terms.consistency_per_sample
        + objective.diversity_weight * terms.diversity_per_sample,
        rtol=0,
        atol=0,
    )


def test_probe_terms_match_released_endpoint_transfer_target() -> None:
    model = _tiny_model().eval()
    objective = OfficialImbDiffObjective(
        class_counts=(9, 3, 1),
        method="released_cm",
        timesteps=8,
        beta_start=1e-4,
        beta_end=1e-2,
        cfg=False,
        transfer_x0=True,
        transfer_mode="full",
        image_shape=(3, 4, 4),
    )
    clean = torch.linspace(-1.0, 1.0, 3 * 3 * 4 * 4).reshape(3, 3, 4, 4)
    labels = torch.tensor([0, 1, 2])
    timesteps = torch.tensor([1, 3, 6])
    generator = torch.Generator().manual_seed(31)
    noise = torch.randn(clean.shape, generator=generator)
    transfer_seed = 101

    terms = objective.probe_terms(
        model=model,
        clean=clean,
        labels=labels,
        timesteps=timesteps,
        noise=noise,
        transfer_seed=transfer_seed,
    )

    trainer = objective._trainer_for(model, torch.device("cpu"))
    coefficient_shape = (len(clean), 1, 1, 1)
    signal = objective._sqrt_alpha_bars[timesteps].float().reshape(coefficient_shape)
    sigma = objective._sqrt_one_minus_alpha_bars[timesteps].float().reshape(
        coefficient_shape
    )
    noisy = signal * clean + sigma * noise
    cx_t = clean + torch.sqrt(signal.reciprocal().square() - 1.0) * noise
    with torch.random.fork_rng():
        torch.manual_seed(transfer_seed)
        expected_target, _ = trainer.do_transfer_x0(
            noisy,
            cx_t,
            clean,
            timesteps,
            labels,
            return_transfer_label=True,
        )

    torch.testing.assert_close(terms.noisy, noisy)
    torch.testing.assert_close(terms.target, expected_target)


def test_probe_manifest_is_balanced_deterministic_and_digest_checked(tmp_path) -> None:
    labels = np.repeat(np.arange(3), 4)
    original_indices = np.arange(100, 112)
    first = build_imbdiff_cm_probe_manifest(
        labels,
        original_indices,
        timesteps=(1, 5),
        samples_per_class=2,
        seed=17,
    )
    second = build_imbdiff_cm_probe_manifest(
        labels,
        original_indices,
        timesteps=(1, 5),
        samples_per_class=2,
        seed=17,
    )

    assert first.digest == second.digest
    assert first.num_rows == 6
    assert np.array_equal(np.unique(first.labels, return_counts=True)[1], [2, 2, 2])
    path = first.save(tmp_path / "manifest.json")
    loaded = ImbDiffCMProbeManifest.load(path)
    assert loaded.digest == first.digest

    payload = json.loads(path.read_text())
    payload["noise_seeds"][0][0] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="digest"):
        ImbDiffCMProbeManifest.load(path)


def test_materialized_noise_is_row_deterministic() -> None:
    first = materialize_probe_noise([3, 7], (3, 4, 4))
    second = materialize_probe_noise([3, 7], (3, 4, 4))
    reversed_rows = materialize_probe_noise([7, 3], (3, 4, 4))

    assert torch.equal(first, second)
    assert torch.equal(first[0], reversed_rows[1])
    assert torch.equal(first[1], reversed_rows[0])


def test_radial_spectral_fractions_separate_constant_and_checkerboard() -> None:
    constant = torch.ones(2, 3, 8, 8)
    coordinates = torch.arange(8)
    checkerboard = ((-1.0) ** (coordinates[:, None] + coordinates[None, :])).expand(
        2, 3, -1, -1
    )

    constant_spectrum = radial_spectral_fractions(constant)
    checkerboard_spectrum = radial_spectral_fractions(checkerboard)

    torch.testing.assert_close(constant_spectrum["low"], torch.ones(2))
    torch.testing.assert_close(constant_spectrum["high"], torch.zeros(2))
    torch.testing.assert_close(checkerboard_spectrum["low"], torch.zeros(2))
    torch.testing.assert_close(checkerboard_spectrum["high"], torch.ones(2))


def test_gradient_summary_separates_general_and_expert_energy() -> None:
    general = nn.Parameter(torch.zeros(2))
    expert_a = nn.Parameter(torch.zeros(2))
    expert_b = nn.Parameter(torch.zeros(2))
    named_parameters = (
        ("network.base.weight", general),
        ("network.adapter.lora_A", expert_a),
        ("network.adapter.lora_B", expert_b),
    )
    gradients = (
        (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        (torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])),
        (torch.tensor([-1.0, 0.0]), torch.tensor([0.0, -1.0]), torch.tensor([-1.0, 0.0])),
    )

    summary = summarize_gradient_components(
        named_parameters,
        gradients,
        consistency_weight=1.0,
        diversity_weight=0.2,
    )

    assert summary["groups"]["general"]["num_parameters"] == 2
    assert summary["groups"]["expert"]["num_parameters"] == 4
    assert set(summary["expert_layers"]) == {"network.adapter"}
    total_fraction = summary["expert_gradient_energy_fraction"]["total"]
    assert 0.0 < total_fraction < 1.0
    assert summary["groups"]["general"]["cosines"]["base__consistency"] == 0.0


def test_checkpoint_probe_reports_frequency_group_gradients() -> None:
    model = _tiny_model().eval()
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
    restored = RestoredImbDiffCMCheckpoint(
        model=model,
        objective=objective,
        config={},
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_step=5,
        method="pure_cm",
        weights="ema",
    )
    manifest = build_imbdiff_cm_probe_manifest(
        np.array([0, 1, 2]),
        np.array([10, 11, 12]),
        timesteps=(3,),
        samples_per_class=1,
        seed=23,
    )
    clean = torch.linspace(-1.0, 1.0, 3 * 3 * 4 * 4).reshape(3, 3, 4, 4)

    summary, rows = probe_imbdiff_cm_checkpoint(
        restored,
        clean_images=clean,
        manifest=manifest,
        class_counts=(9, 3, 1),
        compute_gradients=True,
    )

    assert len(rows) == 3
    timestep = summary["timesteps"][0]
    assert set(timestep["gradients"]) == {"many", "medium", "few"}
    assert timestep["functional"]["few"]["num_rows"] == 1
    assert (
        timestep["gradients"]["few"]["groups"]["expert"]["num_parameters"]
        > 0
    )
