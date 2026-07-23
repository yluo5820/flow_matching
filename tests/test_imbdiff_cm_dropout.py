from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_dropout import probe_imbdiff_cm_dropout
from fm_lab.diagnostics.imbdiff_cm_probe import (
    RestoredImbDiffCMCheckpoint,
    build_imbdiff_cm_probe_manifest,
)
from fm_lab.integrations.official_imbdiff_cm import (
    OfficialImbDiffCMUNet,
    OfficialImbDiffObjective,
)


def _tiny_dropout_model() -> OfficialImbDiffCMUNet:
    model = OfficialImbDiffCMUNet(
        dim=3 * 4 * 4,
        image_shape=(3, 4, 4),
        timesteps=8,
        base_channels=32,
        channel_multipliers=(1,),
        attention_levels=(),
        num_res_blocks=1,
        dropout=0.5,
        num_classes=3,
        rank_ratio=0.1,
        capacity_parts=("up",),
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.fill_(0.01)
    return model


def test_dropout_probe_reports_functional_ratios_and_gradient_controls() -> None:
    model = _tiny_dropout_model().eval()
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
        checkpoint_step=60,
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

    summary, rows = probe_imbdiff_cm_dropout(
        restored,
        clean_images=clean,
        manifest=manifest,
        class_counts=(9, 3, 1),
        repeats=2,
        seed=29,
        compute_gradients=True,
    )

    assert model.training is False
    assert len(rows) == 4 * 4 * 2
    assert {row["condition"] for row in rows} == {
        "expert_plus_dropout",
        "expert_paired_dropout",
        "expert_without_dropout",
        "general_dropout_only",
    }
    assert len(summary["descriptive_ratios"]) == 4
    all_group = next(
        row
        for row in summary["descriptive_ratios"]
        if row["frequency_group"] == "all"
    )
    assert all_group["dropout_only_to_independent_distance"] > 0.0
    assert all_group["paired_to_independent_distance"] >= 0.0
    assert len(summary["gradients"]) == 4 * 3
    dropout_only_expert = next(
        row
        for row in summary["gradients"]
        if row["condition"] == "general_dropout_only"
        and row["parameter_group"] == "expert"
    )
    assert dropout_only_expert["gradient_norm"] == 0.0
    json.dumps(summary)


def test_dropout_probe_without_gradients_omits_gradient_rows() -> None:
    model = _tiny_dropout_model().eval()
    objective = OfficialImbDiffObjective(
        class_counts=(9, 3, 1),
        method="pure_cm",
        timesteps=8,
        cfg=False,
        transfer_x0=False,
        image_shape=(3, 4, 4),
    )
    restored = RestoredImbDiffCMCheckpoint(
        model=model,
        objective=objective,
        config={},
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_step=60,
        method="pure_cm",
        weights="raw",
    )
    manifest = build_imbdiff_cm_probe_manifest(
        np.array([0, 1, 2]),
        np.array([10, 11, 12]),
        timesteps=(3,),
        samples_per_class=1,
        seed=31,
    )
    clean = torch.randn(3, 3, 4, 4)

    summary, _ = probe_imbdiff_cm_dropout(
        restored,
        clean_images=clean,
        manifest=manifest,
        class_counts=(9, 3, 1),
        repeats=1,
        seed=37,
        compute_gradients=False,
    )

    assert summary["gradients"] == []
