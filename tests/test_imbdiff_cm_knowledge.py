from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fm_lab.diagnostics.imbdiff_cm_knowledge import (
    ImbDiffCMKnowledgeManifest,
    build_imbdiff_cm_knowledge_manifest,
    cifar100_fine_to_coarse,
    linear_probe_rows,
    probe_imbdiff_cm_knowledge,
)
from fm_lab.diagnostics.imbdiff_cm_probe import RestoredImbDiffCMCheckpoint
from fm_lab.integrations.official_imbdiff_cm import (
    OfficialImbDiffCMUNet,
    OfficialImbDiffObjective,
)


def _tiny_knowledge_model() -> OfficialImbDiffCMUNet:
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


def test_knowledge_manifest_round_trip_and_canonical_coarse_mapping(
    tmp_path: Path,
) -> None:
    mapping = cifar100_fine_to_coarse()
    assert len(mapping) == 100
    assert mapping[0] == 4
    assert mapping[99] == 13

    manifest = build_imbdiff_cm_knowledge_manifest(
        np.array([0, 0, 1, 1, 2, 2]),
        np.arange(6) + 10,
        timesteps=(2, 6),
        samples_per_class=2,
        seed=17,
        fine_to_coarse=(0, 1, 1),
    )
    path = manifest.save(tmp_path / "manifest.json")
    restored = ImbDiffCMKnowledgeManifest.load(path)

    assert restored.digest == manifest.digest
    assert set(restored.crossfit_folds[restored.probe.labels == 0]) == {0, 1}
    assert restored.coarse_labels.tolist().count(1) == 4


def test_knowledge_probe_reconstructs_local_expert_and_emits_k1_k2_rows() -> None:
    model = _tiny_knowledge_model().eval()
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
    manifest = build_imbdiff_cm_knowledge_manifest(
        np.array([0, 0, 1, 1, 2, 2]),
        np.arange(6) + 20,
        timesteps=(3, 6),
        samples_per_class=2,
        seed=23,
        fine_to_coarse=(0, 1, 1),
    )
    clean = torch.linspace(-1.0, 1.0, 6 * 3 * 4 * 4).reshape(6, 3, 4, 4)

    (
        summary,
        descriptor_rows,
        atlas,
        probe_rows,
        subspace_rows,
        subspace_pairs,
    ) = probe_imbdiff_cm_knowledge(
        restored,
        clean_images=clean,
        manifest=manifest,
        class_counts=(9, 3, 1),
        batch_size=3,
        sketch_dim=8,
        seed=29,
        permutation_repeats=2,
        subspace_rank=2,
    )

    assert model.training is False
    assert len(summary["layers"]) == 4
    assert len(descriptor_rows) == 4 * 2 * 6
    assert atlas["full_sketch"].shape == (len(descriptor_rows), 8)
    assert atlas["low_pass_sketch"].shape == atlas["full_sketch"].shape
    assert atlas["high_pass_sketch"].shape == atlas["full_sketch"].shape
    assert summary["max_reconstruction_relative_rms"] < 1e-5
    assert {row["task"] for row in probe_rows} == {
        "fine_class",
        "coarse_class",
        "frequency_group",
    }
    assert subspace_rows
    assert len(subspace_pairs["overlap"]) == 4 * 3
    json.dumps(summary, allow_nan=False)


def test_linear_probe_recovers_planted_class_structure_over_nulls() -> None:
    class_ids = np.repeat(np.arange(3), 2)
    folds = np.tile(np.arange(2), 3)
    features = np.eye(3, dtype=np.float64)[class_ids]
    atlas = {
        "layer_index": np.zeros(6, dtype=np.int64),
        "timestep": np.full(6, 3, dtype=np.int64),
        "crossfit_fold": folds,
        "class_id": class_ids,
        "coarse_id": class_ids,
        "frequency_group_id": class_ids,
        "full_sketch": features,
        "low_pass_sketch": features,
        "high_pass_sketch": features,
    }

    rows = linear_probe_rows(
        atlas,
        permutation_repeats=20,
        ridge_alpha=0.1,
        seed=31,
    )

    assert len(rows) == 9
    assert all(row["accuracy"] == 1.0 for row in rows)
    assert all(row["accuracy_minus_permutation"] > 0.3 for row in rows)
