from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fm_lab.experiments.factory import build_model, build_source, build_target
from fm_lab.experiments.synthetic_long_tail_geometry import (
    matched_pass_step,
    write_condition_training_configs,
)
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    build_condition_manifests,
    build_master_pools,
)
from fm_lab.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_tiny_factorial_manifests(root: Path) -> tuple[Path, ...]:
    config = {
        "seed": 17,
        "image_size": 16,
        "master_count": 20,
        "counts": [20, 5, 2],
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {"supersample": 1, "render_batch_size": 8},
    }
    cells = build_master_pools(config, root, replicate=0)
    return build_condition_manifests(root, 0, cells, counts=(20, 5, 2))


def test_pilot_config_has_frozen_checkpoints_and_no_augmentation() -> None:
    config = load_config(PROJECT_ROOT / "configs/synthetic_long_tail_geometry/base_train.yaml")

    assert config["training"]["checkpoint_steps"] == [5_000, 10_000, 20_000, 40_000]
    assert config["training"]["early_stopping"]["enabled"] is False
    assert config["source"] == {"name": "gaussian", "dim": 3_072}
    assert config["model"]["image_shape"] == [3, 32, 32]
    assert config["conditioning"]["num_classes"] == 3
    assert "augmentation" not in config["data"]


def test_matched_pass_checkpoint_uses_balanced_example_passes() -> None:
    assert matched_pass_step(20_000, dataset_size=5_550, batch_size=256) == 7_400
    assert matched_pass_step(20_000, dataset_size=15_000, batch_size=256) == 20_000


def test_condition_config_freezes_model_and_changes_only_design_fields(tmp_path: Path) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path)
    paths = write_condition_training_configs(
        base_config_path=PROJECT_ROOT / "configs/synthetic_long_tail_geometry/base_train.yaml",
        condition_manifests=manifests,
        output_root=tmp_path / "configs",
        run_root=tmp_path / "runs",
        total_steps=20_000,
        batch_size=256,
        model_seed=17,
    )
    configs = [load_config(path) for path in paths]
    base = load_config(PROJECT_ROOT / "configs/synthetic_long_tail_geometry/base_train.yaml")

    assert len(configs) == 12
    assert {tuple(item["training"]["checkpoint_steps"]) for item in configs} == {
        (7_400, 20_000),
        (20_000,),
    }
    assert len({json.dumps(item["model"], sort_keys=True) for item in configs}) == 1
    assert all(item["training"]["early_stopping"]["enabled"] is False for item in configs)
    assert all(item["training"]["batch_size"] == 256 for item in configs)

    for config in configs:
        comparable = copy.deepcopy(config)
        comparable["experiment"] = copy.deepcopy(base["experiment"])
        comparable["data"]["condition_manifest"] = base["data"]["condition_manifest"]
        comparable["training"]["steps"] = base["training"]["steps"]
        comparable["training"]["checkpoint_steps"] = base["training"]["checkpoint_steps"]
        comparable["training"]["batch_size"] = base["training"]["batch_size"]
        assert comparable == base


def test_generated_configs_have_stable_hashes_refuse_overwrite_and_build_components(
    tmp_path: Path,
) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path)
    kwargs = {
        "base_config_path": PROJECT_ROOT / "configs/synthetic_long_tail_geometry/base_train.yaml",
        "condition_manifests": manifests,
        "output_root": tmp_path / "configs",
        "run_root": tmp_path / "runs",
        "total_steps": 20_000,
        "batch_size": 256,
        "model_seed": 17,
    }
    paths = write_condition_training_configs(**kwargs)

    repeated_paths = write_condition_training_configs(
        **(kwargs | {"output_root": tmp_path / "second-configs"})
    )
    for path, repeated_path in zip(paths, repeated_paths, strict=True):
        config_hash = path.with_suffix(".sha256").read_text(encoding="utf-8")
        assert config_hash == repeated_path.with_suffix(".sha256").read_text(encoding="utf-8")
        assert len(config_hash.strip()) == 64

    with pytest.raises(FileExistsError, match="Training config destination already exists"):
        write_condition_training_configs(**kwargs)

    config = load_config(paths[0])
    source = build_source(config)
    target = build_target(config)
    model = build_model(config, dim=source.dim)

    assert source.dim == 3_072
    assert len(target.class_counts) == 3
    assert model.image_shape == (3, 32, 32)
    assert model.num_classes == 3
