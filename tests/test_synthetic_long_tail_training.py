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


def write_tiny_factorial_manifests(
    root: Path,
    *,
    replicate: int = 0,
) -> tuple[Path, ...]:
    config = {
        "seed": 17,
        "image_size": 32,
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
    cells = build_master_pools(config, root, replicate=replicate)
    return build_condition_manifests(root, replicate, cells, counts=(20, 5, 2))


def training_kwargs(
    manifests: tuple[Path, ...],
    tmp_path: Path,
    *,
    output_name: str = "configs",
) -> dict[str, object]:
    return {
        "base_config_path": PROJECT_ROOT / "configs/synthetic_long_tail_geometry/base_train.yaml",
        "condition_manifests": manifests,
        "output_root": tmp_path / output_name,
        "run_root": tmp_path / "runs",
        "total_steps": 20_000,
        "batch_size": 256,
        "model_seed": 17,
    }


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


@pytest.mark.parametrize(
    ("total_steps", "dataset_size", "batch_size"),
    [
        (True, 15_000, 256),
        ("20000", 15_000, 256),
        (20_000.0, 15_000, 256),
        (0, 15_000, 256),
        (20_000, False, 256),
        (20_000, 0, 256),
        (20_000, 15_000, "256"),
        (20_000, 15_000, 0),
        (1, 1, 1),
    ],
)
def test_matched_pass_checkpoint_rejects_invalid_integer_inputs(
    total_steps: object,
    dataset_size: object,
    batch_size: object,
) -> None:
    with pytest.raises(ValueError):
        matched_pass_step(total_steps, dataset_size, batch_size)  # type: ignore[arg-type]


def test_condition_config_freezes_model_and_changes_only_design_fields(tmp_path: Path) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path)
    paths = write_condition_training_configs(**training_kwargs(manifests, tmp_path))
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
    kwargs = training_kwargs(manifests, tmp_path)
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
    assert target.dim == source.dim == 3_072
    assert len(target.class_counts) == 3
    assert target.image_shape == model.image_shape == (3, 32, 32)
    assert model.image_shape == (3, 32, 32)
    assert model.num_classes == 3


def test_writer_preserves_absolute_run_root_and_validates_full_matrix_before_publish(
    tmp_path: Path,
) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path)
    kwargs = training_kwargs(manifests, tmp_path)
    paths = write_condition_training_configs(**(kwargs | {"run_root": "/scratch/study-runs"}))
    assert load_config(next(path for path in paths if path.stem == "g0_f0"))["experiment"][
        "output_dir"
    ] == "/scratch/study-runs/replicate_00/g0_f0"

    for malformed_manifests, output_name in (
        (manifests[:-1], "missing"),
        (manifests + (manifests[0],), "duplicate"),
    ):
        with pytest.raises(ValueError):
            write_condition_training_configs(
                **training_kwargs(malformed_manifests, tmp_path, output_name=output_name)
            )
        assert not (tmp_path / output_name).exists()


def test_writer_rejects_traversal_wrong_tree_and_non_32_manifest_before_publish(
    tmp_path: Path,
) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path)
    first = manifests[0]
    original = json.loads(first.read_text(encoding="utf-8"))

    traversal = copy.deepcopy(original)
    traversal["condition_id"] = "../../escaped"
    first.write_text(json.dumps(traversal), encoding="utf-8")
    with pytest.raises(ValueError):
        write_condition_training_configs(
            **training_kwargs(manifests, tmp_path, output_name="traversal")
        )
    assert not (tmp_path / "traversal").exists()
    assert not (tmp_path / "escaped").exists()

    first.write_text(json.dumps(original), encoding="utf-8")
    wrong_tree = first.with_name("foreign.json")
    wrong_tree.write_text(json.dumps(original), encoding="utf-8")
    wrong_tree_manifests = (wrong_tree, *manifests[1:])
    with pytest.raises(ValueError):
        write_condition_training_configs(
            **training_kwargs(wrong_tree_manifests, tmp_path, output_name="wrong-tree")
        )
    assert not (tmp_path / "wrong-tree").exists()

    wrong_shape = copy.deepcopy(original)
    wrong_shape["image_shape"] = [3, 16, 16]
    first.write_text(json.dumps(wrong_shape), encoding="utf-8")
    with pytest.raises(ValueError):
        write_condition_training_configs(
            **training_kwargs(manifests, tmp_path, output_name="shape")
        )
    assert not (tmp_path / "shape").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_steps", True),
        ("total_steps", "20000"),
        ("batch_size", 256.0),
        ("batch_size", 0),
        ("model_seed", True),
        ("model_seed", -1),
    ],
)
def test_writer_rejects_invalid_integer_arguments_before_publish(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path)
    output_name = f"invalid-{field}-{value}"
    with pytest.raises(ValueError):
        write_condition_training_configs(
            **(training_kwargs(manifests, tmp_path, output_name=output_name) | {field: value})
        )
    assert not (tmp_path / output_name).exists()


def test_writer_rejects_mixed_manifest_provenance_before_publish(tmp_path: Path) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path)
    altered = json.loads(manifests[0].read_text(encoding="utf-8"))
    altered["config_hash"] = "different-producer-hash"
    manifests[0].write_text(json.dumps(altered), encoding="utf-8")

    with pytest.raises(ValueError, match="config_hash"):
        write_condition_training_configs(
            **training_kwargs(manifests, tmp_path, output_name="mixed-provenance")
        )
    assert not (tmp_path / "mixed-provenance").exists()


def test_writer_hashes_manifest_provenance_and_accepts_each_replicate(tmp_path: Path) -> None:
    manifests = write_tiny_factorial_manifests(tmp_path / "replicate-zero")
    first_paths = write_condition_training_configs(
        **(
            training_kwargs(manifests, tmp_path, output_name="first")
            | {"run_root": "runs/synthetic-long-tail"}
        )
    )
    first_hash = next(path for path in first_paths if path.stem == "g0_f0").with_suffix(
        ".sha256"
    ).read_text(encoding="utf-8")

    equivalent_manifests = write_tiny_factorial_manifests(tmp_path / "equivalent-root")
    equivalent_paths = write_condition_training_configs(
        **(
            training_kwargs(equivalent_manifests, tmp_path, output_name="equivalent")
            | {"run_root": "runs/synthetic-long-tail"}
        )
    )
    equivalent_hash = next(
        path for path in equivalent_paths if path.stem == "g0_f0"
    ).with_suffix(".sha256").read_text(encoding="utf-8")
    assert equivalent_hash == first_hash

    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["config_hash"] = "different-render-provenance"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    changed_paths = write_condition_training_configs(
        **(
            training_kwargs(manifests, tmp_path, output_name="changed")
            | {"run_root": "runs/synthetic-long-tail"}
        )
    )
    changed_hash = next(path for path in changed_paths if path.stem == "g0_f0").with_suffix(
        ".sha256"
    ).read_text(encoding="utf-8")
    assert changed_hash != first_hash

    all_paths = set(first_paths)
    for replicate in (1, 2):
        replicate_manifests = write_tiny_factorial_manifests(
            tmp_path / f"replicate-{replicate}", replicate=replicate
        )
        generated = write_condition_training_configs(
            **training_kwargs(replicate_manifests, tmp_path, output_name=f"configs-{replicate}")
        )
        assert len(generated) == 12
        all_paths.update(generated)
    assert len(all_paths) == 36
