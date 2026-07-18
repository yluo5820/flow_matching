from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fm_lab.data import SyntheticLongTailImages
from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    BOUNDED_AZIMUTH_DIMENSION_ID,
    BOUNDED_AZIMUTH_HALF_RANGE,
    BOUNDED_ROTATION_CONDITION_ID,
    BOUNDED_ROTATION_G2_CONDITION_ID,
    BOUNDED_ROTATION_MEDIUM_CONDITION_ID,
    BOUNDED_ROTATION_TAIL_CONDITION_ID,
    DIMENSION_IDS,
    FACTOR_COLUMNS,
    FACTOR_IDENTITY_CONDITION_IDS,
    OBJECT_IDS,
    VIEW_DEPTH_DIMENSION_ID,
    ConditionManifest,
    _object_configs,
    _render_map,
    build_bounded_rotation_control,
    build_bounded_rotation_followups,
    build_condition_manifests,
    build_condition_specs,
    build_factor_identity_control,
    build_factor_space,
    build_local_geometry_queries,
    build_master_pools,
    canonical_factor_rows,
)


def _design_config(
    *,
    master_count: int,
    counts: tuple[int, int, int],
    image_size: int,
) -> dict[str, Any]:
    return {
        "seed": 17,
        "image_size": image_size,
        "master_count": master_count,
        "counts": list(counts),
        "objects": [
            {"id": "stepped_monument", "hue_degrees": 25.0, "scale": 1.0},
            {"id": "crooked_arch", "hue_degrees": 145.0, "scale": 1.0},
            {"id": "three_arm_vane", "hue_degrees": 265.0, "scale": 1.0},
        ],
        "material": {"oklch_lightness": 0.70, "oklch_chroma": 0.12},
        "render": {
            "background": [1.0, 1.0, 1.0],
            "camera_distance": 4.0,
            "supersample": 1,
            "render_batch_size": 8,
        },
    }


def test_render_map_applies_frozen_common_lighting() -> None:
    config = _design_config(master_count=2, counts=(2, 1, 1), image_size=8)
    config["render"].update(
        {
            "ambient": 0.8,
            "diffuse": 0.2,
            "light_energy": 300.0,
            "light_position": [2.0, -3.0, 4.0],
        }
    )
    objects = _object_configs(config)

    render_map = _render_map(config, objects["stepped_monument"], build_factor_space("low"))

    assert render_map.synthetic_render.ambient == 0.8
    assert render_map.synthetic_render.diffuse == 0.2
    assert render_map.synthetic_render.light_energy == 300.0
    assert render_map.synthetic_render.light_position == (2.0, -3.0, 4.0)


def test_object_material_can_be_calibrated_per_fixed_class() -> None:
    config = _design_config(master_count=2, counts=(2, 1, 1), image_size=8)
    config["objects"][0].update({"oklch_lightness": 0.76607792, "oklch_chroma": 0.15916399})

    objects = _object_configs(config)

    assert objects["stepped_monument"]["base_color"] != objects["crooked_arch"]["base_color"]


def test_factorial_conditions_cover_every_object_dimension_frequency_cell() -> None:
    conditions = build_condition_specs(replicate=0)
    assert len(conditions) == 12
    imbalanced = [item for item in conditions if item.frequency_mapping != "balanced"]
    observed = {
        (entry.object_id, entry.dimension_id, entry.count)
        for condition in imbalanced
        for entry in condition.classes
    }
    assert observed == {
        (object_id, dimension_id, count)
        for object_id in OBJECT_IDS
        for dimension_id in DIMENSION_IDS
        for count in (5000, 500, 50)
    }


def test_factor_spaces_have_approved_dimensions_and_canonical_columns() -> None:
    assert FACTOR_COLUMNS == ("tx", "ty", "tz", "azimuth", "elevation")
    for level, expected_dimension in (("low", 1), ("medium", 3), ("high", 5)):
        factor = build_factor_space(level)
        values = factor.sample(4, seed=31).values
        rows = canonical_factor_rows(factor, values)
        assert factor.dim == expected_dimension
        assert rows.dtype == np.float32
        assert rows.shape == (4, 5)
        assert np.all(np.sum(np.isfinite(rows), axis=1) == expected_dimension)


def test_local_geometry_queries_are_normalized_paired_five_factor_tangents() -> None:
    config = _design_config(master_count=2, counts=(2, 1, 1), image_size=16)

    queries, tangents, names, factors = build_local_geometry_queries(
        config,
        object_id="stepped_monument",
        dimension_id=BOUNDED_AZIMUTH_DIMENSION_ID,
        count=2,
        seed=37,
        epsilon=0.02,
    )

    assert queries.shape == (2, 3 * 16 * 16)
    assert tangents.shape == (2, 5, 3 * 16 * 16)
    assert factors.shape == (2, 5)
    assert names == (
        "translation_x",
        "translation_y",
        "translation_z",
        "camera_azimuth",
        "camera_elevation",
    )
    assert np.max(np.abs(queries)) <= 1.0
    assert np.all(np.linalg.norm(tangents, axis=2) > 0.0)


def test_tiny_master_pool_is_uint8_and_condition_views_are_nested(tmp_path: Path) -> None:
    config = _design_config(master_count=20, counts=(20, 5, 2), image_size=16)
    cells = build_master_pools(config, tmp_path, replicate=0)
    paths = build_condition_manifests(tmp_path, 0, cells, counts=(20, 5, 2))
    assert len(cells) == 9
    assert len(paths) == 12
    assert {path.name for path in paths} == {
        "g0_balanced.json",
        "g0_f0.json",
        "g0_f1.json",
        "g0_f2.json",
        "g1_balanced.json",
        "g1_f0.json",
        "g1_f1.json",
        "g1_f2.json",
        "g2_balanced.json",
        "g2_f0.json",
        "g2_f1.json",
        "g2_f2.json",
    }
    assert {path.parent for path in paths} == {tmp_path / "replicate_00" / "conditions"}
    assert all("replicate_00/pools" in cell.image_path for cell in cells)

    image_array = np.load(Path(cells[0].image_path), mmap_mode="r")
    factor_array = np.load(Path(cells[0].factor_path), mmap_mode="r")
    assert image_array.dtype == np.uint8
    assert image_array.shape == (20, 3, 16, 16)
    assert factor_array.dtype == np.float32
    assert factor_array.shape == (20, 5)

    for path in paths:
        manifest = ConditionManifest.read(path)
        assert manifest.condition_id == path.stem
        for entry in manifest.classes:
            assert entry.index_start == 0
            assert entry.count in {20, 5, 2}
            assert not Path(entry.image_path).is_absolute()
            assert not Path(entry.factor_path).is_absolute()
            assert (path.parent / entry.image_path).resolve().is_file()
            assert (path.parent / entry.factor_path).resolve().is_file()


def test_pool_seeds_are_deterministic_and_cell_specific(tmp_path: Path) -> None:
    config = _design_config(master_count=2, counts=(2, 1, 1), image_size=8)
    first = build_master_pools(config, tmp_path / "first", replicate=2)
    second = build_master_pools(config, tmp_path / "second", replicate=2)

    assert [cell.seed for cell in first] == [cell.seed for cell in second]
    assert len({cell.seed for cell in first}) == 9
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.load(left.image_path), np.load(right.image_path))
        np.testing.assert_array_equal(np.load(left.factor_path), np.load(right.factor_path))


def test_pool_and_condition_reruns_refuse_overwrite_without_partial_files(
    tmp_path: Path,
) -> None:
    config = _design_config(master_count=2, counts=(2, 1, 1), image_size=8)
    cells = build_master_pools(config, tmp_path, replicate=0)
    pool_root = tmp_path / "replicate_00" / "pools"
    pool_files_before = {
        path.relative_to(pool_root): path.read_bytes()
        for path in pool_root.rglob("*")
        if path.is_file()
    }

    with pytest.raises(FileExistsError, match="Pool destination already exists"):
        build_master_pools(config, tmp_path, replicate=0)

    pool_files_after = {
        path.relative_to(pool_root): path.read_bytes()
        for path in pool_root.rglob("*")
        if path.is_file()
    }
    assert pool_files_after == pool_files_before

    paths = build_condition_manifests(tmp_path, 0, cells, counts=(2, 1, 1))
    condition_root = tmp_path / "replicate_00" / "conditions"
    manifests_before = {path.name: path.read_bytes() for path in paths}

    with pytest.raises(FileExistsError, match="Condition destination already exists"):
        build_condition_manifests(tmp_path, 0, cells, counts=(2, 1, 1))

    manifests_after = {
        path.name: path.read_bytes() for path in condition_root.iterdir() if path.is_file()
    }
    assert manifests_after == manifests_before


def test_bounded_rotation_control_changes_only_paired_class_zero_azimuth(
    tmp_path: Path,
) -> None:
    config = _design_config(master_count=20, counts=(20, 5, 2), image_size=16)
    build_master_pools(config, tmp_path, replicate=0)

    artifacts = build_bounded_rotation_control(config, tmp_path, replicate=0)
    manifest = ConditionManifest.read(artifacts["manifest"])
    assert manifest.condition_id == BOUNDED_ROTATION_CONDITION_ID
    assert [entry.dimension_id for entry in manifest.classes] == [
        BOUNDED_AZIMUTH_DIMENSION_ID,
        "medium",
        "low",
    ]
    assert [entry.count for entry in manifest.classes] == [20, 20, 20]

    baseline = np.load(tmp_path / "replicate_00/pools/stepped_monument/high/factors.npy")
    bounded = np.load(
        tmp_path
        / "replicate_00/bounded_rotation_control/pools/stepped_monument"
        / BOUNDED_AZIMUTH_DIMENSION_ID
        / "factors.npy"
    )
    np.testing.assert_array_equal(baseline[:, (0, 1, 2, 4)], bounded[:, (0, 1, 2, 4)])
    assert np.max(np.abs(bounded[:, 3])) <= BOUNDED_AZIMUTH_HALF_RANGE
    for entry in manifest.classes[1:]:
        assert (artifacts["manifest"].parent / entry.image_path).resolve() == (
            tmp_path / "replicate_00/pools" / entry.object_id / entry.dimension_id / "images.npy"
        ).resolve()

    with pytest.raises(FileExistsError, match="Bounded-rotation control"):
        build_bounded_rotation_control(config, tmp_path, replicate=0)


def test_bounded_rotation_followups_render_one_pool_and_publish_three_conditions(
    tmp_path: Path,
) -> None:
    config = _design_config(master_count=20, counts=(20, 5, 2), image_size=16)
    build_master_pools(config, tmp_path, replicate=0)
    build_bounded_rotation_control(config, tmp_path, replicate=0)

    artifacts = build_bounded_rotation_followups(config, tmp_path, replicate=0)
    manifests = artifacts["manifests"]
    assert set(manifests) == {
        BOUNDED_ROTATION_G2_CONDITION_ID,
        BOUNDED_ROTATION_MEDIUM_CONDITION_ID,
        BOUNDED_ROTATION_TAIL_CONDITION_ID,
    }
    assert SyntheticLongTailImages(manifests[BOUNDED_ROTATION_G2_CONDITION_ID]).class_counts == (
        20,
        20,
        20,
    )
    assert SyntheticLongTailImages(
        manifests[BOUNDED_ROTATION_MEDIUM_CONDITION_ID]
    ).class_counts == (
        5,
        20,
        20,
    )
    assert SyntheticLongTailImages(manifests[BOUNDED_ROTATION_TAIL_CONDITION_ID]).class_counts == (
        2,
        20,
        20,
    )

    baseline = np.load(tmp_path / "replicate_00/pools/crooked_arch/high/factors.npy")
    bounded = np.load(
        tmp_path
        / "replicate_00/bounded_rotation_followups/pools/crooked_arch"
        / BOUNDED_AZIMUTH_DIMENSION_ID
        / "factors.npy"
    )
    np.testing.assert_array_equal(baseline[:, (0, 1, 2, 4)], bounded[:, (0, 1, 2, 4)])
    assert np.max(np.abs(bounded[:, 3])) <= BOUNDED_AZIMUTH_HALF_RANGE

    with pytest.raises(FileExistsError, match="Bounded-rotation follow-up"):
        build_bounded_rotation_followups(config, tmp_path, replicate=0)


def test_factor_identity_control_replaces_each_objects_three_dimensional_class(
    tmp_path: Path,
) -> None:
    config = _design_config(master_count=20, counts=(20, 5, 2), image_size=16)
    build_master_pools(config, tmp_path, replicate=0)

    artifacts = build_factor_identity_control(config, tmp_path, replicate=0)

    assert set(artifacts["manifests"]) == FACTOR_IDENTITY_CONDITION_IDS
    changed_objects = []
    for geometry in range(3):
        condition_id = f"g{geometry}_balanced_view_depth_3d"
        manifest = ConditionManifest.read(artifacts["manifests"][condition_id])
        assert [entry.count for entry in manifest.classes] == [20, 20, 20]
        changed = [
            entry for entry in manifest.classes if entry.dimension_id == VIEW_DEPTH_DIMENSION_ID
        ]
        assert len(changed) == 1
        changed_objects.append(changed[0].object_id)
        target = SyntheticLongTailImages(artifacts["manifests"][condition_id])
        assert target.class_counts == (20, 20, 20)
    assert set(changed_objects) == set(OBJECT_IDS)

    factors = np.load(
        tmp_path
        / "replicate_00/factor_identity_control/pools/stepped_monument"
        / VIEW_DEPTH_DIMENSION_ID
        / "factors.npy"
    )
    assert np.all(np.isnan(factors[:, :2]))
    assert np.all(np.isfinite(factors[:, 2:]))

    with pytest.raises(FileExistsError, match="Factor-identity control"):
        build_factor_identity_control(config, tmp_path, replicate=0)
