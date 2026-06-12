from pathlib import Path

import torch

from fm_lab.experiments.factory import build_model, build_path, build_source, build_target
from fm_lab.utils.config import deep_update, load_config
from fm_lab.utils.logging import create_run_dir


def test_load_default_toy_config() -> None:
    config = load_config("configs/toy/two_moons_baseline.yaml")

    assert config["experiment"]["name"] == "two_moons_linear_independent"
    assert config["data"]["name"] == "two_moons"
    assert config["path"]["name"] == "linear"


def test_deep_update_keeps_nested_values() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    updates = {"a": {"b": 4}}

    assert deep_update(base, updates) == {"a": {"b": 4, "c": 2}, "d": 3}


def test_create_run_dir(tmp_path: Path) -> None:
    config = {"experiment": {"name": "smoke", "seed": 0}}

    run_dir = create_run_dir(config, root=tmp_path / "run")

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metadata.json").exists()
    assert config["experiment"]["output_dir"] == str(run_dir)
    assert not (run_dir / "plots").exists()
    assert not (run_dir / "samples").exists()
    assert not (run_dir / "trajectories").exists()


def test_create_run_dir_uses_suffix_when_default_exists(tmp_path: Path) -> None:
    config = {"experiment": {"name": "smoke", "seed": 0}}

    first = create_run_dir(config, root=tmp_path / "run")
    second = create_run_dir(config, root=tmp_path / "run")
    third = create_run_dir(config, root=tmp_path / "run")

    assert first == tmp_path / "run"
    assert second == tmp_path / "run_1"
    assert third == tmp_path / "run_2"
    assert load_config(third / "config.yaml")["experiment"]["output_dir"] == str(third)


def test_create_run_dir_can_reuse_explicit_path(tmp_path: Path) -> None:
    config = {"experiment": {"name": "smoke", "seed": 0}}

    first = create_run_dir(config, root=tmp_path / "run", unique=False)
    second = create_run_dir(config, root=tmp_path / "run", unique=False)

    assert first == tmp_path / "run"
    assert second == first


def test_3d_linear_toy_configs_build_matching_components() -> None:
    config_paths = (
        "configs/toy/gaussian_to_spherical_shell_linear_3d.yaml",
        "configs/toy/gaussian_to_swiss_roll_linear_3d.yaml",
        "configs/toy/gaussian_to_swiss_roll_linear_3d_straight.yaml",
        "configs/toy/gaussian_to_gaussian_mixture_linear_3d.yaml",
        "configs/toy/gaussian_to_gaussian_mixture_linear_3d_direction_only.yaml",
        "configs/toy/gaussian_to_multi_swiss_roll_linear_3d.yaml",
        "configs/toy/gaussian_to_torus_linear_3d.yaml",
        "configs/toy/gaussian_to_multi_torus_linear_3d.yaml",
        "configs/toy/gaussian_to_helix_mixture_linear_3d.yaml",
        "configs/toy/gaussian_to_nested_spherical_shells_linear_3d.yaml",
    )

    for config_path in config_paths:
        config = load_config(config_path)
        source = build_source(config)
        target = build_target(config)
        path = build_path(config)
        model = build_model(config, dim=source.dim)

        assert source.dim == 3
        assert target.dim == 3
        assert path.name == "linear"
        assert source.sample(8).shape == (8, 3)
        assert target.sample(8).shape == (8, 3)
        x = torch.zeros(8, 3)
        t = torch.zeros(8)
        if getattr(model, "requires_source_label", False):
            assert model(x, t, context={"source_label": x}).shape == (8, 3)
        else:
            assert model(x, t).shape == (8, 3)
