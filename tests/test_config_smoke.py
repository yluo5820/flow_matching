from pathlib import Path

import torch

from fm_lab.data import LongTailedFashionMNIST
from fm_lab.experiments.factory import (
    build_model,
    build_path,
    build_solvers,
    build_source,
    build_target,
)
from fm_lab.paths import GaussianDiffusionPath
from fm_lab.training.losses import DiffusionObjective, FlowMatchingObjective, build_objective
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
        "configs/toy/gaussian_to_uniform_sphere_surface_linear_3d.yaml",
        "configs/toy/gaussian_to_moebius_strip_linear_3d.yaml",
        "configs/toy/gaussian_to_disjoint_tori_surface_linear_3d.yaml",
        "configs/toy/gaussian_to_line_segment_linear_3d.yaml",
        "configs/toy/gaussian_to_helix_linear_3d.yaml",
        "configs/toy/gaussian_to_planar_disk_linear_3d.yaml",
        "configs/toy/gaussian_to_circle_linear_3d.yaml",
        "configs/toy/gaussian_to_trefoil_knot_linear_3d.yaml",
        "configs/toy/gaussian_to_torus_surface_linear_3d.yaml",
        "configs/toy/gaussian_to_swiss_roll_surface_linear_3d.yaml",
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


def test_3d_learned_acceleration_configs_build_matching_components() -> None:
    config_paths = (
        "configs/toy/gaussian_to_gaussian_mixture_learned_acceleration_3d.yaml",
        "configs/toy/gaussian_to_gaussian_mixture_learned_acceleration_kernel_vstar_3d.yaml",
        (
            "configs/toy/"
            "gaussian_to_gaussian_mixture_learned_acceleration_kernel_vstar_"
            "factorized_polynomial_3d.yaml"
        ),
    )

    for config_path in config_paths:
        config = load_config(config_path)
        source = build_source(config)
        target = build_target(config)
        path = build_path(config)
        model = build_model(config, dim=source.dim)
        objective = build_objective(config["objective"])

        assert source.dim == 3
        assert target.dim == 3
        assert path.name == "learned_acceleration"
        assert isinstance(objective, FlowMatchingObjective)
        if "kernel_vstar" in config_path:
            assert objective.learned_interpolant.mode == "kernel_vstar"
        assert source.sample(8).shape == (8, 3)
        assert target.sample(8).shape == (8, 3)
        x0 = torch.zeros(8, 3)
        x1 = torch.ones(8, 3)
        t = torch.full((8,), 0.5)
        assert path.sample_xt(x0, x1, t).shape == (8, 3)
        assert path.target_velocity(x0, x1, t).shape == (8, 3)
        assert model(x0, t).shape == (8, 3)


def test_mnist_config_builds_matching_components_without_loading_data() -> None:
    config = load_config("configs/mnist/mnist_linear_baseline.yaml")

    source = build_source(config)
    target = build_target(config)
    path = build_path(config)
    model = build_model(config, dim=source.dim)

    assert source.dim == 784
    assert target.dim == 784
    assert path.name == "linear"
    x = torch.zeros(2, 784)
    t = torch.zeros(2)
    assert model(x, t).shape == (2, 784)


def test_fashion_mnist_lt_ir100_config_builds_conditional_components(monkeypatch) -> None:
    monkeypatch.setattr(LongTailedFashionMNIST, "_load", lambda self: None)
    config = load_config("configs/fashion_mnist_lt/fashion_mnist_lt_ir100.yaml")

    source = build_source(config)
    target = build_target(config)
    model = build_model(config, dim=source.dim)
    output = model(
        torch.zeros(2, 784),
        torch.zeros(2),
        {"class_labels": torch.tensor([0, 9])},
    )

    assert isinstance(target, LongTailedFashionMNIST)
    assert source.dim == target.dim == 784
    assert target.image_shape == (1, 28, 28)
    assert config["data"]["imbalance_factor"] == 0.01
    assert config["conditioning"]["num_classes"] == 10
    assert config["sampling"]["n_samples"] == 10_000
    assert output.shape == (2, 784)


def test_diffusion_config_builds_path_and_objective() -> None:
    config = {
        "source": {"name": "gaussian", "dim": 3},
        "data": {"name": "gaussian_mixture_3d"},
        "path": {"name": "gaussian_diffusion", "schedule": "linear"},
        "model": {"name": "mlp", "hidden_dim": 8, "depth": 1},
        "objective": {"name": "diffusion", "prediction_type": "epsilon"},
    }

    source = build_source(config)
    path = build_path(config)
    model = build_model(config, dim=source.dim)
    objective = build_objective(config["objective"])

    assert isinstance(path, GaussianDiffusionPath)
    assert isinstance(objective, DiffusionObjective)
    x0 = torch.zeros(2, 3)
    x1 = torch.ones(2, 3)
    t = torch.full((2,), 0.5)
    assert path.sample_training_tuple(x0, x1, t).xt.shape == (2, 3)
    assert model(x0, t).shape == (2, 3)


FASHION_MNIST_CONTINUOUS_CONFIGS = (
    "fashion_mnist_lt_ir100_x_vloss.yaml",
    "fashion_mnist_lt_ir100_x_vloss_cbdm.yaml",
    "fashion_mnist_lt_ir100_x_vloss_oc.yaml",
    "fashion_mnist_lt_ir100_x_vloss_cm.yaml",
)


def test_continuous_fashion_mnist_configs_share_controlled_fields() -> None:
    configs = [
        load_config(Path("configs/fashion_mnist_lt") / name)
        for name in FASHION_MNIST_CONTINUOUS_CONFIGS
    ]
    controlled_fields = (
        "data",
        "source",
        "coupling",
        "path",
        "conditioning",
        "training",
        "solvers",
        "sampling",
    )

    for field in controlled_fields:
        assert all(config[field] == configs[0][field] for config in configs[1:])
    baseline_model = configs[0]["model"]
    for config in configs[1:]:
        assert {
            key: value for key, value in config["model"].items() if key != "capacity"
        } == baseline_model
    assert configs[0]["data"]["imbalance_factor"] == 0.01
    assert configs[0]["source"] == {"name": "gaussian", "dim": 784}
    assert configs[0]["coupling"]["name"] == "independent"
    assert configs[0]["path"]["name"] == "linear"
    assert configs[0]["model"]["image_shape"] == [1, 28, 28]
    assert configs[0]["sampling"]["n_samples"] == 10_000
    assert configs[0]["sampling"]["classes"] == list(range(10))
    assert configs[0]["sampling"]["nfe"] == 64
    assert configs[0]["experiment"]["seed"] == 0
    assert configs[0]["training"]["time_sampling"] == {
        "name": "logit_normal",
        "mean": -0.8,
        "std": 0.8,
    }
    assert configs[0]["training"]["warmup_steps"] == 500
    assert configs[0]["training"]["ema_decay"] == 0.9999
    assert "gradient_clip" not in configs[0]["training"]
    assert configs[0]["objective"] == {
        "name": "flow_matching",
        "model_output": "target",
        "loss_space": "velocity",
        "min_denom": 0.05,
        "modifiers": [],
    }
    assert configs[1]["objective"]["modifiers"] == [
        {
            "name": "cbdm",
            "target_distribution": "train",
            "tau": 0.001,
            "gamma": 0.25,
            "comparison_space": "velocity",
        }
    ]
    assert configs[2]["objective"]["modifiers"] == [
        {"name": "oc", "transfer_mode": "t2h", "cut_t": None, "min_denom": 0.05}
    ]
    assert configs[3]["objective"]["modifiers"] == [
        {"name": "oc", "transfer_mode": "t2h", "cut_t": None, "min_denom": 0.05},
        {
            "name": "cm",
            "consistency_weight": 1.0,
            "diversity_weight": 0.2,
            "comparison_space": "velocity",
        },
    ]
    assert configs[3]["model"]["capacity"]["parts"] == ["up"]


def test_continuous_fashion_mnist_configs_build_all_components(monkeypatch) -> None:
    monkeypatch.setattr(LongTailedFashionMNIST, "_load", lambda self: None)

    for name in FASHION_MNIST_CONTINUOUS_CONFIGS:
        config = load_config(Path("configs/fashion_mnist_lt") / name)
        source = build_source(config)

        assert isinstance(build_target(config), LongTailedFashionMNIST)
        assert build_path(config).name == "linear"
        assert build_model(config, dim=source.dim).is_class_conditional
        assert build_solvers(config)[0].name == "euler"
        assert isinstance(
            build_objective(config["objective"], class_counts=[1] * 10),
            FlowMatchingObjective,
        )


def test_discrete_imbdiff_training_configs_are_removed() -> None:
    assert not list(Path("configs/imbdiff").rglob("*.yaml"))


def test_shipped_training_configs_use_canonical_objective_schema() -> None:
    forbidden_keys = {"diffusion", "prediction_type", "ddim_skip", "eta", "cut_time"}

    for config_path in Path("configs").rglob("*.yaml"):
        config = load_config(config_path)
        data = config.get("data", {})
        is_active_long_tail = (
            "configs/fashion_mnist_lt" in str(config_path)
            or "imbalance_factor" in data
            or "tail" in str(data.get("variant_id", ""))
        )
        if is_active_long_tail:
            assert not (forbidden_keys & _nested_keys(config)), config_path
        objective = config.get("objective", {})
        assert "x_prediction" not in objective, config_path


def _nested_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for nested in value.values() for key in _nested_keys(nested)
        }
    if isinstance(value, list):
        return {key for nested in value for key in _nested_keys(nested)}
    return set()


def test_mnist_image_unet_configs_build_matching_components_without_loading_data() -> None:
    config_paths = (
        "configs/mnist/mnist_direction_only_image_unet_ot.yaml",
        (
            "configs/mnist/"
            "mnist_learned_acceleration_kernel_vstar_factorized_polynomial_image_unet_ot.yaml"
        ),
    )

    for config_path in config_paths:
        config = load_config(config_path)
        source = build_source(config)
        target = build_target(config)
        path = build_path(config)
        model = build_model(config, dim=source.dim)

        assert source.dim == 784
        assert target.dim == 784
        x = torch.zeros(2, 784)
        t = torch.zeros(2)
        if getattr(model, "requires_source_label", False):
            assert model(x, t, context={"source_label": x}).shape == (2, 784)
        else:
            assert model(x, t).shape == (2, 784)
        if "learned_acceleration" in config_path:
            assert path.name == "learned_acceleration"
            assert path.metadata()["network"] == "image_unet"
            assert path.sample_xt(x, x, t).shape == (2, 784)
        else:
            assert path.name == "linear"


def test_geometry_explorer_model_configs_build_core_components_without_data() -> None:
    config_paths = sorted(
        Path("configs/geometry_explorer/datasets").glob("*/*/models/*.yaml")
    )
    assert config_paths

    for config_path in config_paths:
        config = load_config(config_path)
        source = build_source(config)
        path = build_path(config)
        model = build_model(config, dim=source.dim)
        objective = build_objective(config.get("objective", {}))

        assert source.dim in {784, 1024, 3072}
        assert isinstance(objective, FlowMatchingObjective | DiffusionObjective)
        x = torch.zeros(2, source.dim)
        t = torch.zeros(2)
        assert model(x, t).shape == (2, source.dim)
        if config["path"]["name"] == "gaussian_diffusion":
            assert isinstance(path, GaussianDiffusionPath)
        else:
            assert path.name == "linear"
