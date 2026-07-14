from pathlib import Path

import torch

from fm_lab.data import LongTailedFashionMNIST
from fm_lab.experiments.factory import build_model, build_path, build_source, build_target
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


def test_imbdiff_round3_configs_encode_paper_protocol() -> None:
    paths = (
        "configs/imbdiff/cifar10_lt_ddpm_epsilon.yaml",
        "configs/imbdiff/cifar100_lt_ddpm_epsilon.yaml",
    )

    for path in paths:
        config = load_config(path)
        expected_classes = 100 if "cifar100" in path else 10
        assert config["data"]["imbalance_factor"] == 0.01
        assert config["source"]["dim"] == 3072
        assert config["conditioning"]["num_classes"] == expected_classes
        assert config["model"]["channel_multipliers"] == [1, 2, 2, 2]
        assert config["model"]["attention_levels"] == [1]
        assert config["diffusion"] == {
            "timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "variance": "fixed_large",
        }
        assert config["objective"] == {
            "name": "discrete_diffusion",
            "prediction_type": "epsilon",
        }
        assert config["training"]["optimizer"] == "adam"
        assert config["training"]["warmup_steps"] == 5000
        assert config["training"]["ema_decay"] == 0.9999
        assert config["sampling"]["classifier_free_guidance"] == {
            "enabled": True,
            "convention": "fm_lab",
            "scale": 2.5,
            "paper_omega": 1.5,
        }


def test_imbdiff_x_vloss_config_changes_only_objective_track() -> None:
    config = load_config("configs/imbdiff/cifar10_lt_x_vloss.yaml")

    assert config["objective"]["prediction_type"] == "x_vloss"
    assert config["experiment"]["track"] == "ddpm_x_vloss"


def test_imbdiff_cbdm_configs_encode_paper_regularizer() -> None:
    paths = (
        "configs/imbdiff/cifar10_lt_cbdm.yaml",
        "configs/imbdiff/cifar100_lt_cbdm.yaml",
    )

    for path in paths:
        config = load_config(path)
        expected_classes = 100 if "cifar100" in path else 10
        objective = build_objective(
            config["objective"],
            diffusion_config=config["diffusion"],
            class_counts=[1] * expected_classes,
        )

        assert objective.method == "cbdm"
        assert config["objective"]["cbdm"] == {
            "target_distribution": "train",
            "tau": 0.001,
            "gamma": 0.25,
        }
        assert config["experiment"]["track"] == "cbdm"


def test_imbdiff_oc_configs_encode_reference_transfer() -> None:
    paths = (
        "configs/imbdiff/cifar10_lt_oc.yaml",
        "configs/imbdiff/cifar100_lt_oc.yaml",
    )

    for path in paths:
        config = load_config(path)
        expected_classes = 100 if "cifar100" in path else 10
        objective = build_objective(
            config["objective"],
            diffusion_config=config["diffusion"],
            class_counts=[1] * expected_classes,
        )

        assert objective.method == "oc"
        assert config["objective"]["oc"] == {
            "transfer_mode": "t2h",
            "cut_time": -1,
        }
        assert config["experiment"]["track"] == "oc"


def test_imbdiff_cm_configs_encode_reference_capacity_objective() -> None:
    paths = (
        "configs/imbdiff/cifar10_lt_cm.yaml",
        "configs/imbdiff/cifar100_lt_cm.yaml",
    )

    for path in paths:
        config = load_config(path)
        expected_classes = 100 if "cifar100" in path else 10
        objective = build_objective(
            config["objective"],
            diffusion_config=config["diffusion"],
            class_counts=[1] * expected_classes,
        )

        assert objective.method == "cm"
        assert config["model"]["capacity"] == {
            "enabled": True,
            "rank_ratio": 0.1,
            "adapter_scale": 1.0,
            "reference_declared_scale": 0.5,
            "parts": ["up"],
        }
        assert config["objective"]["oc"] == {
            "transfer_mode": "t2h",
            "cut_time": -1,
        }
        assert config["objective"]["cm"] == {
            "consistency_weight": 1.0,
            "diversity_weight": 0.2,
        }
        assert config["experiment"]["track"] == "cm"


def test_all_imbdiff_configs_enable_shared_early_stopping() -> None:
    paths = sorted(Path("configs/imbdiff").glob("*.yaml"))
    expected = {
        "enabled": True,
        "patience_steps": 10000,
        "warmup_steps": 20000,
        "min_delta": 0.0001,
        "ema_alpha": 0.01,
    }

    assert len(paths) == 9
    for path in paths:
        assert load_config(path)["training"]["early_stopping"] == expected


def test_imbdiff_local_cifar10_configs_encode_compact_cpu_profile() -> None:
    expected_files = {
        "cifar10_lt_ddpm_epsilon_local.yaml": (
            "discrete_diffusion",
            "epsilon",
            False,
            8000,
            4000,
            2000,
        ),
        "cifar10_lt_x_vloss_local.yaml": (
            "discrete_diffusion",
            "x_vloss",
            False,
            12000,
            6000,
            4000,
        ),
        "cifar10_lt_cbdm_local.yaml": ("cbdm", "epsilon", False, 8000, 4000, 2000),
        "cifar10_lt_oc_local.yaml": ("oc", "epsilon", False, 8000, 4000, 2000),
        "cifar10_lt_cm_local.yaml": ("cm", "x_vloss", True, 12000, 6000, 4000),
    }
    paths = sorted(Path("configs/imbdiff/local").glob("*.yaml"))

    assert {path.name for path in paths} == set(expected_files)
    for path in paths:
        (
            objective_name,
            prediction_type,
            capacity_enabled,
            steps,
            early_warmup_steps,
            patience_steps,
        ) = expected_files[path.name]
        config = load_config(path)
        assert config["model"]["name"] == "image_unet"
        assert config["model"]["image_shape"] == [3, 32, 32]
        assert config["model"]["base_channels"] == 32
        assert config["model"]["time_embedding_dim"] == 128
        assert config["model"]["activation"] == "silu"
        assert config["model"]["zero_init_head"] is True
        assert config["objective"]["name"] == objective_name
        assert config["objective"]["prediction_type"] == prediction_type
        assert config["training"]["batch_size"] == 32
        assert config["training"]["steps"] == steps
        assert config["training"]["warmup_steps"] == 500
        assert config["training"]["checkpoint_every"] == 2000
        assert config["training"]["ema_decay"] == 0.999
        assert config["training"]["early_stopping"] == {
            "enabled": True,
            "patience_steps": patience_steps,
            "warmup_steps": early_warmup_steps,
            "min_delta": 0.0001,
            "ema_alpha": 0.3,
        }
        assert config["sampling"]["n_samples"] == 256
        assert config["sampling"]["sample_batch_size"] == 32
        assert config["sampling"]["plot_max_points"] == 64
        assert config["sampling"]["ddim_skip"] == 16
        assert config["sampling"]["classifier_free_guidance"]["scale"] == 1.0
        assert config["sampling"]["classifier_free_guidance"]["paper_omega"] == 0.0
        assert config["sampling"]["live_ema_comparison"] == {
            "enabled": True,
            "n_samples": 64,
        }
        assert config["experiment"]["output_dir"].startswith("runs/imbdiff/local/")

        source = build_source(config)
        model = build_model(config, dim=source.dim)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert model.is_class_conditional
        if capacity_enabled:
            assert parameter_count > 1_078_569
        else:
            assert parameter_count == 1_078_569
        assert model.capacity_metadata()["enabled"] is capacity_enabled
        if path.name == "cifar10_lt_cm_local.yaml":
            assert config["model"]["capacity"] == {
                "enabled": True,
                "rank_ratio": 0.1,
                "adapter_scale": 1.0,
                "reference_declared_scale": 0.5,
                "parts": ["up"],
            }
            assert config["objective"]["cm"] == {
                "consistency_weight": 1.0,
                "diversity_weight": 0.2,
            }


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
