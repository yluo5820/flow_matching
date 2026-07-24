import copy
import csv
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import fm_lab.experiments.run_train as run_train_module
import fm_lab.training.trainer as trainer_module
from fm_lab.couplings import IndependentCoupling, MinibatchOTCoupling
from fm_lab.data import GaussianMixture3D, TwoMoons
from fm_lab.paths import GaussianDiffusionPath, LearnedAccelerationPath, LinearPath, SphericalPath
from fm_lab.solvers import EulerSolver, HeunSolver
from fm_lab.sources import GaussianSource
from fm_lab.training.losses import FlowMatchingObjective, build_objective
from fm_lab.training.prediction import velocity_model_for_objective
from fm_lab.training.sampling_guidance import (
    DensityGuidanceConfig,
    DensityGuidedDiffusionVelocity,
    apply_density_prior_rescaling,
)
from fm_lab.training.time_sampling import build_training_time_sampler
from fm_lab.training.trainer import (
    _validate_training_compatibility,
    sample_and_plot,
    train_flow_matching,
    validate_checkpoint_compatibility,
)
from fm_lab.utils.checkpoints import load_checkpoint, save_checkpoint


def test_logit_normal_time_sampler_matches_seeded_definition() -> None:
    expected_generator = torch.Generator().manual_seed(17)
    expected = torch.sigmoid(
        -0.8 + 0.8 * torch.randn(32, generator=expected_generator)
    )
    actual_generator = torch.Generator().manual_seed(17)
    sampler = build_training_time_sampler(
        {"name": "logit_normal", "mean": -0.8, "std": 0.8}
    )

    actual = sampler.sample(32, torch.device("cpu"), generator=actual_generator)

    assert torch.equal(actual, expected)


def test_mixed_precision_defaults_to_inactive_on_cpu() -> None:
    config = trainer_module._build_mixed_precision_config(
        {"mixed_precision": {"enabled": True, "dtype": "auto"}},
        torch.device("cpu"),
    )

    assert config.requested is True
    assert config.active is False
    assert config.inactive_reason == "device_type_not_enabled:cpu"


def test_mixed_precision_fp16_cuda_uses_grad_scaler_without_cuda_allocation() -> None:
    config = trainer_module._build_mixed_precision_config(
        {"mixed_precision": {"enabled": True, "dtype": "fp16"}},
        torch.device("cuda"),
    )

    assert config.active is True
    assert config.dtype is torch.float16
    assert config.scaler_enabled is True


def test_channels_last_runtime_flag_marks_image_models_only_on_enabled_devices() -> None:
    model = nn.Conv2d(3, 4, kernel_size=3)
    model.image_shape = (3, 8, 8)

    inactive = trainer_module._build_channels_last_config(
        {"channels_last": {"enabled": True}},
        torch.device("cpu"),
        model,
    )
    active = trainer_module._build_channels_last_config(
        {"channels_last": {"enabled": True}},
        torch.device("cuda"),
        model,
    )
    trainer_module._apply_channels_last(model)

    assert inactive.active is False
    assert inactive.inactive_reason == "device_type_not_enabled:cpu"
    assert active.active is True
    assert model._fm_lab_channels_last is True


def test_compile_runtime_flag_is_cuda_scoped_by_default() -> None:
    inactive = trainer_module._build_compile_config(
        {"compile": {"enabled": True}},
        torch.device("cpu"),
    )
    active = trainer_module._build_compile_config(
        {"compile": {"enabled": True, "mode": "reduce-overhead"}},
        torch.device("cuda"),
    )

    assert inactive.active is False
    assert inactive.inactive_reason == "device_type_not_enabled:cpu"
    assert active.active is True
    assert active.mode == "reduce-overhead"


def test_run_train_runtime_acceleration_cli_overrides() -> None:
    args = SimpleNamespace(
        steps=None,
        batch_size=None,
        resume_from=None,
        mixed_precision="bf16",
        channels_last="on",
        compile="on",
        compile_backend=None,
        compile_mode="reduce-overhead",
        compile_fullgraph=True,
    )

    overrides = run_train_module._training_overrides(args)

    assert overrides["mixed_precision"] == {
        "enabled": True,
        "dtype": "bf16",
        "device_types": ["cuda"],
    }
    assert overrides["channels_last"] == {
        "enabled": True,
        "device_types": ["cuda"],
    }
    assert overrides["compile"] == {
        "enabled": True,
        "device_types": ["cuda"],
        "mode": "reduce-overhead",
        "fullgraph": True,
    }


@pytest.mark.parametrize("config", [None, "uniform", {"name": "uniform"}])
def test_uniform_time_sampler_preserves_legacy_seeded_sampling(config) -> None:
    expected_generator = torch.Generator().manual_seed(23)
    expected = 1e-5 + (1.0 - 2e-5) * torch.rand(
        32, generator=expected_generator
    )
    actual_generator = torch.Generator().manual_seed(23)
    sampler = build_training_time_sampler(config)

    actual = sampler.sample(32, torch.device("cpu"), generator=actual_generator)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "config, match",
    [
        ({"name": "unknown"}, "training.time_sampling.name"),
        ({"name": "logit_normal", "mean": float("nan")}, "mean"),
        ({"name": "logit_normal", "std": 0.0}, "std"),
        ({"name": "logit_normal", "std": float("inf")}, "std"),
        ({"name": "uniform", "extra": 1}, "unsupported"),
    ],
)
def test_invalid_training_time_sampling_config_fails_early(
    config: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_training_time_sampler(config)


class RecordingFlowMatchingObjective(FlowMatchingObjective):
    def __init__(self) -> None:
        super().__init__()
        self.seen_times: list[torch.Tensor] = []

    def __call__(self, **kwargs):
        self.seen_times.append(kwargs["t"].detach().clone())
        return super().__call__(**kwargs)


@pytest.mark.parametrize(
    ("time_sampling", "expect_all_small"),
    [
        ({"name": "logit_normal", "mean": -8.0, "std": 0.01}, True),
        (None, False),
    ],
)
def test_train_uses_configured_time_sampler(
    tmp_path,
    monkeypatch,
    time_sampling: dict[str, object] | None,
    expect_all_small: bool,
) -> None:
    objective = RecordingFlowMatchingObjective()
    monkeypatch.setattr(
        trainer_module,
        "build_objective",
        lambda *args, **kwargs: objective,
    )
    training = {"batch_size": 16, "steps": 1, "lr": 1e-3, "log_every": 1}
    if time_sampling is not None:
        training["time_sampling"] = time_sampling
    torch.manual_seed(0)

    train_flow_matching(
        config={
            "experiment": {"seed": 0},
            "training": training,
            "sampling": {"n_samples": 2, "n_trajectories": 1, "nfe": 1},
        },
        run_dir=tmp_path / ("logit" if expect_all_small else "uniform"),
        target=ClassCountConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=TinyVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert len(objective.seen_times) == 1
    if expect_all_small:
        assert bool((objective.seen_times[0] < 0.01).all())
    else:
        assert bool((objective.seen_times[0] > 0.01).any())


class ZeroVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class RecordingZeroVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        self.batch_sizes.append(int(x.shape[0]))
        return torch.zeros_like(x)


class UnitXDirectionSpeed(nn.Module):
    requires_source_label = True

    def direction(self, source_label: torch.Tensor) -> torch.Tensor:
        direction = torch.zeros_like(source_label)
        direction[:, 0] = 1.0
        return direction

    def speed(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        source_label: torch.Tensor,
    ) -> torch.Tensor:
        del x, t
        return torch.ones(source_label.shape[0], device=source_label.device)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        source_label = context["source_label"]
        return self.speed(x, t, source_label)[:, None] * self.direction(source_label)


class CustomCoupling:
    name = "custom_coupling"

    def pair(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return x0, x1


class ConstantSource:
    dim = 2

    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(n, self.dim, device=device)

    def metadata(self) -> dict[str, str | int]:
        return {"name": "constant_zero", "dim": self.dim}


class ConstantTarget:
    dim = 2

    def sample(self, n: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.ones(n, self.dim, device=device)

    def metadata(self) -> dict[str, str | int]:
        return {"name": "constant_one", "dim": self.dim}


class ClassCountConstantTarget(ConstantTarget):
    class_counts = (4, 1)


class TinyVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del context
        return self.linear(torch.cat((x, t[:, None]), dim=1))


def _checkpoint_schedule_config(
    checkpoint_steps: list[int],
    *,
    steps: int = 5,
    resume_from=None,
) -> dict:
    training = {
        "batch_size": 2,
        "steps": steps,
        "log_every": 1,
        "optimizer": "adam",
        "lr": 1.0e-3,
        "checkpoint_steps": checkpoint_steps,
        "early_stopping": {"enabled": False},
    }
    if resume_from is not None:
        training["resume_from"] = str(resume_from)
    return {
        "experiment": {"seed": 0},
        "path": {"name": "linear"},
        "objective": {"name": "flow_matching"},
        "training": training,
        "sampling": {"n_samples": 2, "n_trajectories": 1, "nfe": 1},
    }


def _train_checkpoint_fixture(*, config: dict, model: nn.Module, run_dir) -> None:
    train_flow_matching(
        config=config,
        run_dir=run_dir,
        target=ClassCountConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )


def test_training_saves_exact_initial_state_when_checkpoint_zero_is_requested(
    tmp_path,
) -> None:
    model = TinyVelocity()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    config = _checkpoint_schedule_config([0, 1, 3], steps=3)

    _train_checkpoint_fixture(config=config, model=model, run_dir=tmp_path)

    zero = load_checkpoint(tmp_path / "checkpoints/step_000000.pt")
    assert zero["step"] == 0
    assert zero["history"] == []
    assert zero["metrics"]["initial_control"] is True
    assert np.isnan(zero["metrics"]["latest_loss"])
    assert all(
        torch.equal(zero["model_state_dict"][name], value)
        for name, value in initial.items()
    )
    assert sorted(
        checkpoint.name for checkpoint in (tmp_path / "checkpoints").glob("*.pt")
    ) == ["step_000000.pt", "step_000001.pt", "step_000003.pt"]


def test_training_does_not_resave_step_zero_when_resuming(tmp_path) -> None:
    first_run = tmp_path / "first"
    _train_checkpoint_fixture(
        config=_checkpoint_schedule_config([0, 1], steps=1),
        model=TinyVelocity(),
        run_dir=first_run,
    )
    zero_state = load_checkpoint(first_run / "checkpoints/step_000000.pt")
    resumed_run = tmp_path / "resumed"

    _train_checkpoint_fixture(
        config=_checkpoint_schedule_config(
            [0, 2],
            steps=2,
            resume_from=first_run / "checkpoints/step_000001.pt",
        ),
        model=TinyVelocity(),
        run_dir=resumed_run,
    )

    assert not (resumed_run / "checkpoints/step_000000.pt").exists()
    unchanged = load_checkpoint(first_run / "checkpoints/step_000000.pt")
    assert unchanged["step"] == zero_state["step"]
    assert all(
        torch.equal(unchanged["model_state_dict"][name], value)
        for name, value in zero_state["model_state_dict"].items()
    )


def test_training_saves_only_requested_checkpoint_steps(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "path": {"name": "linear"},
        "objective": {"name": "flow_matching"},
        "training": {
            "batch_size": 2,
            "steps": 5,
            "log_every": 1,
            "optimizer": "adam",
            "lr": 1.0e-3,
            "checkpoint_steps": [1, 3, 5],
            "early_stopping": {"enabled": False},
        },
        "sampling": {"n_samples": 2, "n_trajectories": 1, "nfe": 1},
    }

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=TinyVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert "diagnostic_stop" not in metrics
    assert sorted(
        checkpoint.name for checkpoint in (tmp_path / "checkpoints").glob("*.pt")
    ) == ["step_000001.pt", "step_000003.pt", "step_000005.pt"]


@pytest.mark.parametrize(
    "checkpoint_steps",
    [[-1, 1], [1, 1], [1, 6], [1.5, 3]],
)
def test_training_rejects_invalid_explicit_checkpoint_steps(
    tmp_path,
    checkpoint_steps,
) -> None:
    config = {
        "training": {
            "batch_size": 2,
            "steps": 5,
            "checkpoint_steps": checkpoint_steps,
        }
    }

    with pytest.raises(ValueError, match="checkpoint_steps"):
        train_flow_matching(
            config=config,
            run_dir=tmp_path,
            target=ConstantTarget(),
            source=ConstantSource(),
            coupling=IndependentCoupling(),
            path=LinearPath(),
            model=TinyVelocity(),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )


class RejectingStateLoadVelocity(TinyVelocity):
    def load_state_dict(self, *args, **kwargs):
        pytest.fail("model state must not be loaded before resume validation")


class CapacityAwareTarget(nn.Module):
    is_class_conditional = True

    def __init__(self) -> None:
        super().__init__()
        self.contexts: list[dict[str, object]] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del t
        self.contexts.append(context)
        return torch.ones_like(x)


class AnalyticalTargetPrediction(nn.Module):
    is_class_conditional = True

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del t
        assert context is not None
        assert "class_labels" in context
        return torch.ones_like(x)


class CapacityBranchTargetPrediction(nn.Module):
    is_class_conditional = True

    def capacity_metadata(self) -> dict[str, object]:
        return {"enabled": True}

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del t
        assert context is not None
        assert "class_labels" in context
        use_capacity = context.get("use_capacity")
        if use_capacity is None:
            value = 3.0
        elif use_capacity:
            value = 2.0
        else:
            value = 1.0
        return torch.full_like(x, value)


class AuditingEulerSolver(EulerSolver):
    def __init__(self, *, min_denom: float) -> None:
        super().__init__()
        self.min_denom = min_denom
        self.evaluations = 0

    def solve(self, v_fn, x0, t_grid, return_trajectory=False, **kwargs):
        def audited_v_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            velocity = v_fn(x, t)
            expected = (torch.ones_like(x) - x) / (
                1.0 - t[:, None]
            ).clamp_min(self.min_denom)
            assert torch.allclose(velocity, expected)
            self.evaluations += 1
            return velocity

        return super().solve(
            audited_v_fn,
            x0,
            t_grid,
            return_trajectory=return_trajectory,
            **kwargs,
        )


class RejectingSolver(EulerSolver):
    def solve(self, *args, **kwargs):
        pytest.fail("solver must not be evaluated")


def _resume_contract_config(
    *,
    modifiers: list[dict[str, object]] | None = None,
    min_denom: float = 0.001,
    path_tag: str = "a",
    imbalance_factor: float = 0.01,
) -> dict[str, object]:
    return {
        "experiment": {"name": "resume-contract", "seed": 7, "output_dir": "runs/a"},
        "source": {"name": "gaussian", "dim": 2, "scale": 1.0},
        "coupling": {"name": "independent"},
        "path": {"name": "linear", "contract_tag": path_tag},
        "model": {
            "name": "mlp",
            "hidden_dims": [32, 32],
            "activation": "silu",
        },
        "conditioning": {
            "enabled": True,
            "num_classes": 2,
            "embedding_dim": 8,
            "dropout_probability": 0.15,
        },
        "objective": {
            "name": "flow_matching",
            "loss": "mse",
            "model_output": "target",
            "loss_space": "velocity",
            "min_denom": min_denom,
            "modifiers": modifiers or [],
        },
        "data": {
            "name": "fashion_mnist_lt",
            "root": "data/fashion_mnist",
            "download": True,
            "imbalance_type": "exp",
            "imbalance_factor": imbalance_factor,
            "subset_seed": 0,
        },
        "training": {
            "steps": 10,
            "batch_size": 4,
            "optimizer": "adamw",
            "lr": 0.001,
            "weight_decay": 0.01,
            "warmup_steps": 5,
            "ema_decay": 0.9,
            "gradient_clip": 1.0,
            "time_sampling": "uniform",
            "accumulation_steps": 1,
            "log_every": 2,
            "checkpoint_every": 2,
        },
        "sampling": {"n_samples": 20, "nfe": 2},
        "evaluation": {"samples_per_class": 2},
        "plotting": {"max_points": 20},
        "solvers": {"names": ["euler"], "nfes": [2]},
    }


def _build_resume_contract(
    config: dict[str, object],
    *,
    class_counts: list[int] | None = None,
) -> dict[str, object]:
    resolved_class_counts = class_counts or [100, 10]
    return trainer_module.build_training_contract(
        config,
        path=LinearPath(),
        objective=build_objective(
            config["objective"], class_counts=resolved_class_counts
        ),
        class_counts=resolved_class_counts,
    )


def test_training_contract_serializes_resolved_objective_path_and_data_semantics() -> None:
    contract = _build_resume_contract(_resume_contract_config())

    assert contract["version"] == 2
    assert contract["payload"]["objective"]["metadata"]["loss"] == "mse"
    assert contract["payload"]["objective"]["metadata"]["min_denom"] == 0.001
    assert contract["payload"]["objective"]["metadata"]["modifiers"] == []
    assert contract["payload"]["path"] == {
        "config": {"contract_tag": "a", "name": "linear"},
        "metadata": {"name": "linear"},
    }
    assert contract["payload"]["data"]["class_counts"] == [100, 10]
    assert "root" not in contract["payload"]["data"]
    assert "download" not in contract["payload"]["data"]
    assert contract["payload"]["source"]["name"] == "gaussian"
    assert contract["payload"]["coupling"]["name"] == "independent"
    assert contract["payload"]["model"]["hidden_dims"] == [32, 32]
    assert contract["payload"]["conditioning"]["dropout_probability"] == 0.15
    assert contract["payload"]["training"]["batch_size"] == 4
    assert contract["payload"]["training"]["time_sampling"] == "uniform"
    assert "steps" not in contract["payload"]["training"]
    assert "log_every" not in contract["payload"]["training"]
    assert "checkpoint_every" not in contract["payload"]["training"]
    assert "sampling" not in contract["payload"]
    assert "evaluation" not in contract["payload"]
    assert "plotting" not in contract["payload"]
    assert "solvers" not in contract["payload"]


_RESUME_CRITICAL_MUTATIONS = [
    pytest.param("source", "scale", 2.0, id="source"),
    pytest.param("coupling", "name", "minibatch_ot", id="coupling"),
    pytest.param("model", "hidden_dims", [64, 64], id="model"),
    pytest.param("conditioning", "embedding_dim", 16, id="conditioning"),
    pytest.param("conditioning", "dropout_probability", 0.25, id="cfg-dropout"),
    pytest.param("training", "batch_size", 8, id="batch-size"),
    pytest.param("training", "optimizer", "adam", id="optimizer"),
    pytest.param("training", "lr", 0.002, id="learning-rate"),
    pytest.param("training", "weight_decay", 0.02, id="weight-decay"),
    pytest.param("training", "warmup_steps", 10, id="warmup-scheduler"),
    pytest.param("training", "ema_decay", 0.95, id="ema"),
    pytest.param("training", "gradient_clip", 0.5, id="gradient-clipping"),
    pytest.param(
        "training",
        "time_sampling",
        {"name": "logit_normal", "mean": -0.8, "std": 0.8},
        id="time-sampling",
    ),
    pytest.param("training", "accumulation_steps", 2, id="gradient-accumulation"),
    pytest.param("experiment", "seed", 8, id="seed"),
]


@pytest.mark.parametrize(("section", "field", "value"), _RESUME_CRITICAL_MUTATIONS)
def test_training_contract_rejects_resume_critical_config_changes(
    section: str,
    field: str,
    value: object,
) -> None:
    saved_config = _resume_contract_config()
    active_config = copy.deepcopy(saved_config)
    active_config[section][field] = value

    with pytest.raises(ValueError, match="training contract.*incompatible"):
        trainer_module.validate_training_contract(
            _build_resume_contract(saved_config),
            _build_resume_contract(active_config),
        )


def test_training_contract_rejects_resolved_class_count_change() -> None:
    config = _resume_contract_config()

    with pytest.raises(ValueError, match="training contract.*incompatible"):
        trainer_module.validate_training_contract(
            _build_resume_contract(config, class_counts=[100, 10]),
            _build_resume_contract(config, class_counts=[100, 9]),
        )


@pytest.mark.parametrize(
    "active_config",
    [
        _resume_contract_config(
            modifiers=[
                {
                    "name": "cbdm",
                    "target_distribution": "train",
                    "tau": 0.1,
                    "gamma": 0.25,
                    "comparison_space": "velocity",
                }
            ]
        ),
        _resume_contract_config(modifiers=[{"name": "oc", "transfer_mode": "t2h"}]),
        _resume_contract_config(min_denom=0.01),
        _resume_contract_config(path_tag="b"),
        _resume_contract_config(imbalance_factor=0.02),
    ],
    ids=["cbdm", "oc", "min-denom", "path", "data"],
)
def test_training_contract_rejects_semantic_resume_changes(
    active_config: dict[str, object],
) -> None:
    saved_config = _resume_contract_config()

    with pytest.raises(ValueError, match="training contract.*incompatible"):
        trainer_module.validate_training_contract(
            _build_resume_contract(saved_config),
            _build_resume_contract(active_config),
        )


@pytest.mark.parametrize(
    "active_modifiers",
    [
        [
            {"name": "oc", "transfer_mode": "t2h"},
            {
                "name": "cbdm",
                "target_distribution": "train",
                "tau": 0.1,
                "gamma": 0.25,
            },
        ],
        [
            {
                "name": "cbdm",
                "target_distribution": "train",
                "tau": 0.2,
                "gamma": 0.25,
            },
            {"name": "oc", "transfer_mode": "t2h"},
        ],
    ],
    ids=["modifier-order", "modifier-parameter"],
)
def test_training_contract_rejects_modifier_order_and_parameters(
    active_modifiers: list[dict[str, object]],
) -> None:
    saved_modifiers = [
        {
            "name": "cbdm",
            "target_distribution": "train",
            "tau": 0.1,
            "gamma": 0.25,
        },
        {"name": "oc", "transfer_mode": "t2h"},
    ]

    with pytest.raises(ValueError, match="training contract.*incompatible"):
        trainer_module.validate_training_contract(
            _build_resume_contract(_resume_contract_config(modifiers=saved_modifiers)),
            _build_resume_contract(_resume_contract_config(modifiers=active_modifiers)),
        )


@pytest.mark.parametrize("mutation", ["missing", "malformed", "tampered"])
def test_training_contract_rejects_missing_malformed_or_tampered_metadata(
    mutation: str,
) -> None:
    active = _build_resume_contract(_resume_contract_config())
    saved: object
    if mutation == "missing":
        saved = None
    elif mutation == "malformed":
        saved = {"version": 1, "payload": []}
    else:
        saved = {**active, "payload": {**active["payload"], "data": {"name": "tampered"}}}

    with pytest.raises(ValueError, match="training contract"):
        trainer_module.validate_training_contract(saved, active)


def test_training_contract_allows_runtime_only_resume_overrides() -> None:
    saved_config = _resume_contract_config()
    active_config = _resume_contract_config()
    active_config["training"] = {
        **active_config["training"],
        "steps": 100,
        "resume_from": "checkpoint.pt",
        "log_every": 50,
        "checkpoint_every": 25,
    }
    active_config["sampling"] = {"n_samples": 10_000, "nfe": 64, "plot_max_points": 100}
    active_config["evaluation"] = {"samples_per_class": 1_000, "nfe": 64}
    active_config["plotting"] = {"max_points": 100, "save_every": 50}
    active_config["solvers"] = {"names": ["rk4"], "nfes": [64]}
    active_config["experiment"] = {
        **active_config["experiment"],
        "output_dir": "runs/resumed",
    }
    active_config["data"] = {
        **active_config["data"],
        "root": "/local/fashion_mnist",
        "download": False,
        "workspace": "/local/workspace",
    }

    trainer_module.validate_training_contract(
        _build_resume_contract(saved_config),
        _build_resume_contract(active_config),
    )


def test_train_rejects_training_contract_mismatch_before_loading_model_state(
    tmp_path,
) -> None:
    saved_config = {
        "path": {"name": "linear"},
        "objective": {
            "name": "flow_matching",
            "loss": "mse",
            "model_output": "velocity",
            "loss_space": "velocity",
            "min_denom": 0.001,
            "modifiers": [],
        },
        "data": {"name": "constant"},
    }
    objective = build_objective(saved_config["objective"])
    checkpoint_model = TinyVelocity()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=checkpoint_model,
        optimizer=torch.optim.Adam(checkpoint_model.parameters(), lr=1.0e-3),
        step=0,
        config=saved_config,
        prediction_contract={
            "path": "linear",
            "objective": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
        },
        training_contract=trainer_module.build_training_contract(
            saved_config,
            path=LinearPath(),
            objective=objective,
            class_counts=None,
        ),
        metrics={},
    )
    active_config = {
        **saved_config,
        "objective": {**saved_config["objective"], "min_denom": 0.01},
        "training": {
            "steps": 1,
            "batch_size": 2,
            "resume_from": str(checkpoint_path),
        },
        "sampling": {"n_samples": 2, "n_trajectories": 1, "nfe": 1},
    }

    with pytest.raises(ValueError, match="training contract.*incompatible"):
        train_flow_matching(
            config=active_config,
            run_dir=tmp_path / "run",
            target=ConstantTarget(),
            source=ConstantSource(),
            coupling=IndependentCoupling(),
            path=LinearPath(),
            model=RejectingStateLoadVelocity(),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )


def test_train_rejects_missing_resume_state_before_loading_model_state(tmp_path) -> None:
    saved_config = {
        "path": {"name": "linear"},
        "objective": {
            "name": "flow_matching",
            "loss": "mse",
            "model_output": "velocity",
            "loss_space": "velocity",
            "min_denom": 0.001,
        },
        "training": {"steps": 1, "batch_size": 2, "optimizer": "adam"},
    }
    objective = build_objective(saved_config["objective"])
    checkpoint_model = TinyVelocity()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=checkpoint_model,
        optimizer=torch.optim.Adam(checkpoint_model.parameters(), lr=1.0e-4),
        step=0,
        config=saved_config,
        prediction_contract={
            "path": "linear",
            "objective": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
        },
        training_contract=trainer_module.build_training_contract(
            saved_config,
            path=LinearPath(),
            objective=objective,
            class_counts=None,
        ),
        metrics={},
    )
    active_config = copy.deepcopy(saved_config)
    active_config["training"]["steps"] = 2
    active_config["training"]["resume_from"] = str(checkpoint_path)

    with pytest.raises(ValueError, match="resume_state version 1"):
        train_flow_matching(
            config=active_config,
            run_dir=tmp_path / "run",
            target=ConstantTarget(),
            source=ConstantSource(),
            coupling=IndependentCoupling(),
            path=LinearPath(),
            model=RejectingStateLoadVelocity(),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(("section", "field", "value"), _RESUME_CRITICAL_MUTATIONS)
def test_resume_critical_mismatches_reject_before_loading_model_state(
    tmp_path,
    section: str,
    field: str,
    value: object,
) -> None:
    saved_config = _resume_contract_config()
    saved_config["conditioning"]["enabled"] = False
    saved_config["training"]["steps"] = 1
    objective = build_objective(saved_config["objective"], class_counts=None)
    checkpoint_model = TinyVelocity()
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=checkpoint_model,
        optimizer=torch.optim.AdamW(checkpoint_model.parameters(), lr=0.001),
        step=0,
        config=saved_config,
        prediction_contract={
            "path": "linear",
            "objective": "flow_matching",
            "model_output": "target",
            "loss_space": "velocity",
        },
        training_contract=trainer_module.build_training_contract(
            saved_config,
            path=LinearPath(),
            objective=objective,
            class_counts=None,
        ),
        metrics={},
    )
    active_config = copy.deepcopy(saved_config)
    active_config[section][field] = value
    active_config["training"]["resume_from"] = str(checkpoint_path)
    active_config["training"]["steps"] = 2

    with pytest.raises(ValueError, match="training contract.*incompatible"):
        train_flow_matching(
            config=active_config,
            run_dir=tmp_path / "run",
            target=ConstantTarget(),
            source=ConstantSource(),
            coupling=IndependentCoupling(),
            path=LinearPath(),
            model=RejectingStateLoadVelocity(),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )


def test_periodic_checkpoint_resume_matches_uninterrupted_training(tmp_path) -> None:
    config = {
        "training": {
            "steps": 4,
            "batch_size": 4,
            "lr": 1e-3,
            "optimizer": "adam",
            "warmup_steps": 2,
            "ema_decay": 0.9,
            "gradient_clip": 1.0,
            "checkpoint_every": 2,
            "log_every": 1,
        },
        "objective": {"name": "diffusion", "prediction_type": "epsilon"},
        "sampling": {"n_samples": 2, "n_trajectories": 2, "nfe": 2},
    }
    torch.manual_seed(19)
    initial = TinyVelocity().state_dict()

    full_model = TinyVelocity()
    full_model.load_state_dict(initial)
    torch.manual_seed(23)
    train_flow_matching(
        config=config,
        run_dir=tmp_path / "full",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=GaussianDiffusionPath(),
        model=full_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    with (tmp_path / "full" / "diagnostics" / "training_history.csv").open(
        newline=""
    ) as handle:
        history = list(csv.DictReader(handle))
    assert history
    assert all(float(row["gradient_norm"]) >= 0.0 for row in history)

    split_model = TinyVelocity()
    split_model.load_state_dict(initial)
    first_config = {**config, "training": {**config["training"], "steps": 2}}
    torch.manual_seed(23)
    train_flow_matching(
        config=first_config,
        run_dir=tmp_path / "split",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=GaussianDiffusionPath(),
        model=split_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    resume_path = tmp_path / "split" / "checkpoints" / "step_000002.pt"
    resumed_model = TinyVelocity()
    resumed_config = {
        **config,
        "training": {**config["training"], "resume_from": str(resume_path)},
    }
    train_flow_matching(
        config=resumed_config,
        run_dir=tmp_path / "resumed",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=GaussianDiffusionPath(),
        model=resumed_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    full = load_checkpoint(tmp_path / "full" / "checkpoint.pt")
    resumed = load_checkpoint(tmp_path / "resumed" / "checkpoint.pt")
    periodic = load_checkpoint(resume_path)
    assert full["training_contract"]["version"] == 2
    assert periodic["training_contract"]["version"] == 2
    assert resumed["training_contract"]["version"] == 2
    for name, tensor in full["model_state_dict"].items():
        assert torch.equal(tensor, resumed["model_state_dict"][name])
    for name, tensor in full["ema_model_state_dict"].items():
        assert torch.equal(tensor, resumed["ema_model_state_dict"][name])


def test_sample_and_plot_reuses_trajectory_sources_across_solvers(tmp_path) -> None:
    config = _sampling_config(seed=123)

    sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=ZeroVelocity(),
        solvers=[EulerSolver(), HeunSolver()],
        device=torch.device("cpu"),
    )

    reference = np.load(tmp_path / "trajectories" / "source_reference_nfe3.npy")
    euler = np.load(tmp_path / "trajectories" / "euler_nfe3.npy")
    heun = np.load(tmp_path / "trajectories" / "heun_nfe3.npy")

    assert np.allclose(euler[0], reference)
    assert np.allclose(heun[0], reference)
    assert np.allclose(euler[0], heun[0])


def test_sampling_seed_makes_plot_sources_reproducible(tmp_path) -> None:
    config = _sampling_config(seed=321)

    torch.manual_seed(1)
    sample_and_plot(
        config=config,
        run_dir=tmp_path / "run_a",
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=ZeroVelocity(),
        solvers=[EulerSolver(), HeunSolver()],
        device=torch.device("cpu"),
    )

    torch.manual_seed(999)
    sample_and_plot(
        config=config,
        run_dir=tmp_path / "run_b",
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=ZeroVelocity(),
        solvers=[EulerSolver(), HeunSolver()],
        device=torch.device("cpu"),
    )

    source_a = np.load(tmp_path / "run_a" / "samples" / "source_reference.npy")
    source_b = np.load(tmp_path / "run_b" / "samples" / "source_reference.npy")
    trajectories_a = np.load(tmp_path / "run_a" / "trajectories" / "source_reference_nfe3.npy")
    trajectories_b = np.load(tmp_path / "run_b" / "trajectories" / "source_reference_nfe3.npy")

    assert np.allclose(source_a, source_b)
    assert np.allclose(trajectories_a, trajectories_b)


def test_sample_and_plot_chunks_large_final_sample_batches(tmp_path) -> None:
    config = _sampling_config(seed=654)
    config["sampling"]["n_samples"] = 9
    config["sampling"]["n_trajectories"] = 2
    config["sampling"]["sample_batch_size"] = 4
    model = RecordingZeroVelocity()

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    generated = np.load(tmp_path / "samples" / "euler_nfe3.npy")
    assert generated.shape == (9, 3)
    assert summary["sample_batch_size"] == 4
    assert max(model.batch_sizes) == 4
    assert 1 in model.batch_sizes


def test_sample_and_plot_supports_source_label_conditioned_models(tmp_path) -> None:
    config = _sampling_config(seed=777)

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=UnitXDirectionSpeed(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    trajectory = np.load(tmp_path / "trajectories" / "euler_nfe3.npy")

    assert trajectory.shape == (4, 5, 3)
    assert summary["line_containment"]["euler"]["off_line_max"] < 1.0e-6
    assert (tmp_path / "samples" / "euler_nfe3.npy").exists()


def test_sample_and_plot_converts_target_predictions_to_velocity(tmp_path) -> None:
    config = _sampling_config(seed=111)
    config["objective"] = {
        "name": "flow_matching",
        "model_output": "target",
        "loss_space": "velocity",
        "min_denom": 0.05,
    }

    sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        path=LinearPath(),
        model=TrainableConstantVelocity(dim=2, value=1.0),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    generated = np.load(tmp_path / "samples" / "euler_nfe3.npy")
    assert np.allclose(generated, 1.0)


def test_sample_and_plot_records_generic_ode_contract_and_balances_labels(tmp_path) -> None:
    config = _sampling_config(seed=112)
    config["conditioning"] = {"enabled": True, "num_classes": 10}
    config["sampling"]["classes"] = [2, 5, 9]
    config["sampling"]["classifier_free_guidance"] = {"scale": 1.25}
    config["objective"] = {
        "name": "flow_matching",
        "model_output": "target",
        "loss_space": "velocity",
        "min_denom": 0.05,
    }
    solver = AuditingEulerSolver(min_denom=0.05)

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        path=LinearPath(),
        model=AnalyticalTargetPrediction(),
        solvers=[solver],
        device=torch.device("cpu"),
    )

    generated = np.load(tmp_path / "samples" / "euler_nfe3.npy")
    labels = np.load(tmp_path / "samples" / "generated_labels.npy")
    counts = np.asarray([(labels == class_id).sum() for class_id in (2, 5, 9)])
    assert np.allclose(generated, 1.0)
    assert solver.evaluations > 0
    assert counts.max() - counts.min() <= 1
    assert summary["output_kind"] == "target"
    assert summary["path"] == "linear"
    assert summary["min_denom"] == 0.05
    assert summary["solvers"] == ["euler"]
    assert summary["nfe"] == 3
    assert summary["guidance"]["classifier_free_guidance"]["scale"] == 1.25
    assert summary["seed"] == 112


def test_sample_and_plot_warns_for_linear_source_output_endpoint(tmp_path) -> None:
    config = _sampling_config(seed=113)
    config["objective"] = {
        "name": "flow_matching",
        "model_output": "source",
        "loss_space": "source",
        "min_denom": 0.05,
    }

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        path=LinearPath(),
        model=TrainableConstantVelocity(dim=2, value=1.0),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert summary["sampling_warnings"][0]["code"] == (
        "linear_source_output_endpoint_degenerate"
    )


def test_adapter_sampling_defaults_to_model_branch(tmp_path) -> None:
    config = _sampling_config(seed=114)
    config["conditioning"] = {"enabled": True, "num_classes": 2}
    config["objective"] = {
        "name": "flow_matching",
        "model_output": "target",
        "loss_space": "velocity",
        "min_denom": 0.05,
    }

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=ClassCountConstantTarget(),
        source=ConstantSource(),
        path=LinearPath(),
        model=CapacityBranchTargetPrediction(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    generated = np.load(tmp_path / "samples" / "euler_nfe3.npy")
    assert np.allclose(generated, 3.0)
    assert "capacity_branch" not in summary


def test_adapter_sampling_capacity_branch_override_can_sample_full_branch(tmp_path) -> None:
    config = _sampling_config(seed=115)
    config["conditioning"] = {"enabled": True, "num_classes": 2}
    config["sampling"]["capacity_branch"] = "full"
    config["objective"] = {
        "name": "flow_matching",
        "model_output": "target",
        "loss_space": "velocity",
        "min_denom": 0.05,
    }

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=ClassCountConstantTarget(),
        source=ConstantSource(),
        path=LinearPath(),
        model=CapacityBranchTargetPrediction(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    generated = np.load(tmp_path / "samples" / "euler_nfe3.npy")
    assert np.allclose(generated, 2.0)
    assert summary["capacity_branch"] == {
        "configured": "full",
        "resolved": "full",
        "use_capacity": True,
    }


def test_resume_rejects_discrete_checkpoint_metadata_before_loading_weights(tmp_path) -> None:
    checkpoint_model = TrainableConstantVelocity(dim=2, value=0.0)
    checkpoint_optimizer = torch.optim.Adam(checkpoint_model.parameters(), lr=1.0e-3)
    checkpoint_path = tmp_path / "discrete.pt"
    save_checkpoint(
        checkpoint_path,
        model=checkpoint_model,
        optimizer=checkpoint_optimizer,
        step=0,
        config={
            "path": {"name": "linear"},
            "objective": {
                "name": "discrete_diffusion",
                "model_output": "target",
                "loss_space": "velocity",
            },
        },
        prediction_contract={
            "path": "linear",
            "objective": "discrete_diffusion",
            "model_output": "target",
            "loss_space": "velocity",
        },
        metrics={},
        history=[],
    )
    config = {
        "path": {"name": "linear"},
        "objective": {
            "name": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
        },
        "training": {
            "batch_size": 2,
            "steps": 1,
            "lr": 1.0e-3,
            "resume_from": str(checkpoint_path),
        },
        "sampling": {"n_samples": 2, "n_trajectories": 1, "nfe": 1},
    }

    with pytest.raises(ValueError, match="discrete checkpoints are incompatible"):
        train_flow_matching(
            config=config,
            run_dir=tmp_path / "run",
            target=ConstantTarget(),
            source=ConstantSource(),
            coupling=IndependentCoupling(),
            path=LinearPath(),
            model=TrainableConstantVelocity(dim=2, value=0.0),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("model_output", ["target", "source"])
def test_sample_and_plot_rejects_non_velocity_predictions_without_path(
    tmp_path,
    model_output: str,
) -> None:
    config = _sampling_config(seed=111)
    config["objective"] = {
        "name": "flow_matching",
        "model_output": model_output,
        "loss_space": "velocity",
    }

    with pytest.raises(ValueError, match="requires a sampling path"):
        sample_and_plot(
            config=config,
            run_dir=tmp_path,
            target=ConstantTarget(),
            source=ConstantSource(),
            model=TrainableConstantVelocity(dim=2, value=1.0),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )


def test_sample_and_plot_rejects_gaussian_x_prediction_before_solver(tmp_path) -> None:
    config = _sampling_config(seed=111)
    config["objective"] = {"name": "diffusion", "prediction_type": "x"}

    with pytest.raises(
        ValueError,
        match="source/target output requires a ConvertibleFlowPath",
    ):
        sample_and_plot(
            config=config,
            run_dir=tmp_path,
            target=ConstantTarget(),
            source=ConstantSource(),
            path=GaussianDiffusionPath(schedule="linear"),
            model=TrainableConstantVelocity(dim=2, value=1.0),
            solvers=[RejectingSolver()],
            device=torch.device("cpu"),
        )


def test_sample_and_plot_preserves_gaussian_velocity_sampling(tmp_path) -> None:
    config = _sampling_config(seed=111)
    config["objective"] = {"name": "diffusion", "prediction_type": "velocity"}

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        path=GaussianDiffusionPath(schedule="linear"),
        model=ZeroVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert summary["output_kind"] == "velocity"
    assert np.allclose(np.load(tmp_path / "samples" / "euler_nfe3.npy"), 0.0)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("path", ""),
        ("objective", "Flow_Matching"),
        ("model_output", "x"),
        ("loss_space", "v"),
    ],
)
def test_checkpoint_contract_rejects_noncanonical_values(
    field: str,
    invalid_value: str,
) -> None:
    config = {
        "path": {"name": "linear"},
        "objective": {
            "name": "flow_matching",
            "model_output": "velocity",
            "loss_space": "velocity",
        },
    }
    prediction_contract = {
        "path": "linear",
        "objective": "flow_matching",
        "model_output": "velocity",
        "loss_space": "velocity",
    }
    prediction_contract[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        validate_checkpoint_compatibility(
            {"prediction_contract": prediction_contract, "config": config},
            active_config=config,
        )


def test_checkpoint_contract_rejects_score_for_flow_matching() -> None:
    config = {
        "path": {"name": "linear"},
        "objective": {
            "name": "flow_matching",
            "model_output": "score",
            "loss_space": "score",
        },
    }
    contract = {
        "path": "linear",
        "objective": "flow_matching",
        "model_output": "score",
        "loss_space": "score",
    }

    with pytest.raises(ValueError, match="score.*DiffusionObjective"):
        validate_checkpoint_compatibility(
            {"prediction_contract": contract, "config": config},
            active_config=config,
        )


def test_sample_and_plot_rejects_diffusion_score_before_solver(tmp_path) -> None:
    config = _sampling_config(seed=111)
    config["objective"] = {"name": "diffusion", "prediction_type": "score"}

    with pytest.raises(ValueError, match="ODE sampling does not support score output"):
        sample_and_plot(
            config=config,
            run_dir=tmp_path,
            target=ConstantTarget(),
            source=ConstantSource(),
            path=GaussianDiffusionPath(schedule="linear"),
            model=ZeroVelocity(),
            solvers=[RejectingSolver()],
            device=torch.device("cpu"),
        )


def test_velocity_model_converts_source_predictions() -> None:
    objective = build_objective(
        {
            "name": "flow_matching",
            "model_output": "source",
            "loss_space": "target",
        }
    )
    model = velocity_model_for_objective(
        TrainableConstantVelocity(dim=2, value=0.0),
        LinearPath(),
        objective,
    )

    velocity = model(torch.full((1, 2), 0.25), torch.full((1,), 0.25))

    assert torch.allclose(velocity, torch.ones_like(velocity))


def test_velocity_model_forwards_class_and_capacity_context() -> None:
    base_model = CapacityAwareTarget()
    objective = build_objective(
        {
            "name": "flow_matching",
            "model_output": "target",
            "loss_space": "velocity",
        }
    )
    model = velocity_model_for_objective(base_model, LinearPath(), objective)

    velocity = model(
        torch.full((1, 2), 0.5),
        torch.full((1,), 0.5),
        context={"class_labels": torch.tensor([1]), "use_capacity": False},
    )

    assert torch.allclose(velocity, torch.ones_like(velocity))
    assert torch.equal(base_model.contexts[-1]["class_labels"], torch.tensor([1]))
    assert base_model.contexts[-1]["use_capacity"] is False


def test_sample_and_plot_applies_prior_guidance_to_sources(tmp_path) -> None:
    config = _sampling_config(seed=222)
    config["sampling"]["guidance"] = {"prior": {"scale": 0.25}}
    source = GaussianSource(dim=2)

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=source,
        model=ZeroVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    with torch.random.fork_rng():
        torch.manual_seed(222)
        expected_samples = 0.25 * source.sample(8)
        expected_trajectories = 0.25 * source.sample(5)

    source_reference = np.load(tmp_path / "samples" / "source_reference.npy")
    trajectory_reference = np.load(tmp_path / "trajectories" / "source_reference_nfe3.npy")

    assert np.allclose(source_reference, expected_samples.numpy())
    assert np.allclose(trajectory_reference, expected_trajectories.numpy())
    assert summary["guidance"]["prior"]["scale"] == 0.25


def test_density_guided_diffusion_velocity_applies_score_bias() -> None:
    path = GaussianDiffusionPath(schedule="linear")
    model = DensityGuidedDiffusionVelocity(
        ZeroVelocity(),
        path,
        DensityGuidanceConfig(quantile=0.8413447460685429),
    )
    x = torch.tensor([[1.0, 2.0]])
    t = torch.full((1,), 0.5)

    guided_velocity = model(x, t)

    assert torch.allclose(guided_velocity, -2.4 * x, atol=1.0e-6)


def test_density_guidance_skips_singular_initial_endpoint() -> None:
    path = GaussianDiffusionPath(schedule="linear")
    model = DensityGuidedDiffusionVelocity(
        ZeroVelocity(),
        path,
        DensityGuidanceConfig(quantile=0.8413447460685429),
    )
    x = torch.tensor([[1.0, 2.0]])
    t = torch.zeros(1)

    guided_velocity = model(x, t)

    assert torch.allclose(guided_velocity, -x, atol=1.0e-6)


def test_density_guidance_rescales_gaussian_sources_to_median_shell() -> None:
    source = GaussianSource(dim=2)
    samples = torch.tensor([[3.0, 4.0], [1.0, 0.0]])

    rescaled = apply_density_prior_rescaling(
        samples,
        source=source,
        config=DensityGuidanceConfig(quantile=0.25),
    )

    expected_norm = float(np.sqrt(2.0 * np.log(2.0)))
    assert torch.allclose(
        rescaled.norm(dim=1),
        torch.full((2,), expected_norm),
        atol=1.0e-6,
    )


def test_sample_and_plot_rejects_density_guidance_without_diffusion_path(tmp_path) -> None:
    config = _sampling_config(seed=333)
    config["sampling"]["guidance"] = {"density": {"quantile": 0.25}}

    try:
        sample_and_plot(
            config=config,
            run_dir=tmp_path,
            target=ConstantTarget(),
            source=ConstantSource(),
            path=LinearPath(),
            model=ZeroVelocity(),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "Gaussian diffusion path" in str(exc)
    else:
        raise AssertionError("Expected unsupported density guidance to fail.")


def test_sample_and_plot_writes_umap_trajectory_when_enabled(tmp_path, monkeypatch) -> None:
    config = _sampling_config(seed=888)
    config["sampling"]["trajectory_umap"] = {
        "enabled": True,
        "max_target_points": 4,
        "n_neighbors": 3,
        "save_coordinates": True,
    }

    def fake_umap_plot(trajectory, output_path, **kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")
        coordinates_path = kwargs["coordinates_path"]
        np.savez_compressed(coordinates_path, trajectory=np.zeros((*trajectory.shape[:2], 3)))
        interactive_path = kwargs["interactive_path"]
        interactive_path.write_text("html", encoding="utf-8")
        return {
            "plot_path": str(output_path),
            "coordinates_path": str(coordinates_path),
            "interactive_path": str(interactive_path),
            "n_steps": int(trajectory.shape[0]),
            "n_trajectories": int(trajectory.shape[1]),
            "target_points": int(kwargs["max_target_points"]),
            "n_neighbors": int(kwargs["n_neighbors"]),
        }

    monkeypatch.setattr(
        "fm_lab.training.trainer.plot_umap_projected_trajectories",
        fake_umap_plot,
    )

    summary = sample_and_plot(
        config=config,
        run_dir=tmp_path,
        target=GaussianMixture3D(n_modes=4),
        source=GaussianSource(dim=3),
        model=ZeroVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert (tmp_path / "plots" / "trajectory_umap3d_euler_nfe3.png").exists()
    assert (tmp_path / "plots" / "trajectory_umap3d_euler_nfe3.html").exists()
    assert (tmp_path / "trajectories" / "euler_nfe3_umap3d.npz").exists()
    assert summary["trajectory_umap"]["euler"]["target_points"] == 4
    assert summary["trajectory_umap"]["euler"]["n_neighbors"] == 3


def test_direction_only_training_guard_accepts_minibatch_ot_coupling() -> None:
    objective = build_objective({"name": "direction_only_straight"})

    _validate_training_compatibility(
        objective,
        MinibatchOTCoupling(),
        LinearPath(),
        UnitXDirectionSpeed(),
    )


def test_direction_only_objective_declares_velocity_prediction_contract() -> None:
    objective = build_objective({"name": "direction_only_straight"})

    assert objective.model_output == "velocity"
    assert objective.loss_space == "velocity"


def test_direction_only_training_guard_accepts_arbitrary_coupling_name() -> None:
    objective = build_objective({"name": "direction_only_straight"})

    _validate_training_compatibility(
        objective,
        CustomCoupling(),
        LinearPath(),
        UnitXDirectionSpeed(),
    )


def test_direction_only_training_guard_rejects_non_linear_path() -> None:
    objective = build_objective({"name": "direction_only_straight"})

    try:
        _validate_training_compatibility(
            objective,
            IndependentCoupling(),
            SphericalPath(),
            UnitXDirectionSpeed(),
        )
    except ValueError as exc:
        assert "linear" in str(exc)
    else:
        raise AssertionError("Expected non-linear path to be rejected.")


def test_train_flow_matching_updates_trainable_learned_acceleration_path(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {
            "name": "flow_matching",
            "straightness": {"weight": 0.1, "sample_size": 4},
            "interpolant_acceleration": {"weight": 0.001},
        },
        "training": {
            "batch_size": 8,
            "steps": 2,
            "log_every": 1,
            "lr": 1.0e-3,
            "learned_acceleration": {
                "warmup_steps": 0,
                "theta_steps": 1,
                "psi_steps": 1,
                "psi_lr": 1.0e-3,
            },
        },
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }
    path = LearnedAccelerationPath(dim=2, hidden_dim=8, depth=1)
    initial_path_state = {key: value.detach().clone() for key, value in path.state_dict().items()}

    train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=TwoMoons(noise=0.0),
        source=GaussianSource(dim=2),
        coupling=IndependentCoupling(),
        path=path,
        model=TrainableTimeScaledVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    checkpoint = load_checkpoint(tmp_path / "checkpoint.pt")

    assert "path_state_dict" in checkpoint
    assert "theta" in checkpoint["optimizer_state_dict"]
    assert "psi" in checkpoint["optimizer_state_dict"]
    assert any(
        not torch.allclose(value, initial_path_state[key])
        for key, value in path.state_dict().items()
    )
    assert (tmp_path / "diagnostics" / "training_history.csv").exists()
    assert (tmp_path / "plots" / "training_loss.png").exists()
    assert (tmp_path / "plots" / "generated_samples_nfe3.png").exists()


def test_train_flow_matching_restores_best_early_stopping_checkpoint(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {"name": "flow_matching"},
        "training": {
            "batch_size": 8,
            "steps": 4,
            "log_every": 1,
            "optimizer": "adam",
            "lr": 0.1,
            "ema_decay": 0.5,
            "early_stopping": {
                "enabled": True,
                "warmup_steps": 0,
                "patience_steps": 1,
                "min_delta": 1.0e9,
                "ema_alpha": 1.0,
            },
        },
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=TrainableConstantVelocity(dim=2, value=0.0),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    checkpoint = load_checkpoint(tmp_path / "checkpoint.pt")

    assert metrics["trained_steps"] == 2
    assert metrics["checkpoint_step"] == 1
    assert metrics["restored_best_checkpoint"] is True
    assert metrics["final_loss"] == metrics["checkpoint_loss"]
    assert metrics["sampling"]["checkpoint_weights"] == "ema"
    assert checkpoint["step"] == 1
    assert checkpoint["metrics"]["checkpoint_step"] == 1
    model_velocity = checkpoint["model_state_dict"]["velocity"]
    ema_velocity = checkpoint["ema_model_state_dict"]["velocity"]
    assert torch.equal(model_velocity, torch.zeros_like(model_velocity))
    assert torch.equal(ema_velocity, model_velocity)


def test_early_stopping_can_monitor_base_loss_instead_of_negative_total() -> None:
    early_stopping = trainer_module._build_early_stopping(
        {
            "enabled": True,
            "monitor": "base.loss",
            "warmup_steps": 0,
            "patience_steps": 1,
            "min_delta": 0.0,
            "ema_alpha": 1.0,
        }
    )
    first = {"step": 1, "loss": -10.0, "base.loss": 1.0}
    second = {"step": 2, "loss": -100.0, "base.loss": 2.0}

    assert early_stopping.update(first) is False
    assert early_stopping.update(second) is True

    assert first["base.loss_ema"] == 1.0
    assert second["base.loss_ema"] == 2.0
    assert early_stopping.best_step == 1
    assert early_stopping.best_loss == -10.0
    assert early_stopping.summary()["monitor"] == "base.loss_ema"


def test_early_stopping_resume_preserves_best_state_and_ignores_log_cadence(
    tmp_path,
) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {"name": "flow_matching"},
        "training": {
            "batch_size": 8,
            "steps": 8,
            "log_every": 1,
            "checkpoint_every": 2,
            "optimizer": "adam",
            "lr": 0.1,
            "ema_decay": 0.5,
            "early_stopping": {
                "enabled": True,
                "warmup_steps": 0,
                "patience_steps": 3,
                "min_delta": 1.0e9,
                "ema_alpha": 1.0,
            },
        },
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }
    initial_model = TrainableConstantVelocity(dim=2, value=0.0)
    initial_state = copy.deepcopy(initial_model.state_dict())

    full_model = TrainableConstantVelocity(dim=2, value=0.0)
    full_model.load_state_dict(initial_state)
    torch.manual_seed(31)
    full_metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path / "full",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=full_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    split_model = TrainableConstantVelocity(dim=2, value=0.0)
    split_model.load_state_dict(initial_state)
    first_config = {
        **config,
        "training": {**config["training"], "steps": 2, "log_every": 2},
    }
    torch.manual_seed(31)
    train_flow_matching(
        config=first_config,
        run_dir=tmp_path / "split",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=split_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    resume_path = tmp_path / "split" / "checkpoints" / "step_000002.pt"
    periodic = load_checkpoint(resume_path)
    assert periodic["resume_state"]["version"] == 1
    assert periodic["resume_state"]["early_stopping"]["best_step"] == 1
    assert periodic["resume_state"]["best_training_state"]["step"] == 1
    assert "resume_state" not in periodic["resume_state"]["best_training_state"]

    inconsistent_periodic = copy.deepcopy(periodic)
    inconsistent_periodic["resume_state"]["best_training_state"] = None
    inconsistent_periodic_path = tmp_path / "inconsistent_periodic.pt"
    torch.save(inconsistent_periodic, inconsistent_periodic_path)
    inconsistent_config = {
        **config,
        "training": {
            **config["training"],
            "resume_from": str(inconsistent_periodic_path),
        },
    }
    with pytest.raises(ValueError, match="resume_state.*best_training_state"):
        train_flow_matching(
            config=inconsistent_config,
            run_dir=tmp_path / "inconsistent_periodic",
            target=ConstantTarget(),
            source=ConstantSource(),
            coupling=IndependentCoupling(),
            path=LinearPath(),
            model=RejectingStateLoadVelocity(),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )

    resumed_model = TrainableConstantVelocity(dim=2, value=99.0)
    resumed_config = {
        **config,
        "training": {
            **config["training"],
            "resume_from": str(resume_path),
            "log_every": 99,
        },
    }
    resumed_metrics = train_flow_matching(
        config=resumed_config,
        run_dir=tmp_path / "resumed",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=resumed_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    full_checkpoint = load_checkpoint(tmp_path / "full" / "checkpoint.pt")
    resumed_checkpoint = load_checkpoint(tmp_path / "resumed" / "checkpoint.pt")
    assert full_checkpoint["resume_state"]["version"] == 1
    assert resumed_checkpoint["resume_state"]["version"] == 1
    assert (
        resumed_checkpoint["resume_state"]["best_training_state"]["step"] == 1
    )
    assert (
        "resume_state"
        not in resumed_checkpoint["resume_state"]["best_training_state"]
    )
    assert full_metrics["trained_steps"] == resumed_metrics["trained_steps"] == 4
    assert full_metrics["checkpoint_step"] == resumed_metrics["checkpoint_step"] == 1
    assert full_metrics["early_stopping"] == resumed_metrics["early_stopping"]
    assert [row["step"] for row in full_checkpoint["history"]] == [1, 2, 3, 4]
    assert resumed_checkpoint["history"] == full_checkpoint["history"]
    for name, tensor in full_checkpoint["model_state_dict"].items():
        assert torch.equal(tensor, resumed_checkpoint["model_state_dict"][name])
    for name, tensor in full_checkpoint["ema_model_state_dict"].items():
        assert torch.equal(tensor, resumed_checkpoint["ema_model_state_dict"][name])
    torch.testing.assert_close(
        resumed_checkpoint["optimizer_state_dict"],
        full_checkpoint["optimizer_state_dict"],
        rtol=0,
        atol=0,
    )


def test_final_best_checkpoint_resumes_from_terminal_continuation_without_duplicate_history(
    tmp_path,
) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {"name": "flow_matching"},
        "training": {
            "batch_size": 8,
            "steps": 8,
            "log_every": 1,
            "optimizer": "adam",
            "lr": 0.1,
            "warmup_steps": 2,
            "ema_decay": 0.5,
            "early_stopping": {
                "enabled": True,
                "warmup_steps": 0,
                "patience_steps": 3,
                "min_delta": 1.0e9,
                "ema_alpha": 1.0,
            },
        },
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }
    initial = TrainableConstantVelocity(dim=2, value=0.0).state_dict()

    full_model = TrainableConstantVelocity(dim=2, value=0.0)
    full_model.load_state_dict(initial)
    torch.manual_seed(41)
    train_flow_matching(
        config=config,
        run_dir=tmp_path / "full",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=full_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    first_model = TrainableConstantVelocity(dim=2, value=0.0)
    first_model.load_state_dict(initial)
    first_config = {**config, "training": {**config["training"], "steps": 2}}
    torch.manual_seed(41)
    train_flow_matching(
        config=first_config,
        run_dir=tmp_path / "first",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=first_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    first_checkpoint_path = tmp_path / "first" / "checkpoint.pt"
    first_checkpoint = load_checkpoint(first_checkpoint_path)
    assert first_checkpoint["step"] == 1
    assert first_checkpoint["metrics"]["trained_steps"] == 2
    assert first_checkpoint["continuation_state"]["step"] == 2
    assert first_checkpoint["continuation_state"]["version"] == 2
    assert set(first_checkpoint["continuation_state"]) == {
        "version",
        "step",
        "history",
        "rng_state_dict",
        "training_state",
    }
    assert first_checkpoint["resume_state"]["best_training_state"]["step"] == 1

    resumed_model = TrainableConstantVelocity(dim=2, value=99.0)
    resumed_config = {
        **config,
        "training": {
            **config["training"],
            "resume_from": str(first_checkpoint_path),
        },
    }
    train_flow_matching(
        config=resumed_config,
        run_dir=tmp_path / "resumed",
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=resumed_model,
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    full = load_checkpoint(tmp_path / "full" / "checkpoint.pt")
    resumed = load_checkpoint(tmp_path / "resumed" / "checkpoint.pt")
    assert [row["step"] for row in resumed["history"]] == [1, 2, 3, 4]
    assert resumed["history"] == full["history"]
    assert resumed["metrics"]["early_stopping"] == full["metrics"]["early_stopping"]
    assert resumed["metrics"]["checkpoint_step"] == full["metrics"]["checkpoint_step"]
    for key in ("model_state_dict", "ema_model_state_dict", "optimizer_state_dict"):
        torch.testing.assert_close(resumed[key], full[key], rtol=0, atol=0)

    malformed_states = {
        "missing_scheduler": ("theta_scheduler_state", None, True),
        "malformed_ema": ("ema_model_state", None, False),
        "unexpected_path": ("path_state", {}, False),
        "unexpected_path_optimizer": ("psi_optimizer_state", {}, False),
    }
    for name, (field, value, remove) in malformed_states.items():
        malformed_checkpoint = copy.deepcopy(first_checkpoint)
        training_state = malformed_checkpoint["continuation_state"]["training_state"]
        if remove:
            training_state.pop(field)
        else:
            training_state[field] = value
        malformed_path = tmp_path / f"malformed_{name}.pt"
        torch.save(malformed_checkpoint, malformed_path)
        malformed_config = copy.deepcopy(resumed_config)
        malformed_config["training"]["resume_from"] = str(malformed_path)

        with pytest.raises(ValueError, match="continuation_state training_state"):
            train_flow_matching(
                config=malformed_config,
                run_dir=tmp_path / f"malformed_{name}",
                target=ConstantTarget(),
                source=ConstantSource(),
                coupling=IndependentCoupling(),
                path=LinearPath(),
                model=RejectingStateLoadVelocity(),
                solvers=[EulerSolver()],
                device=torch.device("cpu"),
            )

    inconsistent_trackers = {}
    missing_best = copy.deepcopy(first_checkpoint)
    missing_best["resume_state"]["best_training_state"] = None
    inconsistent_trackers["missing_best"] = missing_best
    mismatched_best = copy.deepcopy(first_checkpoint)
    mismatched_best_state = mismatched_best["resume_state"]["best_training_state"]
    mismatched_best_state["step"] = 2
    mismatched_best_state["record"]["step"] = 2
    inconsistent_trackers["mismatched_best"] = mismatched_best
    unexpected_best = copy.deepcopy(first_checkpoint)
    unexpected_best["resume_state"]["early_stopping"]["best_step"] = None
    unexpected_best["resume_state"]["early_stopping"]["best_score"] = None
    unexpected_best["resume_state"]["early_stopping"]["best_loss"] = None
    inconsistent_trackers["unexpected_best"] = unexpected_best
    mismatched_top_level = copy.deepcopy(first_checkpoint)
    mismatched_top_level["step"] = 2
    inconsistent_trackers["mismatched_top_level"] = mismatched_top_level

    for name, malformed_checkpoint in inconsistent_trackers.items():
        malformed_path = tmp_path / f"inconsistent_{name}.pt"
        torch.save(malformed_checkpoint, malformed_path)
        malformed_config = copy.deepcopy(resumed_config)
        malformed_config["training"]["resume_from"] = str(malformed_path)

        with pytest.raises(ValueError, match="resume_state.*best_training_state"):
            train_flow_matching(
                config=malformed_config,
                run_dir=tmp_path / f"inconsistent_{name}",
                target=ConstantTarget(),
                source=ConstantSource(),
                coupling=IndependentCoupling(),
                path=LinearPath(),
                model=RejectingStateLoadVelocity(),
                solvers=[EulerSolver()],
                device=torch.device("cpu"),
            )

    orphaned_continuation = copy.deepcopy(first_checkpoint)
    orphaned_continuation["resume_state"]["best_training_state"] = None
    orphaned_early_state = orphaned_continuation["resume_state"]["early_stopping"]
    orphaned_early_state["best_step"] = None
    orphaned_early_state["best_score"] = None
    orphaned_early_state["best_loss"] = None
    orphaned_path = tmp_path / "orphaned_continuation.pt"
    torch.save(orphaned_continuation, orphaned_path)
    orphaned_config = copy.deepcopy(resumed_config)
    orphaned_config["training"]["resume_from"] = str(orphaned_path)
    with pytest.raises(ValueError, match="continuation_state.*best_training_state"):
        train_flow_matching(
            config=orphaned_config,
            run_dir=tmp_path / "orphaned_continuation",
            target=ConstantTarget(),
            source=ConstantSource(),
            coupling=IndependentCoupling(),
            path=LinearPath(),
            model=RejectingStateLoadVelocity(),
            solvers=[EulerSolver()],
            device=torch.device("cpu"),
        )


def test_final_checkpoint_without_best_rollback_has_no_continuation_copy(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {"name": "flow_matching"},
        "training": {
            "batch_size": 4,
            "steps": 2,
            "log_every": 1,
            "checkpoint_every": 2,
            "optimizer": "adam",
            "lr": 1.0e-3,
            "early_stopping": {"enabled": False},
        },
        "sampling": {"n_samples": 4, "n_trajectories": 2, "nfe": 2},
        "solvers": {"schedule": "uniform"},
    }
    train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=LinearPath(),
        model=TrainableConstantVelocity(dim=2, value=0.0),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    periodic = load_checkpoint(tmp_path / "checkpoints" / "step_000002.pt")
    final = load_checkpoint(tmp_path / "checkpoint.pt")
    assert "continuation_state" not in periodic
    assert "continuation_state" not in final
    assert periodic["resume_state"]["best_training_state"] is None
    assert final["resume_state"]["best_training_state"] is None
    assert final["step"] == periodic["step"] == 2
    assert final["history"] == periodic["history"]
    assert final["resume_state"] == periodic["resume_state"]
    for key in ("model_state_dict", "optimizer_state_dict"):
        torch.testing.assert_close(final[key], periodic[key], rtol=0, atol=0)


def test_train_flow_matching_kernel_vstar_learned_acceleration_smoke(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {
            "name": "flow_matching",
            "straightness": {"weight": 0.1, "sample_size": 4},
            "interpolant_acceleration": {"weight": 0.001},
            "learned_interpolant": {
                "mode": "kernel_vstar",
                "estimator_size": 4,
                "query_size": 2,
                "bandwidth": 10.0,
            },
        },
        "training": {
            "batch_size": 8,
            "steps": 2,
            "log_every": 1,
            "lr": 1.0e-3,
            "learned_acceleration": {
                "warmup_steps": 0,
                "theta_steps": 1,
                "psi_steps": 1,
                "psi_lr": 1.0e-3,
            },
        },
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }

    train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=TwoMoons(noise=0.0),
        source=GaussianSource(dim=2),
        coupling=IndependentCoupling(),
        path=LearnedAccelerationPath(dim=2, hidden_dim=8, depth=1),
        model=TrainableTimeScaledVelocity(),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    checkpoint = load_checkpoint(tmp_path / "checkpoint.pt")
    history_text = (tmp_path / "diagnostics" / "training_history.csv").read_text()

    assert "path_state_dict" in checkpoint
    assert "kernel_vstar_straightness_loss" in history_text
    assert "kernel_vstar_effective_sample_size_mean" in history_text
    assert (tmp_path / "plots" / "training_loss.png").exists()
    assert (tmp_path / "samples" / "euler_nfe3.npy").exists()
    assert (tmp_path / "trajectories" / "euler_nfe3.npy").exists()


def test_train_diffusion_epsilon_skips_velocity_sampling(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "objective": {"name": "diffusion", "prediction_type": "epsilon"},
        "training": {"batch_size": 8, "steps": 1, "log_every": 1, "lr": 1.0e-3},
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=GaussianDiffusionPath(schedule="linear"),
        model=TrainableConstantVelocity(dim=2, value=0.5),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )

    assert metrics["sampling"]["skipped"] is True
    assert "epsilon" in metrics["sampling"]["reason"]
    assert (tmp_path / "checkpoint.pt").exists()
    assert not (tmp_path / "samples").exists()
    assert not (tmp_path / "trajectories").exists()


def test_train_diffusion_score_preserves_training_and_skips_sampling(tmp_path) -> None:
    config = {
        "experiment": {"seed": 0},
        "path": {"name": "gaussian_diffusion"},
        "objective": {"name": "diffusion", "prediction_type": "score"},
        "training": {"batch_size": 8, "steps": 1, "log_every": 1, "lr": 1.0e-3},
        "sampling": {"n_samples": 8, "n_trajectories": 4, "nfe": 3},
        "solvers": {"schedule": "uniform"},
    }

    metrics = train_flow_matching(
        config=config,
        run_dir=tmp_path,
        target=ConstantTarget(),
        source=ConstantSource(),
        coupling=IndependentCoupling(),
        path=GaussianDiffusionPath(schedule="linear"),
        model=TrainableConstantVelocity(dim=2, value=0.5),
        solvers=[EulerSolver()],
        device=torch.device("cpu"),
    )
    checkpoint = load_checkpoint(tmp_path / "checkpoint.pt")

    assert metrics["sampling"]["skipped"] is True
    assert "score" in metrics["sampling"]["reason"]
    assert checkpoint["prediction_contract"]["model_output"] == "score"
    assert checkpoint["prediction_contract"]["loss_space"] == "score"


class TrainableTimeScaledVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.scale * t[:, None] * x


class TrainableConstantVelocity(nn.Module):
    def __init__(self, dim: int, value: float) -> None:
        super().__init__()
        self.velocity = nn.Parameter(torch.full((dim,), value))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return self.velocity.expand_as(x)


def _sampling_config(seed: int) -> dict:
    return {
        "experiment": {"seed": 0},
        "sampling": {
            "n_samples": 8,
            "n_trajectories": 5,
            "nfe": 3,
            "seed": seed,
        },
        "solvers": {"schedule": "uniform"},
    }
