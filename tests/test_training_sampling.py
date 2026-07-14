import numpy as np
import pytest
import torch
from torch import nn

import fm_lab.training.trainer as trainer_module
from fm_lab.couplings import IndependentCoupling, MinibatchOTCoupling
from fm_lab.data import GaussianMixture3D, TwoMoons
from fm_lab.paths import GaussianDiffusionPath, LearnedAccelerationPath, LinearPath, SphericalPath
from fm_lab.solvers import EulerSolver, HeunSolver
from fm_lab.sources import GaussianSource
from fm_lab.training.losses import build_objective
from fm_lab.training.prediction import velocity_model_for_objective
from fm_lab.training.sampling_guidance import (
    DensityGuidanceConfig,
    DensityGuidedDiffusionVelocity,
    apply_density_prior_rescaling,
)
from fm_lab.training.trainer import (
    _validate_training_compatibility,
    sample_and_plot,
    train_flow_matching,
    validate_checkpoint_compatibility,
)
from fm_lab.utils.checkpoints import load_checkpoint, save_checkpoint


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


class TinyVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        del context
        return self.linear(torch.cat((x, t[:, None]), dim=1))


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
        "path": {"name": "linear", "contract_tag": path_tag},
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
        "training": {"steps": 10, "batch_size": 4},
        "sampling": {"n_samples": 20, "nfe": 2},
        "experiment": {"output_dir": "runs/a"},
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

    assert contract["payload"]["objective"]["loss"] == "mse"
    assert contract["payload"]["objective"]["min_denom"] == 0.001
    assert contract["payload"]["objective"]["modifiers"] == []
    assert contract["payload"]["path"] == {
        "config": {"contract_tag": "a", "name": "linear"},
        "metadata": {"name": "linear"},
    }
    assert contract["payload"]["data"]["class_counts"] == [100, 10]
    assert "root" not in contract["payload"]["data"]
    assert "download" not in contract["payload"]["data"]


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
        _resume_contract_config(
            modifiers=[
                {"name": "oc", "transfer_mode": "t2h"},
                {"name": "cm", "consistency_weight": 1.0, "diversity_weight": 0.2},
            ]
        ),
        _resume_contract_config(min_denom=0.01),
        _resume_contract_config(path_tag="b"),
        _resume_contract_config(imbalance_factor=0.02),
    ],
    ids=["cbdm", "oc", "cm", "min-denom", "path", "data"],
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
        "steps": 100,
        "batch_size": 4,
        "resume_from": "checkpoint.pt",
    }
    active_config["sampling"] = {"n_samples": 10_000, "nfe": 64, "plot_max_points": 100}
    active_config["experiment"] = {"output_dir": "runs/resumed"}

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
    assert checkpoint["step"] == 1
    assert checkpoint["metrics"]["checkpoint_step"] == 1
    model_velocity = checkpoint["model_state_dict"]["velocity"]
    ema_velocity = checkpoint["ema_model_state_dict"]["velocity"]
    assert torch.equal(model_velocity, torch.zeros_like(model_velocity))
    assert torch.equal(ema_velocity, model_velocity)


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
