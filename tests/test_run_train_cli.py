import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

import fm_lab.experiments.run_sample_checkpoint as sample_checkpoint_cli
import fm_lab.experiments.run_train as train_cli
from fm_lab.experiments.run_sample_checkpoint import (
    _sampling_overrides as _checkpoint_sampling_overrides,
)
from fm_lab.experiments.run_train import (
    _data_overrides,
    _objective_overrides,
    _sampling_overrides,
    _training_overrides,
)


def test_objective_overrides_from_cli_args() -> None:
    args = Namespace(
        objective="flow_matching",
        objective_loss="mse",
        model_output="target",
        loss_space="velocity",
        prediction_min_denom=0.01,
        straightness_weight=None,
        straightness_sample_size=None,
        direction_weight=None,
        speed_weight=None,
    )

    assert _objective_overrides(args) == {
        "name": "flow_matching",
        "loss": "mse",
        "model_output": "target",
        "loss_space": "velocity",
        "min_denom": 0.01,
    }


def test_objective_overrides_allows_disabling_straightness() -> None:
    args = Namespace(
        objective=None,
        objective_loss=None,
        model_output=None,
        loss_space=None,
        prediction_min_denom=None,
        straightness_weight=0.0,
        straightness_sample_size=None,
        direction_weight=None,
        speed_weight=None,
    )

    assert _objective_overrides(args) == {"straightness": {"weight": 0.0}}


def test_direction_only_weight_overrides_from_cli_args() -> None:
    args = Namespace(
        objective="direction_only_straight",
        objective_loss=None,
        model_output=None,
        loss_space=None,
        prediction_min_denom=None,
        straightness_weight=None,
        straightness_sample_size=None,
        direction_weight=10.0,
        speed_weight=1.0,
    )

    assert _objective_overrides(args) == {
        "name": "direction_only_straight",
        "direction_weight": 10.0,
        "speed_weight": 1.0,
    }


def test_training_parser_help_uses_only_canonical_prediction_flags(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr("sys.argv", ["run_train", "--help"])

    with pytest.raises(SystemExit, match="0"):
        train_cli.parse_args()

    help_text = capsys.readouterr().out
    assert "--model-output" in help_text
    assert "--loss-space" in help_text
    assert "--prediction-min-denom" in help_text
    assert "--diffusion-prediction-type" not in help_text
    assert "--x-prediction-loss-space" not in help_text
    assert "--x-prediction-min-denom" not in help_text


def test_training_overrides_from_cli_args() -> None:
    args = Namespace(
        steps=1234,
        batch_size=256,
        resume_from="runs/example/checkpoints/step_100000.pt",
    )

    assert _training_overrides(args) == {
        "steps": 1234,
        "batch_size": 256,
        "resume_from": "runs/example/checkpoints/step_100000.pt",
    }


def test_training_cli_validates_resume_contract_before_model_construction(
    tmp_path,
    monkeypatch,
) -> None:
    config = {
        "experiment": {"seed": 0, "output_dir": str(tmp_path / "run")},
        "training": {"resume_from": "checkpoint.pt"},
        "path": {"name": "linear"},
        "objective": {
            "name": "flow_matching",
            "model_output": "target",
            "loss_space": "velocity",
        },
    }
    monkeypatch.setattr(
        train_cli,
        "parse_args",
        lambda: SimpleNamespace(
            config="config.yaml",
            output_dir=None,
            dry_run=False,
            device="cpu",
        ),
    )
    monkeypatch.setattr(train_cli, "load_config", lambda path: config)
    monkeypatch.setattr(train_cli, "_training_overrides", lambda args: {})
    monkeypatch.setattr(train_cli, "_data_overrides", lambda args: {})
    monkeypatch.setattr(train_cli, "_sampling_overrides", lambda args: {})
    monkeypatch.setattr(train_cli, "_objective_overrides", lambda args: {})
    monkeypatch.setattr(train_cli, "create_run_dir", lambda *args, **kwargs: tmp_path / "run")
    monkeypatch.setattr("fm_lab.experiments.factory.build_target", lambda config: object())
    monkeypatch.setattr("fm_lab.experiments.factory.build_path", lambda config: object())
    monkeypatch.setattr("fm_lab.experiments.factory.resolve_device", lambda value: "cpu")
    monkeypatch.setattr(
        "fm_lab.experiments.factory.build_model",
        lambda *args, **kwargs: pytest.fail("model must not be built"),
    )
    monkeypatch.setattr(
        "fm_lab.training.trainer.validate_resume_checkpoint_before_model",
        lambda **kwargs: (_ for _ in ()).throw(
            ValueError("Checkpoint training contract is incompatible")
        ),
    )

    with pytest.raises(ValueError, match="training contract.*incompatible"):
        train_cli.main()


def test_data_overrides_include_dataset_variant_workspace() -> None:
    args = Namespace(
        dataset_variant="mnist/tail_digit1",
        workspace="outputs/geometry_explorer",
    )

    assert _data_overrides(args) == {
        "name": "mnist",
        "variant_id": "mnist/tail_digit1",
        "workspace": "outputs/geometry_explorer",
    }


def test_sampling_overrides_from_cli_args() -> None:
    args = Namespace(
        n_samples=8192,
        n_trajectories=128,
        nfe=64,
        plot_max_points=8192,
        sample_batch_size=512,
        trajectory_target_max_points=3000,
    )

    assert _sampling_overrides(args) == {
        "n_samples": 8192,
        "n_trajectories": 128,
        "nfe": 64,
        "plot_max_points": 8192,
        "sample_batch_size": 512,
        "trajectory_target_max_points": 3000,
    }


def test_checkpoint_sampling_overrides_include_trajectory_umap() -> None:
    args = Namespace(
        n_samples=4096,
        n_trajectories=256,
        nfe=96,
        plot_max_points=None,
        sample_batch_size=128,
        trajectory_target_max_points=None,
        trajectory_umap=True,
        no_trajectory_umap=False,
        trajectory_umap_target_points=5000,
        trajectory_umap_neighbors=45,
        trajectory_umap_min_dist=0.05,
    )

    assert _checkpoint_sampling_overrides(args) == {
        "n_samples": 4096,
        "n_trajectories": 256,
        "nfe": 96,
        "sample_batch_size": 128,
        "trajectory_umap": {
            "enabled": True,
            "max_target_points": 5000,
            "n_neighbors": 45,
            "min_dist": 0.05,
        },
    }


def test_checkpoint_sampling_overrides_include_density_guidance() -> None:
    args = Namespace(
        n_samples=None,
        n_trajectories=None,
        nfe=None,
        plot_max_points=None,
        sample_batch_size=None,
        trajectory_target_max_points=None,
        trajectory_umap=False,
        no_trajectory_umap=False,
        trajectory_umap_target_points=None,
        trajectory_umap_neighbors=None,
        trajectory_umap_min_dist=None,
        prior_guidance_scale=0.8,
        density_guidance_quantile=0.3,
        density_guidance_strength=0.5,
        density_guidance_t_min=1.0e-4,
        density_guidance_t_max=0.95,
        density_guidance_prior_quantile=0.5,
        no_density_guidance_prior_rescale=False,
    )

    assert _checkpoint_sampling_overrides(args) == {
        "guidance": {
            "prior": {"scale": 0.8},
            "density": {
                "quantile": 0.3,
                "strength": 0.5,
                "t_min": 1.0e-4,
                "t_max": 0.95,
                "prior_rescale_quantile": 0.5,
            },
        },
    }


def test_checkpoint_sampling_overrides_can_disable_density_prior_rescale() -> None:
    args = Namespace(
        n_samples=None,
        n_trajectories=None,
        nfe=None,
        plot_max_points=None,
        sample_batch_size=None,
        trajectory_target_max_points=None,
        trajectory_umap=False,
        no_trajectory_umap=False,
        trajectory_umap_target_points=None,
        trajectory_umap_neighbors=None,
        trajectory_umap_min_dist=None,
        no_density_guidance_prior_rescale=True,
    )

    assert _checkpoint_sampling_overrides(args) == {
        "guidance": {"density": {"prior_rescale_quantile": None}},
    }


def test_sample_checkpoint_moves_loaded_model_to_sampling_device(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
    model = _FakeCheckpointModel()

    monkeypatch.setattr(
        sample_checkpoint_cli,
        "parse_args",
        lambda: Namespace(
            run_dir=str(run_dir),
            checkpoint=None,
            output_dir=None,
            device="mps",
            n_samples=None,
            n_trajectories=None,
            nfe=None,
            plot_max_points=None,
            sample_batch_size=None,
            trajectory_target_max_points=None,
            trajectory_umap=False,
            no_trajectory_umap=False,
            trajectory_umap_target_points=None,
            trajectory_umap_neighbors=None,
            trajectory_umap_min_dist=None,
        ),
    )
    monkeypatch.setattr(sample_checkpoint_cli, "resolve_device", lambda value: value)
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "load_checkpoint",
        lambda path, map_location: {
            "prediction_contract": {
                "path": "linear",
                "objective": "flow_matching",
                "model_output": "velocity",
                "loss_space": "velocity",
            },
            "config": {
                "path": {"name": "linear"},
                "objective": {
                    "name": "flow_matching",
                    "model_output": "velocity",
                    "loss_space": "velocity",
                },
                "sampling": {"nfe": 3},
            },
            "model_state_dict": {"weight": 1.0},
        },
    )
    monkeypatch.setattr(sample_checkpoint_cli, "build_target", lambda config: object())
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "build_source",
        lambda config: SimpleNamespace(dim=7),
    )
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "build_model",
        lambda config, dim: model,
    )
    monkeypatch.setattr(sample_checkpoint_cli, "build_path", lambda config: object())
    monkeypatch.setattr(sample_checkpoint_cli, "build_solvers", lambda config: ["solver"])

    def fake_sample_and_plot(**kwargs):
        assert kwargs["model"] is model
        assert kwargs["path"] is not None
        assert kwargs["model"].loaded_state == {"weight": 1.0}
        assert kwargs["model"].device == "mps"
        assert kwargs["device"] == "mps"
        return {"ok": True}

    monkeypatch.setattr(sample_checkpoint_cli, "sample_and_plot", fake_sample_and_plot)

    sample_checkpoint_cli.main()

    assert model.device == "mps"


def test_sample_checkpoint_rejects_discrete_metadata_before_building_model(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "parse_args",
        lambda: Namespace(
            run_dir=str(run_dir),
            checkpoint=None,
            output_dir=None,
            device="cpu",
            register_only=False,
        ),
    )
    monkeypatch.setattr(sample_checkpoint_cli, "resolve_device", lambda value: value)
    monkeypatch.setattr(sample_checkpoint_cli, "_sampling_overrides", lambda args: {})
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "load_checkpoint",
        lambda path, map_location: {
            "prediction_contract": {
                "path": "linear",
                "objective": "discrete_diffusion",
                "model_output": "target",
                "loss_space": "velocity",
            },
            "config": {
                "path": {"name": "linear"},
                "objective": {
                    "name": "discrete_diffusion",
                    "model_output": "target",
                    "loss_space": "velocity",
                },
            },
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "build_model",
        lambda *args, **kwargs: pytest.fail("model must not be built"),
    )
    monkeypatch.setattr(sample_checkpoint_cli, "build_target", lambda config: object())
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "build_source",
        lambda config: SimpleNamespace(dim=2),
    )
    monkeypatch.setattr(sample_checkpoint_cli, "build_path", lambda config: object())

    with pytest.raises(ValueError, match="discrete checkpoints are incompatible"):
        sample_checkpoint_cli.main()


@pytest.mark.parametrize(
    "missing_field",
    ["path", "objective", "model_output", "loss_space"],
)
def test_sample_checkpoint_rejects_missing_prediction_contract_field_before_model_build(
    tmp_path,
    monkeypatch,
    missing_field: str,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
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
    prediction_contract.pop(missing_field)
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "parse_args",
        lambda: Namespace(
            run_dir=str(run_dir),
            checkpoint=None,
            output_dir=None,
            device="cpu",
            register_only=False,
        ),
    )
    monkeypatch.setattr(sample_checkpoint_cli, "resolve_device", lambda value: value)
    monkeypatch.setattr(sample_checkpoint_cli, "_sampling_overrides", lambda args: {})
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "load_checkpoint",
        lambda path, map_location: {
            "prediction_contract": prediction_contract,
            "config": config,
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "build_model",
        lambda *args, **kwargs: pytest.fail("model must not be built"),
    )

    with pytest.raises(ValueError, match=rf"prediction_contract.{missing_field}"):
        sample_checkpoint_cli.main()


@pytest.mark.parametrize(
    ("field", "checkpoint_value"),
    [
        ("path", "spherical"),
        ("objective", "diffusion"),
        ("model_output", "target"),
        ("loss_space", "target"),
    ],
)
def test_sample_checkpoint_rejects_prediction_contract_mismatch_before_model_build(
    tmp_path,
    monkeypatch,
    field: str,
    checkpoint_value: str,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoint.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
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
    prediction_contract[field] = checkpoint_value
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "parse_args",
        lambda: Namespace(
            run_dir=str(run_dir),
            checkpoint=None,
            output_dir=None,
            device="cpu",
            register_only=False,
        ),
    )
    monkeypatch.setattr(sample_checkpoint_cli, "resolve_device", lambda value: value)
    monkeypatch.setattr(sample_checkpoint_cli, "_sampling_overrides", lambda args: {})
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "load_checkpoint",
        lambda path, map_location: {
            "prediction_contract": prediction_contract,
            "config": config,
            "model_state_dict": {},
        },
    )
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "build_model",
        lambda *args, **kwargs: pytest.fail("model must not be built"),
    )

    with pytest.raises(ValueError, match=rf"{field}=.*prediction contract"):
        sample_checkpoint_cli.main()


def test_sample_checkpoint_register_only_uses_existing_sampling_payload(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    output_dir = tmp_path / "sampled"
    checkpoint_path = run_dir / "checkpoint.pt"
    summary_path = output_dir / "diagnostics" / "checkpoint_sampling.json"
    run_dir.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")
    (run_dir / "config.yaml").write_text(
        "data:\n  workspace: unused\n",
        encoding="utf-8",
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "output_dir": str(output_dir),
                "sampling": {
                    "nfe": 64,
                    "image_shape": [32, 32, 3],
                    "image_value_range": [-1.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    monkeypatch.setattr(
        sample_checkpoint_cli,
        "parse_args",
        lambda: Namespace(
            run_dir=str(run_dir),
            checkpoint=None,
            output_dir=str(output_dir),
            device="mps",
            register_only=True,
            register_dataset="cifar10/generated_density_pilot",
            dataset_workspace=str(tmp_path / "workspace"),
            dataset_label="density_q005_pilot",
            dataset_solver="euler",
            dataset_base="generated",
            dataset_split="generated",
            dataset_atlas_tile_size=32,
            dataset_atlas_size=2048,
        ),
    )
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "load_checkpoint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("register-only should not load checkpoint")
        ),
    )
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "build_target",
        lambda config: SimpleNamespace(metadata=lambda: {"name": "cifar10"}),
    )
    monkeypatch.setattr(
        sample_checkpoint_cli,
        "build_solvers",
        lambda config: [SimpleNamespace(name="euler")],
    )

    def fake_register_generated_dataset(**kwargs):
        calls.append(kwargs)
        return {
            "variant_id": kwargs["variant_id"],
            "dataset_path": str(tmp_path / "dataset.parquet"),
            "rows": 4096,
        }

    monkeypatch.setattr(
        sample_checkpoint_cli,
        "_register_generated_dataset",
        fake_register_generated_dataset,
    )

    sample_checkpoint_cli.main()

    assert len(calls) == 1
    assert calls[0]["variant_id"] == "cifar10/generated_density_pilot"
    assert calls[0]["output_dir"] == output_dir
    assert calls[0]["summary"]["nfe"] == 64
    updated = json.loads(summary_path.read_text(encoding="utf-8"))
    assert updated["registered_dataset"]["variant_id"] == "cifar10/generated_density_pilot"


class _FakeCheckpointModel:
    def __init__(self) -> None:
        self.loaded_state = None
        self.device = None

    def load_state_dict(self, state):
        self.loaded_state = state

    def to(self, device):
        self.device = device
        return self
