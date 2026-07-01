from argparse import Namespace
from types import SimpleNamespace

import fm_lab.experiments.run_sample_checkpoint as sample_checkpoint_cli
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
        diffusion_prediction_type=None,
        straightness_weight=0.01,
        straightness_sample_size=128,
        direction_weight=None,
        speed_weight=None,
    )

    assert _objective_overrides(args) == {
        "name": "flow_matching",
        "loss": "mse",
        "straightness": {"weight": 0.01, "sample_size": 128},
    }


def test_objective_overrides_allows_disabling_straightness() -> None:
    args = Namespace(
        objective=None,
        objective_loss=None,
        diffusion_prediction_type=None,
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
        diffusion_prediction_type=None,
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


def test_diffusion_prediction_type_override_from_cli_args() -> None:
    args = Namespace(
        objective="diffusion",
        objective_loss=None,
        diffusion_prediction_type="score",
        straightness_weight=None,
        straightness_sample_size=None,
        direction_weight=None,
        speed_weight=None,
    )

    assert _objective_overrides(args) == {
        "name": "diffusion",
        "prediction_type": "score",
    }


def test_training_overrides_from_cli_args() -> None:
    args = Namespace(
        steps=1234,
        batch_size=256,
    )

    assert _training_overrides(args) == {
        "steps": 1234,
        "batch_size": 256,
    }


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
            "config": {"sampling": {"nfe": 3}},
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
    monkeypatch.setattr(sample_checkpoint_cli, "build_solvers", lambda config: ["solver"])

    def fake_sample_and_plot(**kwargs):
        assert kwargs["model"] is model
        assert kwargs["model"].loaded_state == {"weight": 1.0}
        assert kwargs["model"].device == "mps"
        assert kwargs["device"] == "mps"
        return {"ok": True}

    monkeypatch.setattr(sample_checkpoint_cli, "sample_and_plot", fake_sample_and_plot)

    sample_checkpoint_cli.main()

    assert model.device == "mps"


class _FakeCheckpointModel:
    def __init__(self) -> None:
        self.loaded_state = None
        self.device = None

    def load_state_dict(self, state):
        self.loaded_state = state

    def to(self, device):
        self.device = device
        return self
