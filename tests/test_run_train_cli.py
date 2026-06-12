from argparse import Namespace

from fm_lab.experiments.run_train import _objective_overrides, _sampling_overrides


def test_objective_overrides_from_cli_args() -> None:
    args = Namespace(
        objective="flow_matching",
        objective_loss="mse",
        straightness_weight=0.01,
        straightness_sample_size=128,
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
        straightness_weight=0.0,
        straightness_sample_size=None,
    )

    assert _objective_overrides(args) == {"straightness": {"weight": 0.0}}


def test_sampling_overrides_from_cli_args() -> None:
    args = Namespace(
        n_samples=8192,
        n_trajectories=128,
        nfe=64,
        plot_max_points=8192,
        trajectory_target_max_points=3000,
    )

    assert _sampling_overrides(args) == {
        "n_samples": 8192,
        "n_trajectories": 128,
        "nfe": 64,
        "plot_max_points": 8192,
        "trajectory_target_max_points": 3000,
    }
