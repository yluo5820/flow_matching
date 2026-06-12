from argparse import Namespace

from fm_lab.experiments.run_train import _objective_overrides


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
