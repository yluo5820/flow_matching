from pathlib import Path

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
    assert (run_dir / "plots").is_dir()
