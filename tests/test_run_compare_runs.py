from pathlib import Path

import numpy as np
import pytest

from fm_lab.experiments.run_compare_runs import run_compare_runs
from fm_lab.utils.config import ConfigError, save_config


def test_compare_runs_writes_sample_and_loss_plots(tmp_path: Path) -> None:
    run_a = _write_fake_run(tmp_path / "run_a", name="baseline", offset=0.0)
    run_b = _write_fake_run(tmp_path / "run_b", name="straight", offset=1.0)

    summary = run_compare_runs(
        run_dirs=[run_a, run_b],
        labels=["baseline", "straight"],
        output_dir=tmp_path / "comparison",
        nfe=64,
        solver="rk4",
        loss_key="loss",
        max_points=16,
        unique_output=False,
    )

    output_dir = Path(summary["output_dir"])
    sample_plot = output_dir / "plots" / "generated_samples_nfe64.png"
    loss_plot = output_dir / "plots" / "training_loss_comparison.png"
    summary_path = output_dir / "summary.json"

    assert sample_plot.exists()
    assert sample_plot.stat().st_size > 0
    assert loss_plot.exists()
    assert loss_plot.stat().st_size > 0
    assert summary_path.exists()


def test_compare_runs_rejects_source_target_mismatch(tmp_path: Path) -> None:
    run_a = _write_fake_run(tmp_path / "run_a", name="baseline", offset=0.0)
    run_b = _write_fake_run(
        tmp_path / "run_b",
        name="other_target",
        offset=1.0,
        data={"name": "swiss_roll", "noise": 0.05},
    )

    with pytest.raises(ConfigError, match="source/target mismatch"):
        run_compare_runs(
            run_dirs=[run_a, run_b],
            output_dir=tmp_path / "comparison",
            unique_output=False,
        )


def test_compare_runs_auto_selects_single_sample_file(tmp_path: Path) -> None:
    run_a = _write_fake_run(tmp_path / "run_a", name="baseline", offset=0.0)
    run_b = _write_fake_run(tmp_path / "run_b", name="straight", offset=1.0)

    summary = run_compare_runs(
        run_dirs=[run_a, run_b],
        output_dir=tmp_path / "comparison",
        solver="auto",
        unique_output=False,
    )

    assert summary["runs"][0]["sample_path"].endswith("rk4_nfe64.npy")
    assert summary["runs"][1]["sample_path"].endswith("rk4_nfe64.npy")


def _write_fake_run(
    run_dir: Path,
    *,
    name: str,
    offset: float,
    data: dict | None = None,
) -> Path:
    samples_dir = run_dir / "samples"
    diagnostics_dir = run_dir / "diagnostics"
    samples_dir.mkdir(parents=True)
    diagnostics_dir.mkdir(parents=True)

    config = {
        "experiment": {"name": name, "seed": 0, "output_dir": str(run_dir)},
        "data": data or {"name": "two_moons", "noise": 0.05},
        "source": {"name": "gaussian", "dim": 2, "std": 1.0},
    }
    save_config(config, run_dir / "config.yaml")

    rng = np.random.default_rng(0)
    np.save(samples_dir / "target_reference.npy", rng.normal(size=(32, 2)))
    np.save(samples_dir / "rk4_nfe64.npy", rng.normal(size=(32, 2)) + offset)
    (diagnostics_dir / "training_history.csv").write_text(
        "step,loss,flow_matching_loss\n"
        f"1,{1.0 + offset},1.0\n"
        f"10,{0.5 + offset},0.5\n",
        encoding="utf-8",
    )
    return run_dir
