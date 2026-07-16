import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from fm_lab.diagnostics.long_tail_geometry.observation0 import (
    analyze_observation0_study,
    collect_observation0_run,
    prepare_observation0_study,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.diagnostics.long_tail_geometry.registry import update_observation0_run
from fm_lab.experiments.factory import build_model, build_source
from fm_lab.utils.checkpoints import save_checkpoint
from fm_lab.utils.config import load_config, save_config

CANONICAL_PATH = Path(
    "configs/fashion_mnist_lt/long_tail_geometry_observation0_preregistration.yaml"
)
CIFAR_CANONICAL_PATH = Path(
    "configs/cifar10_lt/long_tail_geometry_observation0_preregistration.yaml"
)


class _DiagnosticTarget:
    class_counts = (4, 4)

    def __init__(self) -> None:
        self._ids = {
            "a": np.asarray([100, 101, 110, 111], dtype=np.int64),
            "b": np.asarray([200, 201, 210, 211], dtype=np.int64),
        }

    def diagnostic_indices(self, split: str) -> np.ndarray:
        return self._ids[split].copy()

    def diagnostic_samples(
        self,
        split: str,
        *,
        original_indices=None,
        dequantization_seeds=None,
        device=None,
    ):
        ids = self._ids[split] if original_indices is None else np.asarray(original_indices)
        labels = torch.tensor([(int(value) // 10) % 10 for value in ids])
        rows = []
        seeds = (
            np.zeros(len(ids), dtype=np.int64)
            if dequantization_seeds is None
            else np.asarray(dequantization_seeds)
        )
        for original_id, seed in zip(ids, seeds, strict=True):
            generator = torch.Generator().manual_seed(int(seed % (2**31)) + int(original_id))
            rows.append(torch.randn(4, generator=generator) + float(labels[len(rows)]))
        samples = torch.stack(rows)
        if device is not None:
            samples = samples.to(device)
            labels = labels.to(device)
        return samples, labels, ids.astype(str)


def _tiny_base_config(path: Path) -> dict:
    config = {
        "experiment": {"name": "observation0_test", "seed": 0, "output_dir": "unused"},
        "data": {
            "name": "fashion_mnist_lt",
            "root": "unused",
            "train": True,
            "download": False,
            "dequantize": True,
            "frequency_mapping": {
                "offset": 0,
                "multiplier": 3,
                "diagnostic_pool_per_class": 4,
            },
        },
        "source": {"name": "gaussian", "dim": 4},
        "coupling": {"name": "independent"},
        "path": {"name": "linear"},
        "model": {
            "name": "mlp",
            "hidden_dim": 8,
            "depth": 2,
            "capacity": {"enabled": False},
        },
        "conditioning": {"enabled": True, "num_classes": 2, "embedding_dim": 4},
        "objective": {
            "name": "flow_matching",
            "model_output": "target",
            "loss_space": "velocity",
            "min_denom": 0.05,
            "modifiers": [],
        },
        "training": {
            "steps": 1,
            "batch_size": 2,
            "checkpoint_steps": [0, 1],
            "early_stopping": {"enabled": False},
            "ema": {"enabled": False},
        },
    }
    save_config(config, path)
    return config


def _tiny_preregistration(
    tmp_path: Path,
    *,
    training_seeds: tuple[int, ...] = (0,),
) -> Observation0Preregistration:
    base_path = tmp_path / "base.yaml"
    _tiny_base_config(base_path)
    canonical = Observation0Preregistration.load(CANONICAL_PATH)
    return dataclasses.replace(
        canonical,
        base_config=str(base_path),
        training_seeds=training_seeds,
        checkpoint_steps=(0, 1),
        manifest_seed=17,
        microbatch_size=1,
        primary_microbatches_per_cell=2,
        escalation_microbatches_per_cell=4,
        time_strata=((0.2, 0.8),),
        layers=("net.0.weight", "net.4.weight"),
        sketch_dim=8,
        max_sketch_dim=8,
        sketch_seed=19,
        gate_ranks=(1,),
        descriptive_ranks=(1,),
        null_permutations=99,
        required_seed_repeats=min(2, len(training_seeds)),
        minimum_common_classes=2,
    )


def test_prepare_writes_three_ordinary_fm_seed_configs(tmp_path: Path) -> None:
    result = prepare_observation0_study(CANONICAL_PATH, tmp_path)

    assert len(result.run_configs) == 3
    for seed, config_path in zip((0, 1, 2), result.run_configs, strict=True):
        config = load_config(config_path)
        assert config["experiment"]["seed"] == seed
        assert config["data"]["frequency_mapping"]["offset"] == 0
        assert config["objective"]["modifiers"] == []
        assert not config["model"]["capacity"]["enabled"]
        assert config["training"]["checkpoint_steps"] == [
            0,
            500,
            1000,
            3000,
            7000,
            13000,
            20000,
        ]
        assert not config["training"]["early_stopping"]["enabled"]
        assert config["diagnostics"]["long_tail_geometry"][
            "observation0_preregistration_sha256"
        ] == result.preregistration.digest
    registry = pd.read_csv(tmp_path / "aggregate/run_registry.csv")
    assert list(registry["seed"]) == [0, 1, 2]
    assert set(registry["mapping_offset"]) == {0}


def test_prepare_writes_three_cifar_natural_image_seed_configs(
    tmp_path: Path,
) -> None:
    result = prepare_observation0_study(CIFAR_CANONICAL_PATH, tmp_path)

    assert len(result.run_configs) == 3
    for seed, config_path in zip((0, 1, 2), result.run_configs, strict=True):
        config = load_config(config_path)
        assert config["experiment"]["seed"] == seed
        assert config["data"]["name"] == "cifar10_lt"
        assert config["data"]["dequantize"] is True
        assert config["data"]["frequency_mapping"] == {
            "offset": 0,
            "multiplier": 3,
            "diagnostic_pool_per_class": 256,
        }
        assert config["source"]["dim"] == 3072
        assert config["model"]["image_shape"] == [3, 32, 32]
        assert config["model"]["base_channels"] == 32
        assert config["model"]["capacity"]["enabled"] is False
        assert config["objective"]["modifiers"] == []
        assert config["training"]["steps"] == 100_000
        assert config["training"]["batch_size"] == 64
        assert config["training"]["checkpoint_steps"] == [
            0,
            2500,
            10_000,
            25_000,
            50_000,
            100_000,
        ]
        assert config["training"]["early_stopping"]["enabled"] is False
        assert config["training"]["ema"]["enabled"] is False
        assert config["training"]["ema_decay"] is None


def test_prepare_rejects_nonordinary_base_config(tmp_path: Path) -> None:
    preregistration = _tiny_preregistration(tmp_path)
    config = load_config(preregistration.base_config)
    config["objective"]["straightness"] = {"weight": 0.1}
    save_config(config, preregistration.base_config)
    path = preregistration.lock(tmp_path / "preregistration.yaml")

    with pytest.raises(ValueError, match="ordinary flow matching"):
        prepare_observation0_study(path, tmp_path / "study")


def test_collect_is_resumable_and_updates_registry_only_after_all_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = _tiny_preregistration(tmp_path)
    prereg_path = preregistration.lock(tmp_path / "preregistration.yaml")
    study_dir = tmp_path / "study"
    prepared = prepare_observation0_study(prereg_path, study_dir)
    run_dir = prepared.run_dirs[0]
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    config = load_config(prepared.run_configs[0])
    source = build_source(config)
    model = build_model(config, dim=source.dim)
    prediction_contract = {
        "path": "linear",
        "objective": "flow_matching",
        "model_output": "target",
        "loss_space": "velocity",
    }
    for step in (0, 1):
        save_checkpoint(
            checkpoint_dir / f"step_{step:06d}.pt",
            model=model,
            optimizer=None,
            step=step,
            config=config,
            prediction_contract=prediction_contract,
            metrics={"loss": 1.0},
        )
    monkeypatch.setattr(
        "fm_lab.diagnostics.long_tail_geometry.observation0.build_target",
        lambda config: _DiagnosticTarget(),
    )

    summary = collect_observation0_run(
        preregistration=preregistration,
        study_dir=study_dir,
        run_dir=run_dir,
        device=torch.device("cpu"),
    )

    assert summary.completed_steps == (0, 1)
    assert summary.skipped_steps == ()
    registry = pd.read_csv(study_dir / "aggregate/run_registry.csv")
    assert registry.iloc[0]["status"] == "measured"
    assert (study_dir / "aggregate/manifests/primary/probe_a.npz").exists()
    assert (study_dir / "aggregate/manifests/primary/probe_b.npz").exists()

    repeated = collect_observation0_run(
        preregistration=preregistration,
        study_dir=study_dir,
        run_dir=run_dir,
        device=torch.device("cpu"),
    )
    assert repeated.completed_steps == ()
    assert repeated.skipped_steps == (0, 1)

    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    save_checkpoint(
        checkpoint_dir / "step_000001.pt",
        model=model,
        optimizer=None,
        step=1,
        config=config,
        prediction_contract=prediction_contract,
        metrics={"loss": 0.5},
    )
    with pytest.raises(ValueError, match="changed checkpoint"):
        collect_observation0_run(
            preregistration=preregistration,
            study_dir=study_dir,
            run_dir=run_dir,
            device=torch.device("cpu"),
        )


def test_collect_does_not_mark_partial_run_measured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = _tiny_preregistration(tmp_path)
    prereg_path = preregistration.lock(tmp_path / "preregistration.yaml")
    study_dir = tmp_path / "study"
    prepared = prepare_observation0_study(prereg_path, study_dir)
    run_dir = prepared.run_dirs[0]
    (run_dir / "checkpoints").mkdir(parents=True)
    monkeypatch.setattr(
        "fm_lab.diagnostics.long_tail_geometry.observation0.build_target",
        lambda config: _DiagnosticTarget(),
    )

    with pytest.raises(ValueError, match="missing checkpoint"):
        collect_observation0_run(
            preregistration=preregistration,
            study_dir=study_dir,
            run_dir=run_dir,
            device=torch.device("cpu"),
        )

    registry = pd.read_csv(study_dir / "aggregate/run_registry.csv")
    assert registry.iloc[0]["status"] == "planned"


def test_analyze_refuses_registry_without_all_three_measured_seeds(
    tmp_path: Path,
) -> None:
    prepared = prepare_observation0_study(CANONICAL_PATH, tmp_path)
    update_observation0_run(
        tmp_path,
        seed=0,
        status="measured",
        run_dir=prepared.run_dirs[0],
        measurement_digest="a" * 64,
    )

    with pytest.raises(ValueError, match="all preregistered training seeds"):
        analyze_observation0_study(
            preregistration=prepared.preregistration,
            study_dir=tmp_path,
        )


def test_tiny_three_seed_study_runs_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preregistration = _tiny_preregistration(
        tmp_path,
        training_seeds=(0, 1, 2),
    )
    prereg_path = preregistration.lock(tmp_path / "preregistration.yaml")
    study_dir = tmp_path / "study"
    prepared = prepare_observation0_study(prereg_path, study_dir)
    monkeypatch.setattr(
        "fm_lab.diagnostics.long_tail_geometry.observation0.build_target",
        lambda config: _DiagnosticTarget(),
    )
    prediction_contract = {
        "path": "linear",
        "objective": "flow_matching",
        "model_output": "target",
        "loss_space": "velocity",
    }
    for seed, (config_path, run_dir) in enumerate(
        zip(prepared.run_configs, prepared.run_dirs, strict=True)
    ):
        config = load_config(config_path)
        source = build_source(config)
        torch.manual_seed(seed)
        model = build_model(config, dim=source.dim)
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        for step in preregistration.checkpoint_steps:
            save_checkpoint(
                checkpoint_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=None,
                step=step,
                config=config,
                prediction_contract=prediction_contract,
                metrics={"loss": 1.0},
            )
        summary = collect_observation0_run(
            preregistration=preregistration,
            study_dir=study_dir,
            run_dir=run_dir,
            device=torch.device("cpu"),
        )
        assert summary.completed_steps == (0, 1)

    decision = analyze_observation0_study(
        preregistration=preregistration,
        study_dir=study_dir,
    )

    assert decision.status in {
        "network_wide_measurable",
        "output_layer_only",
        "escalate_probe_rows",
    }
    registry = pd.read_csv(study_dir / "aggregate/run_registry.csv")
    assert set(registry["mapping_offset"]) == {0}
    assert set(registry["status"]) == {"measured"}
    measurement_dirs = list(
        study_dir.glob(
            "mapping_0/seed_*/diagnostics/long_tail_geometry/observation0/primary/"
            "checkpoint_*"
        )
    )
    assert len(measurement_dirs) == 6
    assert len(list((study_dir / "aggregate/manifests/primary").glob("probe_*.npz"))) == 2
    assert (study_dir / "aggregate/reliability.csv").exists()
    assert (study_dir / "aggregate/noise_ceiling.json").exists()


def test_observation0_cli_exposes_calibration_and_audit_but_not_stage1() -> None:
    from fm_lab.experiments.run_long_tail_geometry_observation0 import parse_args

    prepare = parse_args(
        [
            "prepare",
            "--preregistration",
            "protocol.yaml",
            "--study-dir",
            "study",
        ]
    )
    collect = parse_args(
        [
            "collect",
            "--study-dir",
            "study",
            "--run-dir",
            "run",
            "--device",
            "cpu",
            "--escalated",
        ]
    )
    analyze = parse_args(["analyze", "--study-dir", "study"])
    calibrate = parse_args(
        [
            "calibrate",
            "--study-dir",
            "study",
            "--calibration-preregistration",
            "functional.yaml",
            "--device",
            "cpu",
        ]
    )
    audit = parse_args(
        [
            "audit-functional-geometry",
            "--study-dir",
            "study",
            "--audit-preregistration",
            "audit.yaml",
            "--device",
            "cpu",
        ]
    )
    falsify = parse_args(
        [
            "falsify-natural-image-transport",
            "--study-dir",
            "study",
            "--transport-preregistration",
            "transport.yaml",
            "--device",
            "cpu",
        ]
    )

    assert prepare.command == "prepare"
    assert collect.command == "collect"
    assert collect.escalated is True
    assert analyze.command == "analyze"
    assert calibrate.command == "calibrate"
    assert calibrate.calibration_preregistration == "functional.yaml"
    assert calibrate.device == "cpu"
    assert audit.command == "audit-functional-geometry"
    assert audit.audit_preregistration == "audit.yaml"
    assert audit.device == "cpu"
    assert falsify.command == "falsify-natural-image-transport"
    assert falsify.transport_preregistration == "transport.yaml"
    assert falsify.device == "cpu"
    with pytest.raises(SystemExit):
        parse_args(["stage1", "--study-dir", "study"])


def test_observation0_calibration_cli_prints_locked_handoff(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    import fm_lab.experiments.run_long_tail_geometry_observation0 as cli

    decision = SimpleNamespace(
        stage1_unlocked=True,
        positive_control_pass=True,
        selected_relative_steps={
            "down2_block.conv2.weight": 3e-4,
            "middle.conv2.weight": 1e-4,
        },
        next_action="stage1_unlocked_for_separate_preregistration",
    )
    monkeypatch.setattr(
        cli,
        "calibrate_observation0_functional_overlap",
        lambda **kwargs: SimpleNamespace(decision=decision),
    )
    monkeypatch.setattr(cli, "resolve_device", lambda value: torch.device("cpu"))

    cli.main(
        [
            "calibrate",
            "--study-dir",
            "study",
            "--calibration-preregistration",
            "functional.yaml",
            "--device",
            "cpu",
        ]
    )

    output = capsys.readouterr().out
    assert "Functional lock: stage1_unlocked" in output
    assert "Positive control: passed" in output
    assert "down2_block.conv2.weight=0.0003" in output
    assert "Only allowed next action: stage1_unlocked_for_separate_preregistration" in output


def test_observation0_audit_cli_prints_nonunlocking_handoff(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    import fm_lab.experiments.run_long_tail_geometry_observation0 as cli

    decision = SimpleNamespace(
        stage1_unlocked=False,
        probe_b_opened=False,
        status="normalized_representation_rescue",
        layer_summaries={
            "down2_block.conv2.weight": {
                "normalized_target_slope_median": 1.25,
                "raw_target_slope_median": -0.10,
            },
            "middle.conv2.weight": {
                "normalized_target_slope_median": 0.80,
                "raw_target_slope_median": -0.20,
            },
        },
        next_action="review_separate_small_local_step_preregistration",
    )
    monkeypatch.setattr(
        cli,
        "run_functional_geometry_audit",
        lambda **kwargs: SimpleNamespace(decision=decision),
    )
    monkeypatch.setattr(cli, "resolve_device", lambda value: torch.device("cpu"))

    cli.main(
        [
            "audit-functional-geometry",
            "--study-dir",
            "study",
            "--audit-preregistration",
            "audit.yaml",
            "--device",
            "cpu",
        ]
    )

    output = capsys.readouterr().out
    assert "Audit status: normalized_representation_rescue" in output
    assert "Original functional lock: stage1_blocked (unchanged)" in output
    assert "Probe B opened: no" in output
    assert "down2_block.conv2.weight: normalized_slope=1.25, raw_slope=-0.1" in output
    assert "middle.conv2.weight: normalized_slope=0.8, raw_slope=-0.2" in output
    assert (
        "Only allowed audit next action: "
        "review_separate_small_local_step_preregistration"
    ) in output


def test_natural_image_falsification_cli_prints_terminal_handoff(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    import fm_lab.experiments.run_long_tail_geometry_observation0 as cli

    decision = SimpleNamespace(
        stage1_unlocked=False,
        method_opened=False,
        status="natural_image_transport_confirmed",
        baseline_learned=True,
        baseline_loss_ratio=0.42,
        reliable_common_classes=(0, 1, 2, 3, 4, 5),
        layer_summaries={
            "down2_block.conv2.weight": {
                "normalized_target_slope_median": 1.25,
                "normalized_selectivity_slope_median": 0.75,
            },
            "middle.conv2.weight": {
                "normalized_target_slope_median": 0.80,
                "normalized_selectivity_slope_median": 0.30,
            },
        },
        next_action="develop_sign_transport_theory",
    )
    monkeypatch.setattr(
        cli,
        "run_natural_image_transport_falsification",
        lambda **kwargs: SimpleNamespace(decision=decision),
    )
    monkeypatch.setattr(cli, "resolve_device", lambda value: torch.device("cpu"))

    cli.main(
        [
            "falsify-natural-image-transport",
            "--study-dir",
            "study",
            "--transport-preregistration",
            "transport.yaml",
            "--device",
            "cpu",
        ]
    )

    output = capsys.readouterr().out
    assert "Natural-image falsification: natural_image_transport_confirmed" in output
    assert "Baseline learned: yes (final/step-zero loss ratio=0.42)" in output
    assert "Reliable common classes: 6" in output
    assert (
        "down2_block.conv2.weight: normalized_slope=1.25, selectivity=0.75"
        in output
    )
    assert "middle.conv2.weight: normalized_slope=0.8, selectivity=0.3" in output
    assert "Only allowed next action: develop_sign_transport_theory" in output
