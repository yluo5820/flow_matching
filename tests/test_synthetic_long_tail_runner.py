from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fm_lab.experiments.synthetic_long_tail_geometry import (
    RunLedger,
    StageBlockedError,
    build_matrix_commands,
    require_gate,
)


def test_matrix_dry_run_lists_exactly_36_training_commands(tmp_path: Path) -> None:
    config_paths = {
        replicate: tuple(
            tmp_path / f"rep{replicate:02d}" / f"condition_{condition:02d}.yaml"
            for condition in range(12)
        )
        for replicate in range(3)
    }

    commands = build_matrix_commands(config_paths, run_root=tmp_path / "runs")

    assert len(commands) == 36
    assert len({command.condition_id for command in commands}) == 12
    assert len({command.replicate for command in commands}) == 3
    assert all(command.argv("cpu")[0] == sys.executable for command in commands)
    assert len({command.run_dir for command in commands}) == 36


def test_failed_gate_blocks_training(tmp_path: Path) -> None:
    gate_path = tmp_path / "renderer_gate.json"
    gate_path.write_text(
        json.dumps({"passed": False, "reasons": ["renderer rank"]}), encoding="utf-8"
    )

    with pytest.raises(StageBlockedError, match="renderer rank"):
        require_gate(gate_path, stage="renderer_calibration")


def test_gate_requires_literal_boolean_pass(tmp_path: Path) -> None:
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({"passed": 1}), encoding="utf-8")

    with pytest.raises(StageBlockedError, match="literal true"):
        require_gate(gate_path, stage="oracle")


def test_completed_ledger_entry_is_not_overwritten(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "run_ledger.json")
    ledger.complete("rep00_g0_f0", {"metrics": "first.json"})

    with pytest.raises(FileExistsError, match="rep00_g0_f0"):
        ledger.complete("rep00_g0_f0", {"metrics": "second.json"})

    payload = json.loads((tmp_path / "run_ledger.json").read_text(encoding="utf-8"))
    assert payload["entries"][0]["artifacts"] == {"metrics": "first.json"}


def test_ledger_resume_requires_matching_config_hash(tmp_path: Path) -> None:
    ledger = RunLedger(tmp_path / "run_ledger.json")
    ledger.complete("rep00_g0_f0", {}, config_hash="abc")

    assert ledger.is_complete("rep00_g0_f0", config_hash="abc") is True
    assert ledger.is_complete("rep00_g0_f0", config_hash="different") is False
