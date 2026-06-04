"""Run directory and metadata helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from fm_lab.utils.config import save_config


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON file with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def create_run_dir(config: dict[str, Any], root: str | Path | None = None) -> Path:
    """Create a run directory and persist config plus metadata."""

    experiment = config.get("experiment", {})
    if root is None:
        root = experiment.get("output_dir", "runs/unnamed")

    run_dir = Path(root)
    run_dir.mkdir(parents=True, exist_ok=True)

    for name in ("samples", "trajectories", "diagnostics", "plots"):
        (run_dir / name).mkdir(exist_ok=True)

    save_config(config, run_dir / "config.yaml")
    write_json(build_metadata(), run_dir / "metadata.json")
    return run_dir


def build_metadata() -> dict[str, Any]:
    """Collect lightweight reproducibility metadata for an experiment run."""

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_metadata(),
        "packages": _package_versions(
            [
                "fm-lab",
                "matplotlib",
                "numpy",
                "pandas",
                "pyyaml",
                "scikit-learn",
                "scipy",
                "torch",
            ]
        ),
    }


def _git_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {}
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_short": ["git", "status", "--short"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError as exc:
            result[f"{key}_error"] = repr(exc)
            continue

        if completed.returncode == 0:
            result[key] = completed.stdout.strip()
        else:
            result[f"{key}_error"] = completed.stderr.strip()
    return result


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
