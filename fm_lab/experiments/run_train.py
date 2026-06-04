"""Training entry point.

Phase 1 wires config loading, seeding, and run-directory creation. The actual
flow matching training loop is added in the next implementation stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fm_lab.utils.config import load_config
from fm_lab.utils.logging import create_run_dir
from fm_lab.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an fm_lab training experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional run directory override. Defaults to experiment.output_dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the run directory and metadata without launching training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    experiment = config.setdefault("experiment", {})
    seed = int(experiment.get("seed", 0))
    seed_everything(seed)

    run_dir = create_run_dir(config, root=args.output_dir)
    if args.dry_run:
        print(f"Created dry-run directory: {Path(run_dir)}")
        return

    raise NotImplementedError("Training loop is scheduled for Phase 2.")


if __name__ == "__main__":
    main()
