"""CLI for the preregistered long-tail geometry Observation-0 pilot."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fm_lab.diagnostics.long_tail_geometry.observation0 import (
    analyze_observation0_study,
    collect_observation0_run,
    prepare_observation0_study,
)
from fm_lab.diagnostics.long_tail_geometry.preregistration import (
    Observation0Preregistration,
)
from fm_lab.experiments.factory import resolve_device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, collect, or analyze long-tail geometry Observation 0."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Lock the pilot and write seed configs.")
    prepare.add_argument("--preregistration", required=True)
    prepare.add_argument("--study-dir", required=True)

    collect = commands.add_parser(
        "collect",
        help="Collect every preregistered checkpoint for one registered seed.",
    )
    collect.add_argument("--study-dir", required=True)
    collect.add_argument("--run-dir", required=True)
    collect.add_argument("--device", default="auto")
    collect.add_argument(
        "--escalated",
        action="store_true",
        help="Use the locked 32-microbatch escalation after a primary gate requests it.",
    )

    analyze = commands.add_parser(
        "analyze",
        help="Analyze only registry-listed completed measurement artifacts.",
    )
    analyze.add_argument("--study-dir", required=True)
    analyze.add_argument(
        "--escalated",
        action="store_true",
        help="Analyze the locked 32-microbatch escalation artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    study_dir = Path(args.study_dir)
    if args.command == "prepare":
        result = prepare_observation0_study(args.preregistration, study_dir)
        print(f"Prepared Observation 0 at {result.study_dir}")
        for config, run_dir in zip(result.run_configs, result.run_dirs, strict=True):
            print(f"  train {config} -> {run_dir}")
        return

    preregistration = Observation0Preregistration.load(
        study_dir / "aggregate" / "preregistration.yaml"
    )
    if args.command == "collect":
        summary = collect_observation0_run(
            preregistration=preregistration,
            study_dir=study_dir,
            run_dir=args.run_dir,
            device=resolve_device(args.device),
            escalated=args.escalated,
        )
        print(
            f"Collected seed {summary.seed} ({summary.phase}); "
            f"completed={list(summary.completed_steps)}, "
            f"skipped={list(summary.skipped_steps)}"
        )
        return

    decision = analyze_observation0_study(
        preregistration=preregistration,
        study_dir=study_dir,
        escalated=args.escalated,
    )
    print(f"Observation-0 status: {decision.status}")
    print(f"Only allowed next action: {decision.next_action}")


if __name__ == "__main__":
    main()
