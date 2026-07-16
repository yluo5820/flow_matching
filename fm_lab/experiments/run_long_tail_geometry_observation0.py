"""CLI for the preregistered long-tail geometry Observation-0 pilot."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fm_lab.diagnostics.long_tail_geometry.functional_audit import (
    run_functional_geometry_audit,
)
from fm_lab.diagnostics.long_tail_geometry.functional_calibration import (
    calibrate_observation0_functional_overlap,
)
from fm_lab.diagnostics.long_tail_geometry.natural_image import (
    run_natural_image_transport_falsification,
)
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
        description=(
            "Prepare, collect, analyze, functionally calibrate, audit, or falsify "
            "long-tail geometry transport."
        )
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

    calibrate = commands.add_parser(
        "calibrate",
        help="Calibrate exact rank-1 directions using Probe-A only.",
    )
    calibrate.add_argument("--study-dir", required=True)
    calibrate.add_argument("--calibration-preregistration", required=True)
    calibrate.add_argument("--device", default="auto")

    audit = commands.add_parser(
        "audit-functional-geometry",
        help="Compare raw and row-normalized exact geometry using Probe-A only.",
    )
    audit.add_argument("--study-dir", required=True)
    audit.add_argument("--audit-preregistration", required=True)
    audit.add_argument("--device", default="auto")

    falsify = commands.add_parser(
        "falsify-natural-image-transport",
        help="Run the terminal CIFAR-10-LT geometry and transport decision.",
    )
    falsify.add_argument("--study-dir", required=True)
    falsify.add_argument("--transport-preregistration", required=True)
    falsify.add_argument("--device", default="auto")
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

    if args.command == "calibrate":
        result = calibrate_observation0_functional_overlap(
            study_dir=study_dir,
            calibration_preregistration_path=args.calibration_preregistration,
            device=resolve_device(args.device),
        )
        decision = result.decision
        status = "stage1_unlocked" if decision.stage1_unlocked else "stage1_blocked"
        print(f"Functional lock: {status}")
        control = "passed" if decision.positive_control_pass else "failed"
        print(f"Positive control: {control}")
        selected = ", ".join(
            f"{layer}={step:g}"
            for layer, step in sorted(decision.selected_relative_steps.items())
        )
        print(f"Selected relative steps: {selected}")
        print(f"Only allowed next action: {decision.next_action}")
        return

    if args.command == "audit-functional-geometry":
        result = run_functional_geometry_audit(
            study_dir=study_dir,
            audit_preregistration_path=args.audit_preregistration,
            device=resolve_device(args.device),
        )
        decision = result.decision
        print(f"Audit status: {decision.status}")
        print("Original functional lock: stage1_blocked (unchanged)")
        opened = "yes" if decision.probe_b_opened else "no"
        print(f"Probe B opened: {opened}")
        for layer, summary in sorted(decision.layer_summaries.items()):
            normalized = summary["normalized_target_slope_median"]
            raw = summary["raw_target_slope_median"]
            print(f"{layer}: normalized_slope={normalized:g}, raw_slope={raw:g}")
        print(f"Only allowed audit next action: {decision.next_action}")
        return

    if args.command == "falsify-natural-image-transport":
        result = run_natural_image_transport_falsification(
            study_dir=study_dir,
            preregistration_path=args.transport_preregistration,
            device=resolve_device(args.device),
        )
        decision = result.decision
        print(f"Natural-image falsification: {decision.status}")
        learned = "yes" if decision.baseline_learned else "no"
        print(
            f"Baseline learned: {learned} "
            f"(final/step-zero loss ratio={decision.baseline_loss_ratio:g})"
        )
        print(
            f"Reliable common classes: {len(decision.reliable_common_classes)}"
        )
        for layer, summary in sorted(decision.layer_summaries.items()):
            slope = summary["normalized_target_slope_median"]
            selectivity = summary["normalized_selectivity_slope_median"]
            print(
                f"{layer}: normalized_slope={slope:g}, "
                f"selectivity={selectivity:g}"
            )
        print(f"Only allowed next action: {decision.next_action}")
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
