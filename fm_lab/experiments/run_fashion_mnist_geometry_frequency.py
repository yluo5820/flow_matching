"""Run the Fashion-MNIST geometry-by-frequency bridge stages."""

from __future__ import annotations

import argparse
import json

from fm_lab.experiments.fashion_mnist_frequency_response import (
    analyze_frequency_response,
    evaluate_calibration_gate,
    load_stage1_config,
    prepare_stage1,
    run_all_frequency_conditions,
    run_calibration,
    run_frequency_condition,
)
from fm_lab.experiments.fashion_mnist_geometry_frequency import (
    load_stage0_config,
    run_stage0,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    stage0 = subparsers.add_parser("stage0")
    stage0.add_argument("--device", default="auto")
    stage0.add_argument("--dry-run", action="store_true")
    stage1_plan = subparsers.add_parser("stage1-plan")
    stage1_plan.add_argument("--dry-run", action="store_true")
    stage1_calibrate = subparsers.add_parser("stage1-calibrate")
    stage1_calibrate.add_argument("--training-steps", type=int, required=True)
    stage1_calibrate.add_argument("--device", default="auto")
    stage1_calibrate.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("stage1-calibration-status")
    stage1_run = subparsers.add_parser("stage1-run")
    stage1_run.add_argument("--condition", required=True)
    stage1_run.add_argument("--device", default="auto")
    stage1_run.add_argument("--dry-run", action="store_true")
    stage1_all = subparsers.add_parser("stage1-run-all")
    stage1_all.add_argument("--device", default="auto")
    stage1_all.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("stage1-analyze")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "stage0":
        config = load_stage0_config(args.config)
        result = run_stage0(config, device=args.device, dry_run=args.dry_run)
    else:
        config = load_stage1_config(args.config)
        if args.stage == "stage1-plan":
            result = prepare_stage1(config, dry_run=args.dry_run)
        elif args.stage == "stage1-calibrate":
            result = run_calibration(
                config,
                training_steps=args.training_steps,
                device=args.device,
                dry_run=args.dry_run,
            )
        elif args.stage == "stage1-calibration-status":
            prepare_stage1(config)
            result = evaluate_calibration_gate(config, write=True)
        elif args.stage == "stage1-run":
            result = run_frequency_condition(
                config,
                condition_id=args.condition,
                device=args.device,
                dry_run=args.dry_run,
            )
        elif args.stage == "stage1-run-all":
            result = run_all_frequency_conditions(
                config,
                device=args.device,
                dry_run=args.dry_run,
            )
        elif args.stage == "stage1-analyze":
            result = analyze_frequency_response(config)
        else:
            raise ValueError(f"Unsupported stage: {args.stage}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
