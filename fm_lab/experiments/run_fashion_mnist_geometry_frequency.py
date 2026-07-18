"""Run the Fashion-MNIST geometry-by-frequency bridge stages."""

from __future__ import annotations

import argparse
import json

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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_stage0_config(args.config)
    if args.stage != "stage0":
        raise ValueError(f"Unsupported stage: {args.stage}")
    result = run_stage0(config, device=args.device, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

