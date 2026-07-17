"""Command-line orchestration for the synthetic long-tail geometry study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fm_lab.experiments.synthetic_long_tail_geometry import SyntheticLongTailRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen experiment YAML.")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("plan")
    pools = subparsers.add_parser("build-pools")
    pools.add_argument("--replicate", type=int, required=True)
    subparsers.add_parser("calibrate-renderer")
    oracle = subparsers.add_parser("train-oracle")
    _device(oracle)
    pilot = subparsers.add_parser("pilot")
    _device(pilot)
    _dry_run(pilot)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--condition", required=True)
    smoke.add_argument("--replicate", type=int, required=True)
    _device(smoke)
    _dry_run(smoke)
    matrix = subparsers.add_parser("matrix")
    _device(matrix)
    _dry_run(matrix)
    matrix.add_argument("--resume", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    _device(evaluate)
    _dry_run(evaluate)
    evaluate.add_argument("--checkpoint-regimes", default="equal_update,matched_pass")
    evaluate.add_argument("--resume", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--bootstrap-draws", type=int, default=None)
    subparsers.add_parser("report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    runner = SyntheticLongTailRunner(args.config)
    result = _dispatch(runner, args)
    if result is not None:
        print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


def _dispatch(runner: SyntheticLongTailRunner, args: argparse.Namespace) -> Any:
    if args.stage == "plan":
        return runner.plan()
    if args.stage == "build-pools":
        return runner.build_pools(args.replicate)
    if args.stage == "calibrate-renderer":
        return runner.calibrate_renderer()
    if args.stage == "train-oracle":
        return runner.train_oracle(device=args.device)
    if args.stage == "pilot":
        return runner.pilot(device=args.device, dry_run=args.dry_run)
    if args.stage == "smoke":
        return runner.smoke(
            condition_id=args.condition,
            replicate=args.replicate,
            device=args.device,
            dry_run=args.dry_run,
        )
    if args.stage == "matrix":
        return runner.matrix(device=args.device, dry_run=args.dry_run, resume=args.resume)
    if args.stage in {"evaluate", "aggregate", "report"}:
        method = getattr(runner, args.stage, None)
        if method is None:
            raise NotImplementedError(f"{args.stage} is installed by the reporting stage.")
        kwargs = vars(args).copy()
        kwargs.pop("config")
        kwargs.pop("stage")
        return method(**kwargs)
    raise AssertionError(f"Unhandled stage: {args.stage}")


def _device(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto")


def _dry_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
