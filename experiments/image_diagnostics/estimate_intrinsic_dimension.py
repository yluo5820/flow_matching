"""Estimate local and grouped intrinsic dimension in a feature space."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fm_lab.image_diagnostics.id_config import (  # noqa: E402
    apply_id_overrides,
    id_config_from_dict,
)
from fm_lab.image_diagnostics.id_runner import run_id_estimation  # noqa: E402
from fm_lab.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate local and grouped intrinsic dimension."
    )
    parser.add_argument("--config", required=True, help="ID estimation YAML config.")
    parser.add_argument("--diagnostics-dir", default=None)
    parser.add_argument("--embedding-source", default=None)
    parser.add_argument("--feature-space", default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_config(args.config)
    effective = apply_id_overrides(
        raw,
        diagnostics_dir=args.diagnostics_dir,
        embedding_source=args.embedding_source,
        feature_space=args.feature_space,
        recompute=args.recompute,
    )
    config = id_config_from_dict(effective)
    result = run_id_estimation(
        config,
        project_root=PROJECT_ROOT,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        _print_dry_run(result)
    else:
        print(f"Finished intrinsic-dimension estimation: {result['local_id_path']}")
        if result["group_id_path"]:
            print(f"Group estimates: {result['group_id_path']}")
        if result["merged_explorer_path"]:
            print(f"Explorer with ID: {result['merged_explorer_path']}")


def _print_dry_run(result: dict) -> None:
    print(f"ID estimation: {result['id_estimation_name']}")
    print(f"Feature space: {result['feature_space']} ({result['source_type']})")
    print(f"Feature shape: {result['feature_shape']}")
    print(f"Metadata rows: {result['metadata_rows']}")
    print(f"Feature path: {result['feature_path']}")
    print(f"Explorer path: {result['explorer_path']}")
    print(f"Local estimators: {', '.join(result['local_estimators'])}")
    print(f"Global estimators: {', '.join(result['global_estimators'])}")
    print(f"k values: {result['k_values']}")
    for column, sizes in result["group_sizes"].items():
        summary = ", ".join(
            f"{value}={size}" for value, size in list(sizes.items())[:10]
        )
        print(f"Groups {column}: {summary}")
    print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
