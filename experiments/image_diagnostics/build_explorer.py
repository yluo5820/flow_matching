"""Build UMAP projections, local diagnostics, and explorer data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fm_lab.image_diagnostics.config import (  # noqa: E402
    apply_diagnostics_overrides,
    diagnostics_config_from_dict,
)
from fm_lab.image_diagnostics.runner import run_diagnostics_build  # noqa: E402
from fm_lab.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive UMAP explorer for an image or vector dataset."
    )
    parser.add_argument("--config", required=True, help="Path to explorer YAML config.")
    parser.add_argument(
        "--input-path",
        default=None,
        help="Override the configured dataset root, NumPy path, or image experiment path.",
    )
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--feature-mode",
        choices=("raw", "dinov2"),
        default=None,
        help="Override the feature representation.",
    )
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--recompute-embeddings", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--recompute-projection", action="store_true")
    parser.add_argument("--recompute-diagnostics", action="store_true")
    parser.add_argument("--no-explorer", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count samples without computing features or UMAP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_config(args.config)
    effective = apply_diagnostics_overrides(
        raw,
        input_path=args.input_path,
        experiment_dir=args.experiment_dir,
        feature_mode=args.feature_mode,
        recompute_features=args.recompute_features,
        recompute_embeddings=args.recompute_embeddings,
        recompute_projection=args.recompute_projection,
        recompute_diagnostics=args.recompute_diagnostics,
        no_explorer=args.no_explorer,
    )
    config = diagnostics_config_from_dict(effective)
    result = run_diagnostics_build(
        config,
        project_root=PROJECT_ROOT,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        _print_dry_run(result)
    else:
        print(f"Finished dataset explorer build: {result['output_dir']}")
        if result["explorer_data"] is not None:
            print("Launch explorer:")
            print(
                "  streamlit run experiments/image_diagnostics/explorer_app.py"
            )


def _print_dry_run(result: dict) -> None:
    print(f"Explorer: {result['explorer_name']}")
    print(f"Input type: {result['input_type']}")
    print(f"Source: {result['source']}")
    print(f"Available rows: {result['total_rows']}")
    print(f"Selected samples: {result['selected_samples']}")
    print(f"Skipped rows: {result['skipped_rows']}")
    print(f"Features: {result['feature_name']} ({result['feature_mode']})")
    print(f"Requires model download: {result['requires_model_download']}")
    print(f"Projection: {result['projection_method']}")
    print(f"Diagnostics neighbors: {result['k_neighbors']}")
    print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
