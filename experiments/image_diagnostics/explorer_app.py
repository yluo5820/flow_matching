"""CLI entry point for the Streamlit dataset projection explorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fm_lab.image_diagnostics.explorer_app import (  # noqa: E402
    run_auto_explorer,
    run_explorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore a projected image or vector dataset."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data", help="Path to one explorer_data.parquet.")
    source.add_argument(
        "--data-dir",
        default=str(PROJECT_ROOT / "outputs" / "dataset_explorer"),
        help="Directory containing explorer outputs for automatic discovery.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data:
        run_explorer(args.data)
    else:
        run_auto_explorer(args.data_dir)


if __name__ == "__main__":
    main()
