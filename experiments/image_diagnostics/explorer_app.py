"""CLI entry point for the Streamlit dataset UMAP explorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fm_lab.image_diagnostics.explorer_app import run_explorer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore a projected image or vector dataset.")
    parser.add_argument("--data", required=True, help="Path to explorer_data.parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_explorer(args.data)


if __name__ == "__main__":
    main()
