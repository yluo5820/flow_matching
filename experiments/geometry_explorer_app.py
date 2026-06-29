"""Streamlit entry point for the unified geometry explorer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fm_lab.geometry_explorer.app import run_geometry_explorer  # noqa: E402
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the unified geometry explorer.")
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / DEFAULT_WORKSPACE),
        help="Geometry explorer workspace root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_geometry_explorer(args.workspace)


if __name__ == "__main__":
    main()
