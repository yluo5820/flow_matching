"""Combine precomputed explorer projections into one Streamlit artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fm_lab.image_diagnostics.explorer_merge import (  # noqa: E402
    build_combined_explorer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine aligned precomputed projections into one explorer."
    )
    parser.add_argument("--config", required=True, help="Combined explorer YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explorer_path = build_combined_explorer(
        args.config,
        project_root=PROJECT_ROOT,
    )
    print(f"Finished combined explorer build: {explorer_path.parent.parent}")
    print("Launch explorer:")
    print(
        "  streamlit run experiments/image_diagnostics/explorer_app.py -- "
        f"--data {explorer_path}"
    )


if __name__ == "__main__":
    main()
