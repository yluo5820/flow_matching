"""CLI wrapper for FLUX.2-klein batch image generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fm_lab.image_generation.config import (  # noqa: E402
    apply_runtime_overrides,
    batch_generation_config_from_dict,
)
from fm_lab.image_generation.generation_runner import run_generation  # noqa: E402
from fm_lab.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FLUX.2-klein image batches.")
    parser.add_argument("--config", required=True, help="Path to a YAML image generation config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned run.",
    )
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Limit prompts for quick tests.",
    )
    parser.add_argument(
        "--limit-seeds",
        type=int,
        default=None,
        help="Limit seeds for quick tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing image paths and reset run metadata files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
    effective_config = apply_runtime_overrides(
        raw_config,
        dry_run=args.dry_run,
        limit_prompts=args.limit_prompts,
        limit_seeds=args.limit_seeds,
        overwrite=args.overwrite,
    )
    config = batch_generation_config_from_dict(effective_config)
    result = run_generation(config, config_path=args.config, overwrite=args.overwrite)
    if config.runtime.dry_run:
        _print_dry_run(result)
    else:
        print(f"Finished image generation run: {result['output_dir']}")


def _print_dry_run(result: dict) -> None:
    print(f"Experiment: {result['experiment_name']}")
    print(f"Prompts: {result['num_prompts']}")
    print(f"Seeds: {result['num_seeds']}")
    print(f"Planned images: {result['planned_images']}")
    print(f"Output directory: {result['output_dir']}")
    print("First planned output paths:")
    for path in result["preview_paths"]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
