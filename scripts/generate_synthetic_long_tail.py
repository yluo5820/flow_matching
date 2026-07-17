#!/usr/bin/env python3
"""Generate shared pools and condition manifests for the synthetic long-tail study."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from fm_lab.geometry_explorer.synthetic_long_tail_design import (
    DIMENSION_IDS,
    OBJECT_IDS,
    build_condition_manifests,
    build_condition_specs,
    build_master_pools,
)
from fm_lab.utils.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    output_root = args.output_root or Path(str(config["output_root"]))
    replicate = int(args.replicate)
    replicates = int(config.get("replicates", 1))
    if not 0 <= replicate < replicates:
        raise ValueError(f"replicate must lie in [0, {replicates}).")
    counts = tuple(int(value) for value in config["counts"])
    if len(counts) != 3:
        raise ValueError("counts must contain exactly three values.")

    pool_count = len(OBJECT_IDS) * len(DIMENSION_IDS)
    condition_count = len(build_condition_specs(replicate, counts=counts))
    if args.dry_run:
        print(f"Dry run for replicate {replicate} at {output_root}")
        print(f"{pool_count} pool cells")
        print(f"{condition_count} condition manifests")
        return 0

    cells = build_master_pools(config, output_root, replicate)
    manifests = build_condition_manifests(
        output_root,
        replicate,
        cells,
        counts=counts,
    )
    print(f"Wrote {len(cells)} pool cells and {len(manifests)} condition manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
