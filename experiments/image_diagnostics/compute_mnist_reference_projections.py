"""Compute the full MNIST UMAP and t-SNE coordinates locally."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fm_lab.image_diagnostics.mnist_reference_projections import (  # noqa: E402
    DEFAULT_METHODS,
    METHOD_FILENAMES,
    compute_mnist_reference_projections,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Locally compute the 70,000-point MNIST projections used by the "
            "reference-style explorer."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="data/mnist",
        help="Directory containing the four MNIST IDX gzip files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/umap_explorer_local",
        help="Directory for projection JSON files and the reproducibility manifest.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHOD_FILENAMES),
        default=list(DEFAULT_METHODS),
        help="Projection methods to compute. Defaults to the original three 2D views.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. Outputs with fewer than 70,000 rows cannot "
            "be used by the full local explorer config."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Worker count for supported nearest-neighbor phases.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute methods whose output files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.max_samples is not None and args.max_samples < 3:
        raise SystemExit("--max-samples must be at least 3.")
    dataset_root = Path(args.dataset_root).expanduser()
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    manifest = compute_mnist_reference_projections(
        dataset_root=dataset_root,
        output_dir=output_dir,
        methods=args.methods,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        n_jobs=args.n_jobs,
    )
    print(f"Finished local projections for {manifest['samples']} MNIST samples.")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
