"""Project saved sampling trajectories into a 3D UMAP plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from fm_lab.diagnostics.trajectory_umap import (
    TrajectoryUMAPConfig,
    project_saved_trajectories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 3D UMAP visualization from saved sampling trajectories."
    )
    parser.add_argument("--run-dir", required=True, help="Completed training run directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to writing into --run-dir.",
    )
    parser.add_argument("--solver", default="auto", help="Solver trajectory name, or auto.")
    parser.add_argument("--nfe", type=int, default=64, help="NFE suffix to project.")
    parser.add_argument(
        "--max-target-points",
        type=int,
        default=3000,
        help="Maximum target reference points included in the UMAP fit and plot.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Optional cap on trajectory paths included in the UMAP fit and plot.",
    )
    parser.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--metric", default="euclidean", help="UMAP input metric.")
    parser.add_argument("--random-state", type=int, default=42, help="UMAP random seed.")
    parser.add_argument(
        "--no-save-coordinates",
        action="store_true",
        help="Do not write projected trajectory coordinates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = project_saved_trajectories(
        TrajectoryUMAPConfig(
            run_dir=Path(args.run_dir),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            solver=args.solver,
            nfe=args.nfe,
            max_target_points=args.max_target_points,
            max_trajectories=args.max_trajectories,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
            save_coordinates=not args.no_save_coordinates,
        )
    )
    print(f"Wrote trajectory UMAP diagnostics: {result['outputs']['json']}")


if __name__ == "__main__":
    main()
