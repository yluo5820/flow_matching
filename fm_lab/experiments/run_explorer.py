"""Unified geometry explorer CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fm_lab.geometry_explorer.importers import import_existing_outputs
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE
from fm_lab.geometry_explorer.trajectories import build_and_register_trajectory_view
from fm_lab.geometry_explorer.variants import build_dataset_variant, load_variant_config
from fm_lab.geometry_explorer.views import build_projection_view

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and launch the unified geometry explorer.")
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / DEFAULT_WORKSPACE),
        help="Geometry explorer workspace root.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("build-dataset", "build-variant"):
        build = subparsers.add_parser(name, help="Build and register a dataset variant.")
        build.add_argument("--config", required=True, help="Dataset variant YAML config.")

    view = subparsers.add_parser("build-view", help="Build a projection view for a variant.")
    view.add_argument("--dataset", required=True, help="Dataset variant id, e.g. mnist/original.")
    view.add_argument("--config", required=True, help="Image diagnostics projection config.")
    view.add_argument(
        "--feature",
        default=None,
        help="Reserved feature selector for future configs.",
    )

    trajectory = subparsers.add_parser(
        "build-trajectory",
        help="Build and register a trajectory UMAP view for a completed run.",
    )
    trajectory.add_argument("--run-dir", required=True)
    trajectory.add_argument("--solver", default="auto")
    trajectory.add_argument("--nfe", type=int, default=64)
    trajectory.add_argument("--max-target-points", type=int, default=3000)
    trajectory.add_argument("--max-trajectories", type=int, default=None)
    trajectory.add_argument("--n-neighbors", type=int, default=30)
    trajectory.add_argument("--min-dist", type=float, default=0.1)
    trajectory.add_argument("--metric", default="euclidean")
    trajectory.add_argument("--random-state", type=int, default=42)

    importer = subparsers.add_parser("import-existing", help="Index existing outputs/runs.")
    importer.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "outputs" / "dataset_explorer"),
    )
    importer.add_argument("--runs-root", default=str(PROJECT_ROOT / "runs"))

    launch = subparsers.add_parser("launch", help="Launch the Streamlit geometry explorer.")
    launch.add_argument("--dry-run", action="store_true", help="Print the launch command only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    if args.command in {"build-dataset", "build-variant"}:
        config = load_variant_config(args.config)
        result = build_dataset_variant(
            config,
            workspace=workspace,
            project_root=PROJECT_ROOT,
            config_path=args.config,
        )
        print(f"Built dataset variant: {result['variant_id']}")
        print(f"Dataset index: {result['dataset_path']}")
        return
    if args.command == "build-view":
        result = build_projection_view(
            variant_id=args.dataset,
            config_path=args.config,
            workspace=workspace,
            project_root=PROJECT_ROOT,
        )
        print(f"Built projection view: {result['view_id']}")
        print(f"Explorer data: {result['explorer_data']}")
        return
    if args.command == "build-trajectory":
        result = build_and_register_trajectory_view(
            run_dir=args.run_dir,
            workspace=workspace,
            solver=args.solver,
            nfe=args.nfe,
            max_target_points=args.max_target_points,
            max_trajectories=args.max_trajectories,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
        )
        print(f"Registered run: {result['run_id']}")
        print(f"Trajectory views: {', '.join(result['trajectory_views'])}")
        return
    if args.command == "import-existing":
        result = import_existing_outputs(
            workspace=workspace,
            dataset_root=args.dataset_root,
            runs_root=args.runs_root,
        )
        print(
            "Imported existing outputs: "
            f"{result['dataset_views']} dataset views, {result['runs']} runs"
        )
        return
    if args.command == "launch":
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "experiments" / "geometry_explorer_app.py"),
            "--",
            "--workspace",
            str(workspace),
        ]
        if args.dry_run:
            print(" ".join(command))
            return
        raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
