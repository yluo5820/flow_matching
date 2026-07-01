"""Unified geometry explorer CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE
from fm_lab.geometry_explorer.trajectories import build_and_register_trajectory_view
from fm_lab.geometry_explorer.variants import build_dataset_variant, load_variant_config
from fm_lab.geometry_explorer.views import build_projection_view
from fm_lab.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "geometry_explorer"
DEFAULT_VIEW_CONFIG = DEFAULT_CONFIG_DIR / "raw_geometry_view.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and launch the unified geometry explorer.")
    parser.add_argument(
        "--workspace",
        default=str(PROJECT_ROOT / DEFAULT_WORKSPACE),
        help="Geometry explorer workspace root.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-dataset", help="Build and register a dataset.")
    build.add_argument("--config", required=True, help="Dataset YAML config.")

    build_all = subparsers.add_parser(
        "build-all",
        help="Build all standard geometry explorer datasets and views.",
    )
    build_all.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing dataset variant YAML configs.",
    )
    build_all.add_argument(
        "--view-config",
        default=str(DEFAULT_VIEW_CONFIG),
        help="Projection view YAML config to apply to each dataset.",
    )
    build_all.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Only build this dataset variant id, e.g. mnist/original. Repeatable.",
    )
    build_all.add_argument(
        "--clean",
        action="store_true",
        help="Remove the workspace before rebuilding.",
    )
    build_all.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset builds and only build projection views.",
    )
    build_all.add_argument(
        "--skip-views",
        action="store_true",
        help="Skip projection views and only build datasets.",
    )
    build_all.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the build plan without running it.",
    )

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

    launch = subparsers.add_parser("launch", help="Launch the Streamlit geometry explorer.")
    launch.add_argument("--dry-run", action="store_true", help="Print the launch command only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace)
    if args.command == "build-dataset":
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
    if args.command == "build-all":
        _build_all(args, workspace)
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


def _build_all(args: argparse.Namespace, workspace: Path) -> None:
    config_dir = Path(args.config_dir).expanduser()
    view_config = Path(args.view_config).expanduser()
    selected = set(args.dataset or [])
    plan = [
        (path, config, f"{config.family}/{config.variant}")
        for path, config in _discover_dataset_configs(
            config_dir,
            view_config=view_config,
        )
    ]
    if selected:
        plan = [item for item in plan if item[2] in selected]
    if not plan:
        raise SystemExit("No dataset configs matched the build-all request.")

    if args.dry_run:
        print("Geometry explorer build plan")
        print(f"Workspace: {workspace}")
        if args.clean:
            print(f"Would remove workspace: {workspace}")
        if not args.skip_datasets:
            print("Datasets:")
            for path, _, variant_id in plan:
                print(f"  - {variant_id}: {path}")
        if not args.skip_views:
            print(f"Views: {view_config}")
            for _, _, variant_id in plan:
                print(f"  - {variant_id}")
        return

    if args.clean:
        shutil.rmtree(workspace, ignore_errors=True)

    if not args.skip_datasets:
        for path, config, _ in plan:
            result = build_dataset_variant(
                config,
                workspace=workspace,
                project_root=PROJECT_ROOT,
                config_path=path,
            )
            print(f"Built dataset variant: {result['variant_id']}")
            print(f"Dataset index: {result['dataset_path']}")

    if not args.skip_views:
        for _, _, variant_id in plan:
            result = build_projection_view(
                variant_id=variant_id,
                config_path=view_config,
                workspace=workspace,
                project_root=PROJECT_ROOT,
            )
            print(f"Built projection view: {result['view_id']}")
            print(f"Explorer data: {result['explorer_data']}")


def _discover_dataset_configs(
    config_dir: Path,
    *,
    view_config: Path,
) -> list[tuple[Path, object]]:
    if not config_dir.is_dir():
        raise SystemExit(f"Geometry explorer config directory does not exist: {config_dir}")
    view_config_path = view_config.resolve()
    discovered = []
    for path in sorted(config_dir.glob("*.yaml")):
        if path.resolve() == view_config_path:
            continue
        raw = load_config(path)
        if "family" not in raw or "variant" not in raw:
            continue
        discovered.append((path, load_variant_config(path)))
    return discovered


if __name__ == "__main__":
    main()
