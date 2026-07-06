"""Unified geometry explorer CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from fm_lab.experiments.factory import resolve_device
from fm_lab.geometry_explorer.bundles import load_projection_group_diagnostics
from fm_lab.geometry_explorer.display import metric_label
from fm_lab.geometry_explorer.mnist_labeling import (
    label_fashion_mnist_dataset_variant,
    label_mnist_dataset_variant,
)
from fm_lab.geometry_explorer.model_diagnostics import build_model_diagnostics
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.trajectories import build_and_register_trajectory_view
from fm_lab.geometry_explorer.variants import build_dataset_variant, load_variant_config
from fm_lab.geometry_explorer.views import build_projection_view
from fm_lab.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GEOMETRY_CONFIG_DIR = PROJECT_ROOT / "configs" / "geometry_explorer"
DEFAULT_CONFIG_DIR = DEFAULT_GEOMETRY_CONFIG_DIR / "datasets"
DEFAULT_VIEW_CONFIG = DEFAULT_GEOMETRY_CONFIG_DIR / "views" / "raw_pixels.yaml"


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

    registered_views = subparsers.add_parser(
        "build-registered-views",
        help="Build projection views for currently registered dataset variants.",
    )
    registered_views.add_argument(
        "--view-config",
        default=str(DEFAULT_VIEW_CONFIG),
        help="Projection view YAML config to apply to each registered dataset.",
    )
    registered_views.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Only build this registered dataset variant id. Repeatable.",
    )
    registered_views.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the registered dataset view build plan without running it.",
    )

    label_mnist = subparsers.add_parser(
        "label-mnist",
        help="Annotate an MNIST variant with classifier-predicted digit labels.",
    )
    label_mnist.add_argument("--dataset", required=True, help="Dataset variant id.")
    label_mnist.add_argument(
        "--data-root",
        default="data/mnist",
        help="MNIST IDX root used to train/cache the classifier.",
    )
    label_mnist.add_argument(
        "--normalize",
        default="auto",
        choices=("auto", "zero_one", "minus_one_one"),
        help="Classifier data normalization. Default infers from the dataset value range.",
    )
    label_mnist.add_argument(
        "--classifier-checkpoint",
        default="artifacts/mnist_classifier.pt",
        help="Classifier checkpoint cache path. Normalization suffix is added automatically.",
    )
    label_mnist.add_argument("--classifier-steps", type=int, default=1000)
    label_mnist.add_argument("--classifier-batch-size", type=int, default=256)
    label_mnist.add_argument("--classifier-eval-samples", type=int, default=2048)
    label_mnist.add_argument("--classifier-lr", type=float, default=1.0e-3)
    label_mnist.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence cutoff for classifier_outlier_label metadata.",
    )
    label_mnist.add_argument(
        "--keep-existing-label",
        action="store_true",
        help="Write classifier metadata without replacing the visible label column.",
    )
    label_mnist.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")

    label_fashion = subparsers.add_parser(
        "label-fashion-mnist",
        help="Annotate a Fashion-MNIST variant with classifier-predicted class labels.",
    )
    label_fashion.add_argument("--dataset", required=True, help="Dataset variant id.")
    label_fashion.add_argument(
        "--data-root",
        default="data/fashion_mnist",
        help="Fashion-MNIST IDX root used to train/cache the classifier.",
    )
    label_fashion.add_argument(
        "--normalize",
        default="auto",
        choices=("auto", "zero_one", "minus_one_one"),
        help="Classifier data normalization. Default infers from the dataset value range.",
    )
    label_fashion.add_argument(
        "--classifier-checkpoint",
        default="artifacts/fashion_mnist_classifier.pt",
        help="Classifier checkpoint cache path. Normalization suffix is added automatically.",
    )
    label_fashion.add_argument("--classifier-steps", type=int, default=1000)
    label_fashion.add_argument("--classifier-batch-size", type=int, default=256)
    label_fashion.add_argument("--classifier-eval-samples", type=int, default=2048)
    label_fashion.add_argument("--classifier-lr", type=float, default=1.0e-3)
    label_fashion.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence cutoff for classifier_outlier_label metadata.",
    )
    label_fashion.add_argument(
        "--keep-existing-label",
        action="store_true",
        help="Write classifier metadata without replacing the visible label column.",
    )
    label_fashion.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")

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

    model_diag = subparsers.add_parser(
        "build-model-diagnostics",
        help="Compute model-dependent diagnostics and merge them into registered views.",
    )
    model_diag.add_argument("--dataset", required=True, help="Dataset variant id.")
    model_diag.add_argument("--run-dir", required=True, help="Completed training run directory.")
    model_diag.add_argument(
        "--estimator",
        action="append",
        choices=(
            "fm_jacobian",
            "fm_flipd",
            "diffusion_normal_bundle",
            "diffusion_flipd",
        ),
        default=None,
        help="Model diagnostic estimator to run. Repeatable. Default: fm_jacobian.",
    )
    model_diag.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 0.9, 0.95],
        help="Intermediate FM times for model-dependent diagnostics.",
    )
    model_diag.add_argument("--eps", type=float, default=1e-2)
    model_diag.add_argument("--num-directions", type=int, default=64)
    model_diag.add_argument("--threshold", type=float, default=1e-2)
    model_diag.add_argument(
        "--num-trace-samples",
        type=int,
        default=1,
        help="Hutchinson trace probes for FLIPD estimators. Use 0 for exact divergence.",
    )
    model_diag.add_argument(
        "--num-perturbations",
        type=int,
        default=64,
        help="Score perturbations for diffusion_normal_bundle.",
    )
    model_diag.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for batchable model diagnostics.",
    )
    model_diag.add_argument(
        "--fm-schedule",
        default="auto",
        choices=("auto", "linear", "trig", "cosine"),
        help="Gaussian FM schedule used by fm_flipd. Default infers from checkpoint config.",
    )
    model_diag.add_argument(
        "--diffusion-sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Optional diffusion sigma values. Supply one value or one per --t-values.",
    )
    model_diag.add_argument("--nfe", type=int, default=32)
    model_diag.add_argument(
        "--solver",
        default="rk4",
        choices=("euler", "heun", "midpoint", "rk4"),
    )
    model_diag.add_argument("--max-samples", type=int, default=None)
    model_diag.add_argument("--sample-seed", type=int, default=0)
    model_diag.add_argument("--device", default="auto")
    model_diag.add_argument(
        "--normalize",
        default="auto",
        help="Input normalization for checkpoint evaluation. Default: checkpoint data.normalize.",
    )
    model_diag.add_argument(
        "--view-id",
        default=None,
        help="Only merge into this projection view. Defaults to all views for the dataset.",
    )

    launch = subparsers.add_parser("launch", help="Launch the Streamlit geometry explorer.")
    launch.add_argument("--dry-run", action="store_true", help="Print the launch command only.")

    summarize = subparsers.add_parser(
        "summarize",
        help="Print global/class intrinsic-dimension summaries for registered views.",
    )
    summarize.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Only summarize this dataset variant id. Repeatable.",
    )
    summarize.add_argument(
        "--metric",
        default=None,
        help="Metric column to print. Defaults to the view's primary global ID metric.",
    )
    summarize.add_argument(
        "--include-classes",
        action="store_true",
        help="Print per-class rows below each dataset summary.",
    )
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
    if args.command == "build-registered-views":
        _build_registered_views(args, workspace)
        return
    if args.command == "label-mnist":
        data_root = Path(args.data_root).expanduser()
        if not data_root.is_absolute():
            data_root = PROJECT_ROOT / data_root
        result = label_mnist_dataset_variant(
            variant_id=args.dataset,
            workspace=workspace,
            data_root=data_root,
            normalize=args.normalize,
            classifier_checkpoint=Path(args.classifier_checkpoint),
            classifier_steps=args.classifier_steps,
            classifier_batch_size=args.classifier_batch_size,
            classifier_eval_samples=args.classifier_eval_samples,
            classifier_lr=args.classifier_lr,
            low_confidence_threshold=args.low_confidence_threshold,
            replace_label=not args.keep_existing_label,
            device=resolve_device(args.device),
        )
        print(f"Labeled dataset variant: {result['variant_id']}")
        print(f"Dataset index: {result['dataset_path']}")
        print(f"Classifier checkpoint: {result['classifier']['checkpoint_path']}")
        print(f"Classifier test accuracy: {result['classifier']['test_accuracy']:.4f}")
        print(
            "Low-confidence samples: "
            f"{result['low_confidence_count']} "
            f"({result['low_confidence_fraction']:.2%})"
        )
        return
    if args.command == "label-fashion-mnist":
        data_root = Path(args.data_root).expanduser()
        if not data_root.is_absolute():
            data_root = PROJECT_ROOT / data_root
        result = label_fashion_mnist_dataset_variant(
            variant_id=args.dataset,
            workspace=workspace,
            data_root=data_root,
            normalize=args.normalize,
            classifier_checkpoint=Path(args.classifier_checkpoint),
            classifier_steps=args.classifier_steps,
            classifier_batch_size=args.classifier_batch_size,
            classifier_eval_samples=args.classifier_eval_samples,
            classifier_lr=args.classifier_lr,
            low_confidence_threshold=args.low_confidence_threshold,
            replace_label=not args.keep_existing_label,
            device=resolve_device(args.device),
        )
        print(f"Labeled dataset variant: {result['variant_id']}")
        print(f"Dataset index: {result['dataset_path']}")
        print(f"Classifier checkpoint: {result['classifier']['checkpoint_path']}")
        print(f"Classifier test accuracy: {result['classifier']['test_accuracy']:.4f}")
        print(
            "Low-confidence samples: "
            f"{result['low_confidence_count']} "
            f"({result['low_confidence_fraction']:.2%})"
        )
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
    if args.command == "build-model-diagnostics":
        result = build_model_diagnostics(
            variant_id=args.dataset,
            run_dir=args.run_dir,
            workspace=workspace,
            estimators=tuple(args.estimator or ["fm_jacobian"]),
            t_values=tuple(args.t_values),
            eps=args.eps,
            num_directions=args.num_directions,
            threshold=args.threshold,
            num_trace_samples=None if args.num_trace_samples == 0 else args.num_trace_samples,
            num_perturbations=args.num_perturbations,
            batch_size=args.batch_size,
            fm_schedule=args.fm_schedule,
            diffusion_sigmas=(
                tuple(args.diffusion_sigmas) if args.diffusion_sigmas is not None else None
            ),
            nfe=args.nfe,
            solver=args.solver,
            max_samples=args.max_samples,
            sample_seed=args.sample_seed,
            device=args.device,
            normalize=args.normalize,
            view_id=args.view_id,
        )
        print(f"Built model diagnostics for run: {result['run_id']}")
        print(f"Local diagnostics: {result['local_path']}")
        print(f"Merged views: {len(result['merged_views'])}")
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
    if args.command == "summarize":
        _summarize(args, workspace)
        return


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


def _build_registered_views(args: argparse.Namespace, workspace: Path) -> None:
    view_config = Path(args.view_config).expanduser()
    registry = GeometryRegistry(workspace)
    selected = set(args.dataset or [])
    variants = registry.dataset_variants()
    if selected:
        variants = [variant for variant in variants if variant.variant_id in selected]
    if not variants:
        raise SystemExit("No registered dataset variants matched the request.")

    if args.dry_run:
        print("Registered dataset view build plan")
        print(f"Workspace: {workspace}")
        print(f"View config: {view_config}")
        print("Datasets:")
        for variant in variants:
            print(f"  - {variant.variant_id} ({variant.row_count:,} rows)")
        return

    for variant in variants:
        result = build_projection_view(
            variant_id=variant.variant_id,
            config_path=view_config,
            workspace=workspace,
            project_root=PROJECT_ROOT,
        )
        print(f"Built projection view: {result['view_id']}")
        print(f"Explorer data: {result['explorer_data']}")


def _summarize(args: argparse.Namespace, workspace: Path) -> None:
    registry = GeometryRegistry(workspace)
    selected = set(args.dataset or [])
    variants = registry.dataset_variants()
    if selected:
        variants = [variant for variant in variants if variant.variant_id in selected]
    if not variants:
        raise SystemExit("No dataset variants matched the summarize request.")

    print(f"Geometry explorer summary: {workspace}")
    for variant in variants:
        views = registry.projection_views(variant.variant_id)
        if not views:
            print(f"{variant.variant_id}: no projection views")
            continue
        view = _preferred_projection_view(views)
        diagnostics = load_projection_group_diagnostics(
            view.view_id,
            workspace=workspace,
        )
        if not diagnostics:
            print(f"{variant.variant_id}: no group intrinsic-dimension estimates")
            continue
        metric = args.metric or diagnostics.get("primaryMetric")
        if metric not in diagnostics.get("metrics", []):
            available = ", ".join(diagnostics.get("metrics", []))
            print(f"{variant.variant_id}: metric {metric!r} not found. Available: {available}")
            continue
        label = metric_label(metric)
        overall = diagnostics.get("overall") or {}
        print(
            f"{variant.variant_id} | {view.feature_name} | "
            f"{label}: {_format_summary_value(overall.get(metric))} | "
            f"n={_format_summary_value(overall.get('n_samples'))}"
        )
        if args.include_classes:
            for class_label, values in sorted(
                diagnostics.get("groups", {}).items(),
                key=lambda item: item[0],
            ):
                print(
                    f"  {class_label}: {label}={_format_summary_value(values.get(metric))}; "
                    f"n={_format_summary_value(values.get('n_samples'))}; "
                    f"share={_format_summary_percent(values.get('class_share'))}"
                )


def _preferred_projection_view(views: list[object]) -> object:
    return sorted(
        views,
        key=lambda view: (
            0 if view.feature_name == "raw_pixels" else 1,
            view.view_id,
        ),
    )[0]


def _format_summary_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_summary_percent(value: object) -> str:
    if not isinstance(value, int | float):
        return "-"
    return f"{float(value) * 100:.1f}%"


def _discover_dataset_configs(
    config_dir: Path,
    *,
    view_config: Path,
) -> list[tuple[Path, object]]:
    if not config_dir.is_dir():
        raise SystemExit(f"Geometry explorer config directory does not exist: {config_dir}")
    view_config_path = view_config.resolve()
    discovered = []
    for path in sorted(config_dir.rglob("*.yaml")):
        if path.resolve() == view_config_path:
            continue
        if "models" in path.relative_to(config_dir).parts:
            continue
        raw = load_config(path)
        if "family" not in raw or "variant" not in raw:
            continue
        discovered.append((path, load_variant_config(path)))
    return discovered


if __name__ == "__main__":
    main()
