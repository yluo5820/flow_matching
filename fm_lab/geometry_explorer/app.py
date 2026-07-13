"""Streamlit app for the unified geometry explorer."""

from __future__ import annotations

from pathlib import Path

from fm_lab.geometry_explorer.bundles import (
    load_projection_payload,
    load_trajectory_payload,
)
from fm_lab.geometry_explorer.display import (
    family_label,
    model_run_label,
    projection_view_label,
    trajectory_option_label,
    variant_label,
)
from fm_lab.geometry_explorer.registry import (
    DEFAULT_WORKSPACE,
    GeometryRegistry,
    ModelRunRecord,
    TrajectoryViewRecord,
)
from fm_lab.geometry_explorer.viewer import build_geometry_html
from fm_lab.utils.config import load_config

EXPLORER_HEIGHT = 920
VIEWER_CACHE_VERSION = 10


def run_geometry_explorer(workspace: str | Path = DEFAULT_WORKSPACE) -> None:
    """Run the registry-backed unified explorer in Streamlit."""

    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            'The geometry explorer requires streamlit. Install ".[image-diagnostics]".'
        ) from exc

    registry = GeometryRegistry(workspace)
    st.set_page_config(
        page_title="Geometry Explorer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stSidebar"] { display: none; }
        .stApp { background: #111; }
        .block-container { padding: 0; max-width: 100%; }
        iframe { display: block; border: 0; }
        .geometry-toolbar {
            display: flex;
            align-items: end;
            gap: 14px;
            padding: 10px 14px 8px;
            background: #181818;
            border-bottom: 1px solid #2b2b2b;
            color: #f2f2f2;
        }
        .geometry-toolbar-title {
            min-width: 132px;
            padding-bottom: 6px;
            font-size: 15px;
            font-weight: 650;
            color: #f2f2f2;
        }
        .geometry-toolbar [data-testid="stSelectbox"],
        .geometry-toolbar [data-testid="stRadio"] {
            min-width: 145px;
        }
        .geometry-toolbar label,
        .geometry-toolbar p {
            color: #c8c8c8 !important;
            font-size: 12px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    variants = registry.dataset_variants(explorable_only=True)
    if not variants:
        st.error(
            "No dataset variants with projection or trajectory views are "
            f"registered under {registry.workspace}."
        )
        st.stop()

    st.markdown('<div class="geometry-toolbar">', unsafe_allow_html=True)
    columns = st.columns(
        [0.95, 1.25, 1.75, 2.1, 1.05, 2.0, 1.35],
        vertical_alignment="bottom",
    )
    columns[0].markdown(
        '<div class="geometry-toolbar-title">Geometry Explorer</div>',
        unsafe_allow_html=True,
    )
    families = sorted({variant.family for variant in variants})
    family = columns[1].selectbox(
        "Dataset family",
        families,
        format_func=family_label,
        label_visibility="visible",
        key="geometry_family",
    )
    family_variants = [variant for variant in variants if variant.family == family]
    variant_labels = {
        variant.variant_id: (
            f"{variant_label(variant.variant)} · {variant.row_count:,} rows"
        )
        for variant in family_variants
    }
    selected_variant_id = columns[2].selectbox(
        "Dataset",
        list(variant_labels),
        format_func=variant_labels.__getitem__,
        key=f"geometry_dataset_{family}",
    )
    projection_views = registry.projection_views(selected_variant_id)
    model_runs = registry.model_runs(selected_variant_id)
    trajectory_views = registry.trajectory_views(variant_id=selected_variant_id)
    trajectory_views_by_run = _trajectory_views_by_run(trajectory_views)
    model_runs_with_trajectories = [
        run for run in model_runs if trajectory_views_by_run.get(run.run_id)
    ]
    if projection_views:
        projection_labels = {
            view.view_id: projection_view_label(
                feature_name=view.feature_name,
                projection_names=view.projection_names,
            )
            for view in projection_views
        }
        selected_projection_view = columns[3].selectbox(
            "Dataset view",
            list(projection_labels),
            format_func=projection_labels.__getitem__,
            key=f"geometry_projection_{selected_variant_id}",
        )
    else:
        selected_projection_view = None
        columns[3].selectbox(
            "Dataset view",
            ["No projection views"],
            key=f"geometry_projection_empty_{selected_variant_id}",
        )

    mode_options = ["Dataset geometry"]
    if model_runs_with_trajectories:
        mode_options.append("Model trajectory")
    default_mode_index = (
        1 if selected_projection_view is None and len(mode_options) > 1 else 0
    )
    selected_mode = columns[4].radio(
        "Mode",
        mode_options,
        index=default_mode_index,
        horizontal=True,
        key=f"geometry_mode_{selected_variant_id}",
    )

    if selected_mode == "Model trajectory":
        run_labels = _model_run_labels(model_runs_with_trajectories)
        selected_run_id = columns[5].selectbox(
            "Model",
            [run.run_id for run in model_runs_with_trajectories],
            format_func=run_labels.__getitem__,
            key=f"geometry_model_{selected_variant_id}",
        )
        run_trajectory_views = trajectory_views_by_run.get(selected_run_id, [])
        labels = {
            view.view_id: trajectory_option_label(
                solver=view.solver,
                nfe=view.nfe,
            )
            for view in run_trajectory_views
        }
        selected_view = columns[6].selectbox(
            "Trajectory",
            list(labels),
            format_func=labels.__getitem__,
            key=f"geometry_trajectory_{selected_run_id}",
        )
        html = _cached_view_html(
            st,
            mode="trajectory",
            view_id=selected_view,
            workspace=registry.workspace,
        )
    else:
        columns[5].selectbox(
            "Model",
            ["None"],
            key=f"geometry_model_none_{selected_variant_id}",
        )
        columns[6].selectbox(
            "Trajectory",
            ["None"],
            key=f"geometry_trajectory_none_{selected_variant_id}",
        )
        if selected_projection_view is None:
            st.error("No projection views are registered for this dataset variant.")
            st.stop()
        html = _cached_view_html(
            st,
            mode="projection",
            view_id=selected_projection_view,
            workspace=registry.workspace,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.iframe(html, height=EXPLORER_HEIGHT, width="stretch", tab_index=0)


def _cached_view_html(st, *, mode: str, view_id: str, workspace: Path) -> str:
    cache = st.session_state.setdefault("_geometry_view_html_cache", {})
    registry_path = workspace / "registry.sqlite"
    registry_mtime = registry_path.stat().st_mtime_ns if registry_path.exists() else 0
    key = (VIEWER_CACHE_VERSION, mode, view_id, str(workspace), registry_mtime)
    if key not in cache:
        with st.spinner("Loading geometry view..."):
            payload = (
                load_trajectory_payload(view_id, workspace=workspace)
                if mode == "trajectory"
                else load_projection_payload(view_id, workspace=workspace)
            )
            cache[key] = build_geometry_html(
                payload,
                height=EXPLORER_HEIGHT,
                vendor_dir=workspace / "assets" / "vendor",
            )
        while len(cache) > 12:
            cache.pop(next(iter(cache)))
    return cache[key]


def _trajectory_views_by_run(
    trajectory_views: list[TrajectoryViewRecord],
) -> dict[str, list[TrajectoryViewRecord]]:
    grouped: dict[str, list[TrajectoryViewRecord]] = {}
    for view in trajectory_views:
        grouped.setdefault(view.run_id, []).append(view)
    return grouped


def _model_run_labels(model_runs: list[ModelRunRecord]) -> dict[str, str]:
    return {
        run.run_id: model_run_label(
            run_id=run.run_id,
            config=_load_model_run_config(run),
        )
        for run in model_runs
    }


def _load_model_run_config(run: ModelRunRecord) -> dict | None:
    if run.config_path is None or not run.config_path.exists():
        return None
    try:
        return load_config(run.config_path)
    except Exception:
        return None
