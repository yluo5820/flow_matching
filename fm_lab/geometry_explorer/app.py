"""Streamlit app for the unified geometry explorer."""

from __future__ import annotations

from pathlib import Path

from fm_lab.geometry_explorer.bundles import (
    load_projection_payload,
    load_trajectory_payload,
)
from fm_lab.geometry_explorer.display import (
    family_label,
    projection_view_label,
    trajectory_view_label,
    variant_label,
)
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.viewer import build_geometry_html

VIEWER_CACHE_VERSION = 2


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
            min-width: 150px;
            padding-bottom: 6px;
            font-size: 15px;
            font-weight: 650;
            color: #f2f2f2;
        }
        .geometry-toolbar [data-testid="stSelectbox"],
        .geometry-toolbar [data-testid="stRadio"] {
            min-width: 170px;
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
    variants = registry.dataset_variants()
    if not variants:
        st.error(f"No dataset variants are registered under {registry.workspace}.")
        st.stop()

    st.markdown('<div class="geometry-toolbar">', unsafe_allow_html=True)
    columns = st.columns([1.1, 1.25, 1.75, 1.1, 2.2], vertical_alignment="bottom")
    columns[0].markdown(
        '<div class="geometry-toolbar-title">Geometry Explorer</div>',
        unsafe_allow_html=True,
    )
    families = sorted({variant.family for variant in variants})
    family = columns[1].selectbox(
        "Dataset",
        families,
        format_func=family_label,
        label_visibility="visible",
    )
    family_variants = [variant for variant in variants if variant.family == family]
    variant_labels = {
        variant.variant_id: (
            f"{variant_label(variant.variant)} · {variant.row_count:,} rows"
        )
        for variant in family_variants
    }
    selected_variant_id = columns[2].selectbox(
        "Variant",
        list(variant_labels),
        format_func=variant_labels.__getitem__,
    )
    projection_views = registry.projection_views(selected_variant_id)
    trajectory_views = registry.trajectory_views(variant_id=selected_variant_id)
    mode_options = ["Dataset geometry"]
    if trajectory_views:
        mode_options.append("Model trajectory")
    selected_mode = columns[3].radio("Mode", mode_options, horizontal=True)

    if selected_mode == "Model trajectory":
        labels = {
            view.view_id: trajectory_view_label(
                run_id=view.run_id,
                solver=view.solver,
                nfe=view.nfe,
            )
            for view in trajectory_views
        }
        selected_view = columns[4].selectbox(
            "Trajectory view",
            list(labels),
            format_func=labels.__getitem__,
        )
        html = _cached_view_html(
            st,
            mode="trajectory",
            view_id=selected_view,
            workspace=registry.workspace,
        )
    else:
        if not projection_views:
            st.error("No projection views are registered for this dataset variant.")
            st.stop()
        labels = {
            view.view_id: projection_view_label(
                feature_name=view.feature_name,
                projection_names=view.projection_names,
            )
            for view in projection_views
        }
        selected_view = columns[4].selectbox(
            "Projection view",
            list(labels),
            format_func=labels.__getitem__,
        )
        html = _cached_view_html(
            st,
            mode="projection",
            view_id=selected_view,
            workspace=registry.workspace,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.iframe(html, height=760, width="stretch", tab_index=0)


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
                height=760,
                vendor_dir=workspace / "assets" / "vendor",
            )
        while len(cache) > 3:
            cache.pop(next(iter(cache)))
    return cache[key]
