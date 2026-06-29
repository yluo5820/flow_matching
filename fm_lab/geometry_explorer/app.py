"""Streamlit app for the unified geometry explorer."""

from __future__ import annotations

from pathlib import Path

from fm_lab.geometry_explorer.bundles import (
    load_projection_payload,
    load_trajectory_payload,
)
from fm_lab.geometry_explorer.registry import DEFAULT_WORKSPACE, GeometryRegistry
from fm_lab.geometry_explorer.viewer import build_geometry_html


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
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        [data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
        .block-container { padding: 0; max-width: 100%; }
        iframe { display: block; border: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    variants = registry.dataset_variants()
    if not variants:
        st.error(f"No dataset variants are registered under {registry.workspace}.")
        st.stop()

    families = sorted({variant.family for variant in variants})
    family = st.sidebar.selectbox("Dataset family", families)
    family_variants = [variant for variant in variants if variant.family == family]
    variant_labels = {
        variant.variant_id: (
            f"{variant.variant} · {variant.row_count:,} rows"
        )
        for variant in family_variants
    }
    selected_variant_id = st.sidebar.selectbox(
        "Dataset variant",
        list(variant_labels),
        format_func=variant_labels.__getitem__,
    )
    projection_views = registry.projection_views(selected_variant_id)
    trajectory_views = registry.trajectory_views(variant_id=selected_variant_id)
    mode_options = ["Dataset geometry"]
    if trajectory_views:
        mode_options.append("Model trajectory")
    selected_mode = st.sidebar.radio("Mode", mode_options)

    if selected_mode == "Model trajectory":
        labels = {
            view.view_id: f"{view.run_id} · {view.solver} · nfe={view.nfe}"
            for view in trajectory_views
        }
        selected_view = st.sidebar.selectbox(
            "Trajectory view",
            list(labels),
            format_func=labels.__getitem__,
        )
        payload = load_trajectory_payload(selected_view, workspace=registry.workspace)
    else:
        if not projection_views:
            st.error("No projection views are registered for this dataset variant.")
            st.stop()
        labels = {
            view.view_id: f"{view.feature_name} · {len(view.projection_names)} projections"
            for view in projection_views
        }
        selected_view = st.sidebar.selectbox(
            "Projection view",
            list(labels),
            format_func=labels.__getitem__,
        )
        payload = load_projection_payload(selected_view, workspace=registry.workspace)

    html = build_geometry_html(
        payload,
        height=760,
        vendor_dir=registry.workspace / "assets" / "vendor",
    )
    st.iframe(html, height=760, width="stretch", tab_index=0)
