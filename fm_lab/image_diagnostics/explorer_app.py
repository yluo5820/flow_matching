"""Streamlit explorer for generated-image embedding diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import pairwise_distances

from fm_lab.image_diagnostics.canvas_explorer import render_thumbnail_canvas
from fm_lab.image_diagnostics.config import ExplorerConfig, diagnostics_config_from_dict
from fm_lab.image_diagnostics.label_store import MANUAL_LABELS, save_manual_label
from fm_lab.image_diagnostics.projections import projection_variants
from fm_lab.image_diagnostics.save_utils import read_parquet
from fm_lab.utils.config import load_config


def run_explorer(data_path: str | Path) -> None:
    """Render the Streamlit explorer for a built diagnostics dataset."""

    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError(
            'The explorer requires streamlit. Install ".[image-diagnostics]".'
        ) from exc

    path = Path(data_path).expanduser().resolve()
    st.set_page_config(
        page_title="Dataset UMAP Explorer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        [data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
        .block-container { padding: 0; max-width: 100%; }
        [data-testid="stExpander"] { margin: 12px 16px 24px; }
        iframe { display: block; border: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not path.exists():
        st.error(f"Explorer data does not exist: {path}")
        st.stop()

    frame = read_parquet(path)
    frame = _merge_latest_labels(frame, path.parent / "manual_labels.csv")
    explorer_config, projection_names = _explorer_settings(path)
    render_thumbnail_canvas(
        frame,
        data_path=path,
        config=explorer_config,
        projection_names=projection_names,
    )
    if not explorer_config.show_workspace:
        return

    with st.expander("Diagnostics workspace"):
        filtered = _inline_filters(st, frame)
        if filtered.empty:
            st.warning("No samples match the current filters.")
            return
        row_ids = filtered["row_id"].astype(int).tolist()
        selected_row_id = st.selectbox(
            "Selected row",
            row_ids,
            format_func=lambda value: _row_label(filtered, value),
        )
        selected = filtered[
            filtered["row_id"].astype(int) == int(selected_row_id)
        ].iloc[0]
        _render_image_details(st, selected, path)
        _render_neighbors(st, selected, frame, path)
        st.dataframe(_display_frame(filtered), width="stretch", hide_index=True)


def _inline_filters(st: Any, frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame.copy()
    filter_columns = [
        column
        for column in ("dataset", "split", "sample_type", "label", "manual_label")
        if column in filtered
    ]
    controls = st.columns(max(1, len(filter_columns)))
    for control, column in zip(controls, filter_columns, strict=False):
        if column not in filtered:
            continue
        values = sorted(
            filtered[column].dropna().unique().tolist(),
            key=lambda value: str(value),
        )
        selected = control.multiselect(column, values)
        if selected:
            filtered = filtered[filtered[column].isin(selected)]

    if "tags" in filtered and not filtered.empty:
        tag_options = sorted(
            {
                tag
                for value in filtered["tags"]
                for tag in _tag_list(value)
            }
        )
        selected_tags = st.multiselect("tags", tag_options)
        if selected_tags:
            selected_set = set(selected_tags)
            filtered = filtered[
                filtered["tags"].map(lambda value: bool(selected_set & set(_tag_list(value))))
            ]
    return filtered


def _render_image_details(st: Any, selected: pd.Series, data_path: Path) -> None:
    image_column, detail_column = st.columns([1, 1])
    raw_image_path = str(selected.get("image_path", "")).strip()
    image_path = Path(raw_image_path) if raw_image_path else None
    with image_column:
        if image_path is not None and image_path.is_file():
            st.image(str(image_path), width="stretch")
        elif (preview := _atlas_preview(selected)) is not None:
            st.image(preview, width="stretch")
        else:
            st.info("This sample has no image preview.")
    with detail_column:
        label = selected.get("label", "")
        heading = f"Label {label}" if str(label) else str(selected.get("prompt_id", "Sample"))
        st.subheader(heading)
        prompt = str(selected.get("prompt", ""))
        if prompt:
            st.write(prompt)
        detail_keys = [
            key
            for key in (
                "dataset",
                "split",
                "sample_type",
                "label",
                "source_index",
                "family",
                "tags",
                "seed",
                "model_repo_id",
                "two_nn_lid",
                _first_column(selected.index, "knn_radius_k"),
                _first_column(selected.index, "participation_ratio_k"),
                "distance_to_label_centroid",
                "distance_to_prompt_centroid",
                "distance_to_family_centroid",
                "outlier_score",
            )
            if key and key in selected.index
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    {"metric": key, "value": _display_value(selected.get(key))}
                    for key in detail_keys
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        current_label = str(selected.get("manual_label", "unlabeled"))
        label = st.selectbox(
            "Manual label",
            MANUAL_LABELS,
            index=MANUAL_LABELS.index(current_label)
            if current_label in MANUAL_LABELS
            else 0,
            key=f"label_{int(selected['row_id'])}",
        )
        notes = st.text_area(
            "Notes",
            value=str(selected.get("manual_notes", "")),
            key=f"notes_{int(selected['row_id'])}",
        )
        if st.button("Save label", type="primary"):
            save_manual_label(
                data_path.parent / "manual_labels.csv",
                row_id=int(selected["row_id"]),
                manual_label=label,
                manual_notes=notes,
            )
            st.success("Label saved.")
            st.rerun()


def _render_neighbors(
    st: Any,
    selected: pd.Series,
    explorer_frame: pd.DataFrame,
    data_path: Path,
) -> None:
    feature_dir = data_path.parent.parent / "features"
    feature_paths = sorted(feature_dir.glob("*_features.npy"))
    if not feature_paths:
        return
    feature_names = [path.name.removesuffix("_features.npy") for path in feature_paths]
    feature_name = st.selectbox("Neighbor features", feature_names)
    feature_path = feature_dir / f"{feature_name}_features.npy"
    metadata_path = feature_dir / f"{feature_name}_metadata.parquet"
    if not metadata_path.exists():
        return
    features = np.load(feature_path, mmap_mode="r")
    metadata = read_parquet(metadata_path)
    matches = np.flatnonzero(metadata["row_id"].astype(int).to_numpy() == int(selected["row_id"]))
    if not len(matches):
        st.info("The selected row is not present in this embedding cache.")
        return
    selected_position = int(matches[0])
    distances = pairwise_distances(
        np.asarray(features[selected_position : selected_position + 1]),
        np.asarray(features),
        metric=_feature_metric(data_path),
    )[0]
    order = np.argsort(distances)
    order = order[order != selected_position][:5]
    st.subheader("Nearest neighbors")
    columns = st.columns(max(1, len(order)))
    labels = explorer_frame.set_index("row_id")
    for column, position in zip(columns, order, strict=False):
        neighbor = metadata.iloc[int(position)]
        row_id = int(neighbor["row_id"])
        with column:
            raw_path = str(neighbor.get("image_path", "")).strip()
            path = Path(raw_path) if raw_path else None
            if path is not None and path.is_file():
                st.image(str(path), width="stretch")
            elif (preview := _atlas_preview(neighbor)) is not None:
                st.image(preview, width="stretch")
            manual_label = (
                labels.loc[row_id, "manual_label"]
                if row_id in labels.index
                else "unlabeled"
            )
            st.caption(
                f"label={neighbor.get('label', '')} | {neighbor.get('family', '')}\n"
                f"distance={distances[position]:.4f} | {manual_label}"
            )


def _merge_latest_labels(frame: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        return frame
    labels = pd.read_csv(labels_path)
    if labels.empty:
        return frame
    labels = labels.drop_duplicates("row_id", keep="last").set_index("row_id")
    updated = frame.copy()
    for column in ("manual_label", "manual_notes", "timestamp"):
        if column in labels:
            mapping = labels[column].to_dict()
            replacement = updated["row_id"].map(mapping)
            existing = (
                updated[column]
                if column in updated
                else pd.Series(index=updated.index, dtype=object)
            )
            updated[column] = replacement.combine_first(existing)
    return updated


def _explorer_settings(
    data_path: Path,
) -> tuple[ExplorerConfig, dict[str, str]]:
    config_path = data_path.parent.parent / "config_used.yaml"
    if not config_path.exists():
        return ExplorerConfig(), {}
    config = diagnostics_config_from_dict(load_config(config_path))
    names = {
        variant.key: variant.name
        for variant in projection_variants(config.projection)
    }
    return config.explorer, names


def _atlas_preview(row: pd.Series) -> Image.Image | None:
    required = (
        "sprite_atlas_path",
        "sprite_atlas_column",
        "sprite_atlas_row",
        "sprite_tile_size",
    )
    if not all(key in row.index for key in required):
        return None
    path = Path(str(row.get("sprite_atlas_path", "")))
    if not path.is_file():
        return None
    tile_size = int(row["sprite_tile_size"])
    left = int(row["sprite_atlas_column"]) * tile_size
    top = int(row["sprite_atlas_row"]) * tile_size
    with Image.open(path) as atlas:
        return atlas.crop((left, top, left + tile_size, top + tile_size)).copy()


def _first_column(columns: Any, prefix: str) -> str | None:
    return next((column for column in columns if str(column).startswith(prefix)), None)


def _tag_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _row_label(frame: pd.DataFrame, row_id: int) -> str:
    row = frame[frame["row_id"].astype(int) == int(row_id)].iloc[0]
    label = row.get("label", "")
    source_index = row.get("source_index", "")
    return f"{row_id}: label={label} source={source_index}"


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    displayed = frame.copy()
    for column in displayed.select_dtypes(include="object"):
        displayed[column] = displayed[column].map(_display_value)
    return displayed


def _display_value(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(str(item) for item in value)
    if pd.isna(value):
        return ""
    return str(value)


def _feature_metric(data_path: Path) -> str:
    config_path = data_path.parent.parent / "config_used.yaml"
    if not config_path.exists():
        return "euclidean"
    raw = load_config(config_path)
    diagnostics = raw.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        return "euclidean"
    return str(diagnostics.get("metric", "euclidean"))
