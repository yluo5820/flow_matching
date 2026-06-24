"""Merge metadata, projections, diagnostics, and manual labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fm_lab.image_diagnostics.label_store import load_manual_labels


def build_explorer_data(
    metadata: pd.DataFrame,
    projections: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    labels_path: str | Path,
) -> pd.DataFrame:
    """Build the row-aligned table consumed by the Streamlit explorer."""

    frame = metadata.copy()
    frame = frame.merge(projections, on="row_id", how="left", validate="one_to_one")
    diagnostic_columns = [
        column
        for column in diagnostics.columns
        if column == "row_id" or column not in frame.columns
    ]
    frame = frame.merge(
        diagnostics[diagnostic_columns],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    labels = load_manual_labels(labels_path)
    if not labels.empty:
        labels = labels.drop_duplicates("row_id", keep="last")
    frame = frame.merge(labels, on="row_id", how="left", validate="one_to_one")
    frame["manual_label"] = frame["manual_label"].fillna("unlabeled")
    frame["manual_notes"] = frame["manual_notes"].fillna("")
    frame["timestamp"] = frame["timestamp"].fillna("")
    return frame
