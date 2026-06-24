"""CSV-backed manual failure labels for the diagnostics explorer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

MANUAL_LABELS = [
    "unlabeled",
    "pass",
    "interesting",
    "outlier",
    "misclustered",
    "artifact",
    "uncertain",
]
LABEL_COLUMNS = ["row_id", "manual_label", "manual_notes", "timestamp"]


def ensure_label_store(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        pd.DataFrame(columns=LABEL_COLUMNS).to_csv(output_path, index=False)
    return output_path


def load_manual_labels(path: str | Path) -> pd.DataFrame:
    label_path = ensure_label_store(path)
    frame = pd.read_csv(label_path)
    for column in LABEL_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame.loc[:, LABEL_COLUMNS]


def save_manual_label(
    path: str | Path,
    *,
    row_id: int,
    manual_label: str,
    manual_notes: str,
) -> Path:
    """Atomically insert or replace one row label."""

    if manual_label not in MANUAL_LABELS:
        raise ValueError(f"Unknown manual label: {manual_label}")
    output_path = ensure_label_store(path)
    frame = load_manual_labels(output_path)
    if not frame.empty:
        frame = frame[frame["row_id"].astype(str) != str(row_id)]
    new_row = pd.DataFrame(
        [
            {
                "row_id": int(row_id),
                "manual_label": manual_label,
                "manual_notes": manual_notes,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ]
    )
    frame = pd.concat([frame, new_row], ignore_index=True)
    frame = frame.sort_values("row_id")
    temporary = output_path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output_path)
    return output_path
