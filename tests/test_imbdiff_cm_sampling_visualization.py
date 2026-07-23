from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from fm_lab.experiments.run_imbdiff_cm_sampling_visualization import (
    load_cifar100_class_names,
    render_expert_residual_grid,
    select_visualization_rows,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    sample_index = 0
    for group_name, class_ids in (("many", (0, 1, 2)), ("few", (7, 8, 9))):
        for class_id in class_ids:
            for rms in (0.1, 0.2, 0.9):
                rows.append(
                    {
                        "sample_index": sample_index,
                        "class_id": class_id,
                        "frequency_group": group_name,
                        "learned_vs_general_rms": rms,
                    }
                )
                sample_index += 1
    return rows


def test_selection_uses_frequency_quantiles_and_class_median_samples() -> None:
    selected = select_visualization_rows(
        _rows(),
        groups=("many", "few"),
        classes_per_group=2,
        samples_per_class=1,
    )

    assert [(row["frequency_group"], row["class_id"]) for row in selected] == [
        ("many", 0),
        ("many", 2),
        ("few", 7),
        ("few", 9),
    ]
    assert [row["sample_residual_rms"] for row in selected] == [0.2] * 4


def test_class_names_load_from_python_cifar_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "cifar-100-python"
    data_dir.mkdir()
    expected = [f"class_{index}".encode() for index in range(100)]
    with (data_dir / "meta").open("wb") as handle:
        pickle.dump({b"fine_label_names": expected}, handle)

    names = load_cifar100_class_names(tmp_path)

    assert names[0] == "class_0"
    assert names[-1] == "class_99"


def test_render_expert_residual_grid_writes_png(tmp_path: Path) -> None:
    general = np.zeros((4, 3, 32, 32), dtype=np.float32)
    learned = general.copy()
    learned[:, 0, 8:24, 8:24] = 0.1
    selected = [
        {
            "sample_index": index,
            "class_id": index,
            "frequency_group": "many" if index < 2 else "few",
            "sample_residual_rms": 0.05,
        }
        for index in range(4)
    ]
    output = tmp_path / "grid.png"

    render_expert_residual_grid(
        general=general,
        learned=learned,
        selected=selected,
        residual_scale=0.1,
        residual_quantile=0.995,
        class_names=tuple(f"class_{index}" for index in range(100)),
        output_path=output,
        dpi=80,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
