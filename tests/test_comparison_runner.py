from pathlib import Path

from fm_lab.experiments.run_comparison import _summarize_rows, _write_report, _write_summary_csv


def test_summarize_rows_ignores_nan_values() -> None:
    rows = [{"t": 0.1, "knn": 1.0}, {"t": 0.2, "knn": float("nan")}, {"t": 0.3, "knn": 3.0}]

    summary = _summarize_rows(rows, prefix="path")

    assert summary["path_knn_mean"] == 2.0
    assert summary["path_knn_max"] == 3.0


def test_write_comparison_artifacts(tmp_path: Path) -> None:
    summaries = [
        {
            "variant": "a",
            "final_loss": 1.0,
            "path_knn_ambiguity_mean": 2.0,
            "path_bayes_gap_mean": 3.0,
            "field_acceleration_mean_mean": 4.0,
            "solver_sliced_wasserstein_max_max": 5.0,
        }
    ]

    _write_summary_csv(summaries, tmp_path / "summary.csv")
    _write_report({"experiment": {"name": "smoke"}}, summaries, tmp_path / "report.md")

    assert "variant" in (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "Mean kNN ambiguity" in (tmp_path / "report.md").read_text(encoding="utf-8")
