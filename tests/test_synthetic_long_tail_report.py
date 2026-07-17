from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fm_lab.geometry_explorer.synthetic_long_tail_report import (
    fit_frequency_dimension_effect,
    paired_hierarchical_bootstrap,
    render_research_report,
)


def _effect_frame() -> pd.DataFrame:
    rows = []
    for replicate in range(3):
        for object_id in ("a", "b", "c"):
            for dimension in (1, 3, 5):
                for count in (50, 500, 5000):
                    log_count = math.log10(count)
                    rows.append(
                        {
                            "replicate": replicate,
                            "object_id": object_id,
                            "dimension": dimension,
                            "count": count,
                            "wasserstein_error": (
                                -0.10 * log_count
                                + 0.02 * dimension
                                - 0.08 * log_count * dimension
                                + replicate * 0.01
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def test_effect_fit_recovers_frequency_dimension_interaction() -> None:
    estimate = fit_frequency_dimension_effect(_effect_frame(), outcome="wasserstein_error")

    assert estimate.interaction == pytest.approx(-0.08, abs=1e-8)
    assert estimate.replicate_interactions == pytest.approx((-0.08, -0.08, -0.08))


def test_paired_bootstrap_is_deterministic_and_finite() -> None:
    frame = _effect_frame()

    first = paired_hierarchical_bootstrap(frame, draws=50, seed=7)
    second = paired_hierarchical_bootstrap(frame, draws=50, seed=7)

    assert np.array_equal(first, second)
    assert first.shape == (50,)
    assert np.isfinite(first).all()


def test_effect_fit_rejects_rank_deficient_design() -> None:
    frame = _effect_frame().query("dimension == 1")

    with pytest.raises(ValueError, match="full column rank"):
        fit_frequency_dimension_effect(frame, outcome="wasserstein_error")


def test_report_keeps_hypotheses_separate_from_observations(tmp_path: Path) -> None:
    path = render_research_report(
        {"effects": {}, "calibration": {}, "conditions": []},
        {"entries": []},
        tmp_path / "report.md",
    )
    text = path.read_text(encoding="utf-8")

    assert "## Frozen hypotheses" in text
    assert "## Calibration record" in text
    assert "## Observations" in text
    assert "## Interpretation" in text
    assert text.index("## Frozen hypotheses") < text.index("## Observations")


def test_report_preserves_handwritten_observations(tmp_path: Path) -> None:
    destination = tmp_path / "report.md"
    render_research_report({}, {"entries": []}, destination)
    original = destination.read_text(encoding="utf-8")
    destination.write_text(
        original.replace(
            "Record observations here without changing the frozen hypotheses.",
            "Handwritten observation that must survive regeneration.",
        ),
        encoding="utf-8",
    )

    render_research_report({"effects": {"H1": "inconclusive"}}, {"entries": []}, destination)

    assert "Handwritten observation that must survive regeneration." in destination.read_text(
        encoding="utf-8"
    )
