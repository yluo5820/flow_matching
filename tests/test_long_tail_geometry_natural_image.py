import dataclasses

import numpy as np
import pandas as pd
import pytest

from fm_lab.diagnostics.long_tail_geometry.natural_image import (
    analyze_natural_image_transport,
)
from fm_lab.diagnostics.long_tail_geometry.natural_image_preregistration import (
    NaturalImageTransportPreregistration,
)


def _preregistration() -> NaturalImageTransportPreregistration:
    canonical = NaturalImageTransportPreregistration.load(
        "configs/cifar10_lt/long_tail_geometry_natural_image_transport.yaml"
    )
    return dataclasses.replace(canonical, bootstrap_resamples=199)


CLASS_RANKS = (0, 3, 6, 9, 2, 5, 8, 1, 4, 7)
CLASS_COUNTS = (1000, 215, 46, 10, 359, 77, 16, 599, 129, 27)


def _tables(
    prereg: NaturalImageTransportPreregistration,
    *,
    normalized_by_layer: dict[str, float],
    raw_by_layer: dict[str, float] | None = None,
    final_loss_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_by_layer = raw_by_layer or {layer: 0.2 for layer in prereg.layers}
    slope_rows = []
    finite_rows = []
    basis_rows = []
    for checkpoint_step in prereg.checkpoint_steps:
        if checkpoint_step == prereg.baseline_checkpoint_step:
            base_loss = 1.0
            checkpoint_factor = 0.2
        elif checkpoint_step == prereg.early_checkpoint_step:
            base_loss = 0.7
            checkpoint_factor = 0.6
        else:
            base_loss = final_loss_ratio
            checkpoint_factor = 1.0
        for layer in prereg.layers:
            for seed in (0, 1, 2):
                seed_shift = (seed - 1) * 0.02
                for fold in range(len(prereg.fold_offsets)):
                    fold_shift = (fold - 1.5) * 0.002
                    for basis_kind, source in (
                        ("raw", raw_by_layer),
                        ("row_normalized", normalized_by_layer),
                    ):
                        for direction_class in prereg.classes:
                            target_slope = (
                                checkpoint_factor * source[layer]
                                + seed_shift
                                + fold_shift
                                + 0.005 * direction_class
                            )
                            basis_rows.append(
                                {
                                    "checkpoint_step": checkpoint_step,
                                    "layer": layer,
                                    "seed": seed,
                                    "fold": fold,
                                    "basis_kind": basis_kind,
                                    "direction_class": direction_class,
                                    "fit_explained_fraction": 0.7,
                                    "projection_fraction": 0.3,
                                    "fit_row_norm_cv": 0.4,
                                    "basis_vector_sha256": "a" * 64,
                                    "direction_vector_sha256": "b" * 64,
                                    "orientation_gradient_sha256": "c" * 64,
                                    "raw_normalized_basis_abs_cosine": 0.9,
                                }
                            )
                            for evaluation_class in prereg.classes:
                                slope_rows.append(
                                    {
                                        "checkpoint_step": checkpoint_step,
                                        "layer": layer,
                                        "seed": seed,
                                        "fold": fold,
                                        "basis_kind": basis_kind,
                                        "direction_class": direction_class,
                                        "evaluation_class": evaluation_class,
                                        "base_loss": base_loss,
                                        "parameter_norm": 2.0,
                                        "benefit_slope": (
                                            target_slope
                                            if evaluation_class == direction_class
                                            else -0.02 - 0.001 * evaluation_class
                                        ),
                                    }
                                )
                            for partition in ("scale", "evaluation"):
                                predicted = (
                                    target_slope + 0.1
                                    if partition == "scale"
                                    else target_slope
                                )
                                for step in prereg.relative_step_grid:
                                    benefit = step * predicted
                                    finite_rows.append(
                                        {
                                            "checkpoint_step": checkpoint_step,
                                            "layer": layer,
                                            "seed": seed,
                                            "fold": fold,
                                            "basis_kind": basis_kind,
                                            "direction_class": direction_class,
                                            "partition": partition,
                                            "relative_step": step,
                                            "base_loss": base_loss,
                                            "perturbed_loss": base_loss * (1.0 - benefit),
                                            "benefit": benefit,
                                        }
                                    )
    return (
        pd.DataFrame(slope_rows),
        pd.DataFrame(finite_rows),
        pd.DataFrame(basis_rows),
    )


def _reliability(
    prereg: NaturalImageTransportPreregistration,
    *,
    reliable_classes: int,
) -> pd.DataFrame:
    rows = []
    for layer in prereg.layers:
        for seed in (0, 1, 2):
            for class_id in prereg.classes:
                rows.append(
                    {
                        "training_seed": seed,
                        "checkpoint_step": prereg.primary_checkpoint_step,
                        "stratum_id": prereg.stratum_id,
                        "class_id": class_id,
                        "layer_name": layer,
                        "representation": "centered_covariance",
                        "rank": prereg.rank,
                        "measurable": class_id < reliable_classes,
                    }
                )
    return pd.DataFrame(rows)


def _analyze(
    *,
    normalized_by_layer: dict[str, float],
    final_loss_ratio: float = 0.5,
    reliable_classes: int = 10,
):
    prereg = _preregistration()
    slopes, finite, basis = _tables(
        prereg,
        normalized_by_layer=normalized_by_layer,
        final_loss_ratio=final_loss_ratio,
    )
    return analyze_natural_image_transport(
        slopes,
        finite,
        basis,
        _reliability(prereg, reliable_classes=reliable_classes),
        class_counts=CLASS_COUNTS,
        class_ranks=CLASS_RANKS,
        preregistration=prereg,
    )


def test_analysis_stops_when_the_baseline_did_not_learn() -> None:
    prereg = _preregistration()
    analysis = _analyze(
        normalized_by_layer={layer: 1.0 for layer in prereg.layers},
        final_loss_ratio=0.8,
    )

    assert analysis.decision.status == "baseline_not_learned"
    assert analysis.decision.baseline_learned is False
    assert analysis.decision.baseline_loss_ratio == pytest.approx(0.8)
    assert analysis.decision.next_action == "repair_ordinary_cifar_baseline"


def test_analysis_rejects_geometry_that_does_not_repeat_on_cifar() -> None:
    prereg = _preregistration()
    analysis = _analyze(
        normalized_by_layer={layer: 1.0 for layer in prereg.layers},
        reliable_classes=4,
    )

    assert analysis.decision.status == "no_reliable_cifar_geometry"
    assert analysis.decision.reliable_common_classes == (0, 1, 2, 3)
    assert analysis.decision.next_action == "pivot_from_spectral_gradient_geometry"


def test_analysis_confirms_two_layer_natural_image_transport() -> None:
    prereg = _preregistration()
    analysis = _analyze(
        normalized_by_layer={layer: 1.0 for layer in prereg.layers},
    )

    assert analysis.decision.status == "natural_image_transport_confirmed"
    assert analysis.decision.baseline_learned is True
    assert len(analysis.decision.reliable_common_classes) == 10
    assert analysis.decision.next_action == "develop_sign_transport_theory"
    for summary in analysis.decision.layer_summaries.values():
        assert summary["normalized_target_slope_ci_lower"] > 0
        assert summary["normalized_selectivity_slope_median"] > 0
        assert summary["normalized_positive_seed_repeats"] == 3
        assert summary["normalized_largest_concordant_evaluation_step"] == 1e-3


def test_analysis_distinguishes_geometry_without_transport() -> None:
    prereg = _preregistration()
    analysis = _analyze(
        normalized_by_layer={layer: -1.0 for layer in prereg.layers},
    )

    assert analysis.decision.status == "geometry_without_transport"
    assert analysis.decision.next_action == "pivot_from_spectral_gradient_geometry"
    assert all(
        summary["normalized_target_slope_ci_upper"] < 0
        for summary in analysis.decision.layer_summaries.values()
    )


def test_analysis_reports_heterogeneous_natural_image_transport() -> None:
    prereg = _preregistration()
    analysis = _analyze(
        normalized_by_layer={
            prereg.layers[0]: 1.0,
            prereg.layers[1]: -1.0,
        },
    )

    assert analysis.decision.status == "heterogeneous_natural_image_transport"
    assert analysis.decision.next_action == "explain_locked_transport_heterogeneity"


def test_analysis_collapses_folds_before_bootstrap() -> None:
    prereg = _preregistration()
    slopes, finite, basis = _tables(
        prereg,
        normalized_by_layer={layer: 1.0 for layer in prereg.layers},
    )
    outlier = (
        (slopes["checkpoint_step"] == prereg.primary_checkpoint_step)
        & (slopes["layer"] == prereg.layers[0])
        & (slopes["basis_kind"] == "row_normalized")
        & (slopes["seed"] == 0)
        & (slopes["direction_class"] == 0)
        & (slopes["evaluation_class"] == 0)
        & (slopes["fold"] == 0)
    )
    slopes.loc[outlier, "benefit_slope"] = -1000.0

    analysis = analyze_natural_image_transport(
        slopes,
        finite,
        basis,
        _reliability(prereg, reliable_classes=10),
        class_counts=CLASS_COUNTS,
        class_ranks=CLASS_RANKS,
        preregistration=prereg,
    )

    summary = analysis.decision.layer_summaries[prereg.layers[0]]
    assert summary["normalized_target_slope_median"] > 0.9
    assert summary["normalized_positive_seed_repeats"] == 3


def test_analysis_emits_complete_class_frequency_and_interference_outputs() -> None:
    prereg = _preregistration()
    analysis = _analyze(
        normalized_by_layer={layer: 1.0 for layer in prereg.layers},
    )

    assert len(analysis.class_transport) == 3 * 2 * 2 * 10
    assert set(analysis.class_transport["class_count"]) == set(CLASS_COUNTS)
    assert set(analysis.class_transport["class_rank"]) == set(CLASS_RANKS)
    assert len(analysis.frequency_associations) == 3 * 2 * 2
    correlation_columns = [
        column
        for column in analysis.frequency_associations
        if column.endswith("_spearman")
    ]
    assert correlation_columns
    assert np.isfinite(
        analysis.frequency_associations[correlation_columns].to_numpy()
    ).all()
    assert len(analysis.interference_matrices) == 3 * 2 * 2
    assert all(
        matrix.shape == (10, 10)
        for matrix in analysis.interference_matrices.values()
    )


def test_analysis_rejects_incomplete_transport_or_reliability_tables() -> None:
    prereg = _preregistration()
    slopes, finite, basis = _tables(
        prereg,
        normalized_by_layer={layer: 1.0 for layer in prereg.layers},
    )
    reliability = _reliability(prereg, reliable_classes=10)

    with pytest.raises(ValueError, match="complete slope table"):
        analyze_natural_image_transport(
            slopes.iloc[:-1],
            finite,
            basis,
            reliability,
            class_counts=CLASS_COUNTS,
            class_ranks=CLASS_RANKS,
            preregistration=prereg,
        )
    with pytest.raises(ValueError, match="reliability scope"):
        analyze_natural_image_transport(
            slopes,
            finite,
            basis,
            reliability.iloc[:-1],
            class_counts=CLASS_COUNTS,
            class_ranks=CLASS_RANKS,
            preregistration=prereg,
        )
