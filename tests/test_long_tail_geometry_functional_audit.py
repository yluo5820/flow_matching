import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from fm_lab.diagnostics.long_tail_geometry.functional_audit import (
    analyze_functional_geometry_audit,
    paired_projected_directions,
    relative_benefit_slope,
)
from fm_lab.diagnostics.long_tail_geometry.functional_audit_preregistration import (
    FunctionalGeometryAuditPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.gradients import GradientRows


def _preregistration() -> FunctionalGeometryAuditPreregistration:
    canonical = FunctionalGeometryAuditPreregistration.load(
        "configs/fashion_mnist_lt/long_tail_geometry_functional_audit.yaml"
    )
    return dataclasses.replace(canonical, bootstrap_resamples=199)


def test_paired_directions_change_only_fit_representation() -> None:
    raw = torch.tensor(
        [
            [100.0, 0.0],
            [-100.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 1.2],
            [0.0, -0.8],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    norms = torch.linalg.vector_norm(raw, dim=1)
    rows = GradientRows(raw=raw, norms=norms, normalized=raw / norms[:, None])

    result = paired_projected_directions(
        rows,
        fit_positions=(0, 1, 2, 3, 4, 5),
        scale_positions=(6, 7),
        minimum_projection_fraction=1e-8,
    )

    assert set(result) == {"raw", "row_normalized"}
    assert result["raw"].basis_kind == "raw"
    assert result["row_normalized"].basis_kind == "row_normalized"
    assert result["raw"].orientation_gradient_sha256 == (
        result["row_normalized"].orientation_gradient_sha256
    )
    assert result["raw"].basis_vector_sha256 != (
        result["row_normalized"].basis_vector_sha256
    )
    assert abs(
        torch.dot(
            result["raw"].basis_vector,
            result["row_normalized"].basis_vector,
        ).item()
    ) < 1e-5
    assert result["raw"].fit_row_norm_cv > 1.0
    assert torch.linalg.vector_norm(result["raw"].vector).item() == pytest.approx(1.0)
    assert torch.linalg.vector_norm(
        result["row_normalized"].vector
    ).item() == pytest.approx(1.0)


def test_paired_directions_rejects_overlapping_or_invalid_positions() -> None:
    raw = torch.eye(4)
    rows = GradientRows(
        raw=raw,
        norms=torch.ones(4),
        normalized=raw,
    )
    with pytest.raises(ValueError, match="disjoint"):
        paired_projected_directions(
            rows,
            fit_positions=(0, 1),
            scale_positions=(1, 2),
            minimum_projection_fraction=1e-8,
        )
    with pytest.raises(ValueError, match="range"):
        paired_projected_directions(
            rows,
            fit_positions=(0, 9),
            scale_positions=(1, 2),
            minimum_projection_fraction=1e-8,
        )


def test_relative_benefit_slope_matches_definition() -> None:
    direction = torch.tensor([3.0, 4.0]) / 5.0
    mean_gradient = torch.tensor([2.0, -1.0])

    slope = relative_benefit_slope(
        direction=direction,
        evaluation_mean_gradient=mean_gradient,
        parameter_norm=10.0,
        base_loss=4.0,
    )

    expected = -10.0 * torch.dot(mean_gradient, direction).item() / 4.0
    assert slope == pytest.approx(expected)


def test_relative_benefit_slope_matches_central_finite_difference() -> None:
    parameter = torch.tensor([0.8, -0.4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.1, 0.2], dtype=torch.float64)
    direction = torch.tensor([3.0, 4.0], dtype=torch.float64) / 5.0
    base_loss_tensor = (parameter - target).square().mean()
    gradient = torch.autograd.grad(base_loss_tensor, parameter)[0]
    parameter_norm = float(torch.linalg.vector_norm(parameter.detach()))
    base_loss = float(base_loss_tensor.detach())

    analytic = relative_benefit_slope(
        direction=direction,
        evaluation_mean_gradient=gradient,
        parameter_norm=parameter_norm,
        base_loss=base_loss,
    )
    step = 1e-5

    def benefit(relative_step: float) -> float:
        changed = parameter.detach() + relative_step * parameter_norm * direction
        loss = (changed - target).square().mean()
        return -float(loss - base_loss_tensor.detach()) / base_loss

    central = (benefit(step) - benefit(-step)) / (2.0 * step)
    assert analytic == pytest.approx(central, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize(
    ("direction", "gradient", "parameter_norm", "base_loss", "message"),
    [
        (torch.ones(2), torch.ones(3), 1.0, 1.0, "same shape"),
        (torch.ones(2), torch.ones(2), 1.0, 1.0, "unit"),
        (torch.tensor([1.0, 0.0]), torch.tensor([float("nan"), 0.0]), 1.0, 1.0, "finite"),
        (torch.tensor([1.0, 0.0]), torch.ones(2), 0.0, 1.0, "parameter norm"),
        (torch.tensor([1.0, 0.0]), torch.ones(2), 1.0, 0.0, "base loss"),
    ],
)
def test_relative_benefit_slope_rejects_invalid_inputs(
    direction: torch.Tensor,
    gradient: torch.Tensor,
    parameter_norm: float,
    base_loss: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        relative_benefit_slope(
            direction=direction,
            evaluation_mean_gradient=gradient,
            parameter_norm=parameter_norm,
            base_loss=base_loss,
        )


def _tables(
    prereg: FunctionalGeometryAuditPreregistration,
    *,
    raw_slopes: dict[str, float],
    normalized_slopes: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    slope_rows = []
    finite_rows = []
    for checkpoint_step in prereg.checkpoint_steps:
        checkpoint_factor = 1.0 if checkpoint_step == prereg.primary_checkpoint_step else 1.2
        for layer in prereg.layers:
            for seed in (0, 1, 2):
                seed_shift = (seed - 1) * 0.02
                for fold in range(len(prereg.fold_offsets)):
                    fold_shift = (fold - 1.5) * 0.002
                    for basis_kind, source in (
                        ("raw", raw_slopes),
                        ("row_normalized", normalized_slopes),
                    ):
                        target_slope = checkpoint_factor * source[layer] + seed_shift + fold_shift
                        for direction_class in prereg.classes:
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
                                        "base_loss": 1.0,
                                        "parameter_norm": 2.0,
                                        "benefit_slope": (
                                            target_slope
                                            if evaluation_class == direction_class
                                            else 0.05
                                        ),
                                    }
                                )
                            for partition in ("scale", "evaluation"):
                                for step in prereg.relative_step_grid:
                                    slope = (
                                        target_slope
                                        if partition == "evaluation"
                                        else target_slope + 0.1
                                    )
                                    benefit = step * slope
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
                                            "base_loss": 1.0,
                                            "perturbed_loss": 1.0 - benefit,
                                            "benefit": benefit,
                                        }
                                    )
    return pd.DataFrame(slope_rows), pd.DataFrame(finite_rows)


def _constant_by_layer(prereg, value: float) -> dict[str, float]:
    return {layer: value for layer in prereg.layers}


@pytest.mark.parametrize(
    ("raw", "normalized", "expected_status", "expected_action"),
    [
        (
            0.1,
            1.0,
            "normalized_representation_rescue",
            "review_separate_small_local_step_preregistration",
        ),
        (
            0.8,
            0.8,
            "representation_independent_local_transport",
            "study_partition_and_finite_step_curvature",
        ),
        (
            -1.0,
            -1.0,
            "no_transferable_local_descent",
            "pivot_to_gradient_sign_transport_failure",
        ),
    ],
)
def test_audit_analysis_assigns_closed_nonunlocking_statuses(
    raw: float,
    normalized: float,
    expected_status: str,
    expected_action: str,
) -> None:
    prereg = _preregistration()
    slopes, finite = _tables(
        prereg,
        raw_slopes=_constant_by_layer(prereg, raw),
        normalized_slopes=_constant_by_layer(prereg, normalized),
    )

    decision = analyze_functional_geometry_audit(slopes, finite, prereg)

    assert decision.stage1_unlocked is False
    assert decision.probe_b_opened is False
    assert decision.status == expected_status
    assert decision.next_action == expected_action
    assert set(decision.layer_summaries) == set(prereg.layers)
    assert all(
        summary["normalized_largest_concordant_evaluation_step"] == 1e-3
        for summary in decision.layer_summaries.values()
    )


def test_audit_analysis_reports_mixed_layer_transport() -> None:
    prereg = _preregistration()
    raw = _constant_by_layer(prereg, -0.5)
    normalized = {
        prereg.layers[0]: 1.0,
        prereg.layers[1]: -1.0,
    }
    slopes, finite = _tables(
        prereg,
        raw_slopes=raw,
        normalized_slopes=normalized,
    )

    decision = analyze_functional_geometry_audit(slopes, finite, prereg)

    assert decision.status == "mixed_or_class_heterogeneous_transport"
    assert decision.next_action == "analyze_class_and_seed_transport_heterogeneity"


def test_audit_analysis_collapses_folds_before_seed_class_bootstrap() -> None:
    prereg = _preregistration()
    slopes, finite = _tables(
        prereg,
        raw_slopes=_constant_by_layer(prereg, 0.1),
        normalized_slopes=_constant_by_layer(prereg, 1.0),
    )
    outlier = (
        (slopes["checkpoint_step"] == prereg.primary_checkpoint_step)
        & (slopes["layer"] == prereg.layers[0])
        & (slopes["basis_kind"] == "row_normalized")
        & (slopes["seed"] == 0)
        & (slopes["direction_class"] == prereg.classes[0])
        & (slopes["evaluation_class"] == prereg.classes[0])
        & (slopes["fold"] == 0)
    )
    slopes.loc[outlier, "benefit_slope"] = -1000.0

    decision = analyze_functional_geometry_audit(slopes, finite, prereg)

    summary = decision.layer_summaries[prereg.layers[0]]
    assert summary["normalized_target_slope_median"] > 0.9
    assert summary["normalized_positive_seed_repeats"] == 3


def test_audit_analysis_rejects_missing_duplicate_or_nonfinite_cells() -> None:
    prereg = _preregistration()
    slopes, finite = _tables(
        prereg,
        raw_slopes=_constant_by_layer(prereg, 0.1),
        normalized_slopes=_constant_by_layer(prereg, 1.0),
    )
    with pytest.raises(ValueError, match="complete slope table"):
        analyze_functional_geometry_audit(slopes.iloc[:-1], finite, prereg)
    with pytest.raises(ValueError, match="duplicate slope cells"):
        analyze_functional_geometry_audit(
            pd.concat([slopes, slopes.iloc[[0]]], ignore_index=True),
            finite,
            prereg,
        )
    changed = finite.copy()
    changed.loc[0, "benefit"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze_functional_geometry_audit(slopes, changed, prereg)
