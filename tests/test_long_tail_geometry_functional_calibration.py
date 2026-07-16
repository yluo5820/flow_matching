import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from fm_lab.diagnostics.long_tail_geometry.functional_calibration import (
    analyze_functional_calibration,
    cell_microbatch_rows,
    deterministic_random_unit_direction,
    projected_descent_direction,
    response_block_metrics,
    select_layer_scales,
    top_centered_covariance_direction,
    virtual_layer_update,
)
from fm_lab.diagnostics.long_tail_geometry.functional_preregistration import (
    FunctionalCalibrationPreregistration,
)
from fm_lab.diagnostics.long_tail_geometry.manifest import build_probe_manifest


def _manifest():
    return build_probe_manifest(
        np.arange(12, dtype=np.int64),
        np.repeat(np.arange(2, dtype=np.int64), 6),
        split="a",
        rows_per_class_per_stratum=4,
        batch_size=1,
        time_strata=((0.1, 0.4), (0.6, 0.9)),
        seed=17,
    )


def test_cell_microbatch_rows_returns_stable_within_cell_positions() -> None:
    manifest = _manifest()

    rows = cell_microbatch_rows(manifest, class_id=1, stratum_id=0)

    assert len(rows) == 4
    assert [int(manifest.microbatch_ids[value].item()) for value in rows] == [8, 9, 10, 11]
    assert all(int(manifest.labels[value].item()) == 1 for value in rows)
    assert all(int(manifest.stratum_ids[value].item()) == 0 for value in rows)
    assert [int(value.item()) for value in rows] != [0, 1, 2, 3]


def test_cell_microbatch_rows_rejects_missing_or_mixed_cells() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="no microbatches"):
        cell_microbatch_rows(manifest, class_id=9, stratum_id=0)


def test_top_centered_direction_matches_dense_svd_up_to_sign() -> None:
    rows = torch.tensor(
        [
            [3.0, 0.0, 0.2, 1.0],
            [2.0, 1.0, 0.1, 1.0],
            [-2.0, 0.0, -0.2, 1.0],
            [-3.0, -1.0, -0.1, 1.0],
        ]
    )

    result = top_centered_covariance_direction(rows)
    _, singular_values, vh = torch.linalg.svd(
        rows.double() - rows.double().mean(dim=0),
        full_matrices=False,
    )

    assert result.vector.dtype == torch.float32
    assert result.vector.device.type == "cpu"
    assert torch.linalg.vector_norm(result.vector).item() == pytest.approx(1.0)
    assert abs(torch.dot(result.vector.double(), vh[0]).item()) == pytest.approx(
        1.0, abs=1e-6
    )
    assert result.eigenvalue == pytest.approx(float(singular_values[0].square()))
    assert result.explained_fraction == pytest.approx(
        float(singular_values[0].square() / singular_values.square().sum())
    )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (torch.ones(1, 3), "two rows"),
        (torch.tensor([[1.0, float("nan")], [2.0, 3.0]]), "finite"),
        (torch.ones(3, 2), "zero centered rank"),
        (torch.ones(2, 3, 1), "matrix"),
    ],
)
def test_top_centered_direction_rejects_invalid_rows(
    rows: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        top_centered_covariance_direction(rows)


def test_projected_descent_direction_is_oriented_and_reports_fraction() -> None:
    subspace = torch.tensor([3.0, 4.0, 0.0])
    mean_gradient = torch.tensor([2.0, -1.0, 8.0])

    result = projected_descent_direction(
        subspace,
        mean_gradient,
        minimum_projection_fraction=1e-8,
    )

    unit = subspace / torch.linalg.vector_norm(subspace)
    expected_projection = unit * torch.dot(unit, mean_gradient)
    assert torch.allclose(
        result.vector,
        -expected_projection / torch.linalg.vector_norm(expected_projection),
    )
    assert torch.dot(result.vector, mean_gradient).item() < 0
    assert result.projection_fraction == pytest.approx(
        float(
            torch.linalg.vector_norm(expected_projection)
            / torch.linalg.vector_norm(mean_gradient)
        )
    )


def test_projected_descent_direction_rejects_invalid_or_negligible_projection() -> None:
    with pytest.raises(ValueError, match="same shape"):
        projected_descent_direction(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="negligible"):
        projected_descent_direction(
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            minimum_projection_fraction=1e-6,
        )


def test_random_unit_direction_is_deterministic_and_keyed() -> None:
    first = deterministic_random_unit_direction(
        100,
        base_seed=19,
        key=(0, 20_000, 2, "middle.conv2.weight", 7),
    )
    repeated = deterministic_random_unit_direction(
        100,
        base_seed=19,
        key=(0, 20_000, 2, "middle.conv2.weight", 7),
    )
    changed = deterministic_random_unit_direction(
        100,
        base_seed=19,
        key=(0, 20_000, 2, "middle.conv2.weight", 8),
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert torch.linalg.vector_norm(first).item() == pytest.approx(1.0, abs=1e-6)


def test_virtual_layer_update_restores_exact_parameter_after_success_and_error() -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    original = model[0].weight.detach().clone()
    direction = torch.arange(6, dtype=torch.float32) + 1
    direction /= torch.linalg.vector_norm(direction)

    with virtual_layer_update(
        model,
        layer_name="0.weight",
        direction=direction,
        relative_step=1e-3,
    ) as applied_norm:
        assert not torch.equal(model[0].weight, original)
        assert applied_norm == pytest.approx(
            1e-3 * float(torch.linalg.vector_norm(original))
        )
    assert torch.equal(model[0].weight, original)

    with pytest.raises(RuntimeError, match="inside"):
        with virtual_layer_update(
            model,
            layer_name="0.weight",
            direction=direction,
            relative_step=1e-3,
        ):
            raise RuntimeError("inside")
    assert torch.equal(model[0].weight, original)


@pytest.mark.parametrize(
    ("direction", "relative_step", "message"),
    [
        (torch.ones(5), 1e-3, "shape"),
        (torch.ones(6), 0.0, "positive"),
        (torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.float32), 1e-3, "unit"),
    ],
)
def test_virtual_layer_update_rejects_invalid_update(
    direction: torch.Tensor,
    relative_step: float,
    message: str,
) -> None:
    model = nn.Sequential(nn.Linear(3, 2, bias=False))

    with pytest.raises(ValueError, match=message):
        with virtual_layer_update(
            model,
            layer_name="0.weight",
            direction=direction,
            relative_step=relative_step,
        ):
            pass


def _analysis_preregistration() -> FunctionalCalibrationPreregistration:
    canonical = FunctionalCalibrationPreregistration.load(
        "configs/fashion_mnist_lt/long_tail_geometry_functional_calibration.yaml"
    )
    return dataclasses.replace(
        canonical,
        relative_step_grid=(1e-4, 3e-4),
        random_controls=3,
        bootstrap_resamples=199,
    )


def _passing_scale_table(prereg) -> pd.DataFrame:
    rows = []
    for layer in prereg.layers:
        for seed in (0, 1, 2):
            for class_id in prereg.classes:
                for relative_step, benefit, doubled in (
                    (1e-4, 0.004, 0.0082),
                    (3e-4, 0.010, 0.0195),
                ):
                    rows.append(
                        {
                            "checkpoint_step": prereg.primary_checkpoint_step,
                            "layer": layer,
                            "seed": seed,
                            "class_id": class_id,
                            "relative_step": relative_step,
                            "benefit": benefit,
                            "doubled_benefit": doubled,
                        }
                    )
    return pd.DataFrame(rows)


def _passing_responses(prereg) -> pd.DataFrame:
    rows = []
    for checkpoint_step in prereg.checkpoint_steps:
        for layer in prereg.layers:
            for seed in (0, 1, 2):
                for direction_class in prereg.classes:
                    for evaluation_class in prereg.classes:
                        rows.append(
                            {
                                "checkpoint_step": checkpoint_step,
                                "layer": layer,
                                "seed": seed,
                                "direction_class": direction_class,
                                "evaluation_class": evaluation_class,
                                "direction_kind": "primary",
                                "control_id": -1,
                                "benefit": (
                                    0.018
                                    if evaluation_class == direction_class
                                    else 0.001
                                ),
                            }
                        )
                        for control_id in range(prereg.random_controls):
                            rows.append(
                                {
                                    "checkpoint_step": checkpoint_step,
                                    "layer": layer,
                                    "seed": seed,
                                    "direction_class": direction_class,
                                    "evaluation_class": evaluation_class,
                                    "direction_kind": "random",
                                    "control_id": control_id,
                                    "benefit": (
                                        0.003
                                        if evaluation_class == direction_class
                                        else -0.004
                                    ),
                                }
                            )
    return pd.DataFrame(rows)


def test_select_layer_scales_uses_one_shared_step_and_checks_linearity() -> None:
    prereg = _analysis_preregistration()

    selections = select_layer_scales(_passing_scale_table(prereg), prereg)

    assert set(selections) == set(prereg.layers)
    for selection in selections.values():
        assert selection.relative_step == 3e-4
        assert selection.median_benefit == pytest.approx(0.01)
        assert selection.doubled_median_benefit == pytest.approx(0.0195)
        assert selection.local_linearity_relative_error == pytest.approx(0.025)
        assert selection.valid


def test_select_layer_scales_tie_breaks_toward_smaller_step() -> None:
    prereg = _analysis_preregistration()
    table = _passing_scale_table(prereg)
    table.loc[table["relative_step"] == 1e-4, "benefit"] = 0.009
    table.loc[table["relative_step"] == 3e-4, "benefit"] = 0.011

    selections = select_layer_scales(table, prereg)

    assert {value.relative_step for value in selections.values()} == {1e-4}


def test_scale_selection_fails_closed_on_missing_or_nonfinite_cells() -> None:
    prereg = _analysis_preregistration()
    table = _passing_scale_table(prereg)
    with pytest.raises(ValueError, match="complete scale grid"):
        select_layer_scales(table.iloc[:-1], prereg)
    changed = table.copy()
    changed.loc[0, "benefit"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select_layer_scales(changed, prereg)

    missing_block = table[
        ~(
            (table["layer"] == prereg.layers[0])
            & (table["seed"] == 0)
            & (table["class_id"] == prereg.classes[0])
        )
    ]
    with pytest.raises(ValueError, match="seed/class blocks"):
        select_layer_scales(missing_block, prereg)


def test_response_metrics_preserve_target_and_worst_offclass_harm() -> None:
    prereg = _analysis_preregistration()
    metrics = response_block_metrics(_passing_responses(prereg), prereg)
    primary = metrics[metrics["direction_kind"] == "primary"]
    random = metrics[metrics["direction_kind"] == "random"]

    assert set(primary["target_benefit"]) == {0.018}
    assert set(primary["non_target_harm"]) == {0.0}
    assert set(primary["selectivity_margin"]) == {0.018}
    assert set(random["target_benefit"]) == {0.003}
    assert set(random["non_target_harm"]) == {0.004}
    assert set(random["selectivity_margin"]) == {-0.001}


def test_response_metrics_rejects_a_wholly_missing_direction_block() -> None:
    prereg = _analysis_preregistration()
    responses = _passing_responses(prereg)
    missing = responses[
        ~(
            (responses["checkpoint_step"] == prereg.primary_checkpoint_step)
            & (responses["layer"] == prereg.layers[0])
            & (responses["seed"] == 0)
            & (responses["direction_class"] == prereg.classes[0])
            & (responses["direction_kind"] == "primary")
        )
    ]

    with pytest.raises(ValueError, match="complete seed/class response blocks"):
        response_block_metrics(missing, prereg)


def test_functional_decision_unlocks_only_when_both_layers_and_control_pass() -> None:
    prereg = _analysis_preregistration()

    decision = analyze_functional_calibration(
        scale_table=_passing_scale_table(prereg),
        responses=_passing_responses(prereg),
        preregistration=prereg,
    )

    assert decision.stage1_unlocked
    assert decision.positive_control_pass
    assert decision.next_action == "stage1_unlocked_for_separate_preregistration"
    assert set(decision.selected_relative_steps) == set(prereg.layers)
    assert all(summary["passed"] for summary in decision.layer_summaries.values())
    assert all(summary["seed_repeats"] == 3 for summary in decision.layer_summaries.values())


@pytest.mark.parametrize("failure", ["invalid_scale", "offclass_harm", "random_control"])
def test_functional_decision_fails_closed(failure: str) -> None:
    prereg = _analysis_preregistration()
    scales = _passing_scale_table(prereg)
    responses = _passing_responses(prereg)
    if failure == "invalid_scale":
        scales["benefit"] = 0.03
    elif failure == "offclass_harm":
        mask = (
            (responses["direction_kind"] == "primary")
            & (responses["direction_class"] != responses["evaluation_class"])
            & (responses["checkpoint_step"] == prereg.primary_checkpoint_step)
        )
        responses.loc[mask, "benefit"] = -0.02
    else:
        mask = (
            (responses["direction_kind"] == "random")
            & (responses["direction_class"] == responses["evaluation_class"])
            & (responses["checkpoint_step"] == prereg.primary_checkpoint_step)
        )
        responses.loc[mask, "benefit"] = 0.03
        offclass = (
            (responses["direction_kind"] == "random")
            & (responses["direction_class"] != responses["evaluation_class"])
            & (responses["checkpoint_step"] == prereg.primary_checkpoint_step)
        )
        responses.loc[offclass, "benefit"] = 0.0

    decision = analyze_functional_calibration(
        scale_table=scales,
        responses=responses,
        preregistration=prereg,
    )

    assert not decision.stage1_unlocked
    assert decision.next_action == "stop_stage1_and_revise_functional_geometry"


def test_early_positive_control_failure_gets_dynamics_interpretation() -> None:
    prereg = _analysis_preregistration()
    responses = _passing_responses(prereg)
    mask = (
        (responses["checkpoint_step"] == prereg.positive_control_checkpoint_step)
        & (responses["direction_kind"] == "primary")
        & (responses["direction_class"] == responses["evaluation_class"])
    )
    responses.loc[mask, "benefit"] = -0.01

    decision = analyze_functional_calibration(
        scale_table=_passing_scale_table(prereg),
        responses=responses,
        preregistration=prereg,
    )

    assert not decision.positive_control_pass
    assert not decision.stage1_unlocked
    assert decision.next_action == "stop_stage1_and_revise_functional_geometry"
