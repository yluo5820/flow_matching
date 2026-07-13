import inspect

import numpy as np
import pytest

from fm_lab.evaluation.groups import frequency_ranked_groups, grouped_fid
from fm_lab.evaluation.metrics import (
    classwise_fid,
    fid_score,
    generative_recall,
    inception_score,
    kid_score,
    summarize,
)


def test_fid_is_zero_for_identical_features() -> None:
    features = np.random.default_rng(3).normal(size=(20, 4))

    assert fid_score(features, features) == pytest.approx(0.0, abs=1e-8)


def test_fid_detects_a_mean_shift() -> None:
    real = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    generated = real + 2.0

    assert fid_score(generated, real) == pytest.approx(4.0)


def test_kid_is_seeded_and_requires_two_samples() -> None:
    rng = np.random.default_rng(4)
    real = rng.normal(size=(20, 5))
    generated = rng.normal(size=(20, 5))

    first = kid_score(generated, real, num_subsets=5, max_subset_size=10, seed=7)
    second = kid_score(generated, real, num_subsets=5, max_subset_size=10, seed=7)

    assert first == second
    with pytest.raises(ValueError, match="at least two"):
        kid_score(generated[:1], real)


def test_generative_recall_is_one_for_an_identical_manifold() -> None:
    real = np.array([[0.0], [1.0], [2.0], [4.0]])

    assert generative_recall(real.copy(), real, nearest_k=2) == 1.0


def test_generative_recall_is_zero_for_distant_generated_features() -> None:
    real = np.array([[0.0], [1.0], [2.0], [3.0]])
    generated = real + 100.0

    assert generative_recall(generated, real, nearest_k=1) == 0.0


def test_generative_recall_defaults_to_paper_k_five() -> None:
    assert inspect.signature(generative_recall).parameters["nearest_k"].default == 5


def test_inception_score_matches_balanced_confident_predictions() -> None:
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]] * 5)

    result = inception_score(probabilities, splits=1)

    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == 0.0
    assert result["all"] == pytest.approx([2.0])


def test_classwise_fid_uses_matching_labels() -> None:
    real = np.array([[0.0], [1.0], [10.0], [11.0]])
    generated = np.array([[1.0], [2.0], [10.0], [11.0]])
    labels = np.array([0, 0, 1, 1])

    scores = classwise_fid(generated, labels, real, labels)

    assert scores == {0: pytest.approx(1.0), 1: pytest.approx(0.0)}


def test_frequency_ranked_groups_cover_classes_once() -> None:
    groups = frequency_ranked_groups([100, 80, 60, 40, 20, 10, 5])

    assert groups == {"many": [0, 1, 2], "medium": [3, 4], "few": [5, 6]}
    assert sorted(groups["many"] + groups["medium"] + groups["few"]) == list(range(7))


def test_grouped_fid_pools_features_by_frequency_group() -> None:
    real = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0]])
    generated = real.copy()
    labels = np.array([0, 0, 1, 1, 2, 2])
    groups = {"many": [0], "medium": [1], "few": [2]}

    assert grouped_fid(generated, labels, real, labels, groups) == {
        "many": pytest.approx(0.0),
        "medium": pytest.approx(0.0),
        "few": pytest.approx(0.0),
    }


def test_summarize_uses_population_standard_deviation() -> None:
    assert summarize([1.0, 3.0]) == {"mean": 2.0, "std": 1.0, "all": [1.0, 3.0]}
