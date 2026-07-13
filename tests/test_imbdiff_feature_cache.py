import json

import numpy as np
import pytest

from fm_lab.evaluation.cache import (
    FeatureCache,
    cache_fingerprint,
    load_feature_cache,
    save_feature_cache,
)


def _cache() -> FeatureCache:
    return FeatureCache(
        features=np.arange(12, dtype=np.float32).reshape(4, 3),
        probabilities=np.full((4, 2), 0.5, dtype=np.float32),
        labels=np.array([0, 1, 0, 1]),
        sample_ids=np.array(["a", "b", "c", "d"]),
        provenance={
            "dataset": "cifar10",
            "split": "train",
            "extractor": "tf_fid_inception_v3",
            "weights_sha256": "abc",
            "evaluator_version": 1,
        },
    )


def test_feature_cache_round_trip_preserves_arrays_and_provenance(tmp_path) -> None:
    path = tmp_path / "features.npz"
    expected = _cache()

    save_feature_cache(path, expected)
    actual = load_feature_cache(path, expected_provenance=expected.provenance)

    assert np.array_equal(actual.features, expected.features)
    assert np.array_equal(actual.probabilities, expected.probabilities)
    assert np.array_equal(actual.labels, expected.labels)
    assert np.array_equal(actual.sample_ids, expected.sample_ids)
    assert actual.provenance == expected.provenance
    assert actual.fingerprint == cache_fingerprint(expected.provenance)


def test_cache_fingerprint_is_stable_to_mapping_order() -> None:
    first = {"dataset": "cifar10", "split": "train"}
    second = {"split": "train", "dataset": "cifar10"}

    assert cache_fingerprint(first) == cache_fingerprint(second)


def test_loading_rejects_provenance_mismatch(tmp_path) -> None:
    path = tmp_path / "features.npz"
    save_feature_cache(path, _cache())

    with pytest.raises(ValueError, match="fingerprint"):
        load_feature_cache(path, expected_provenance={"dataset": "cifar100"})


def test_save_rejects_misaligned_arrays(tmp_path) -> None:
    cache = _cache()
    cache.labels = cache.labels[:-1]

    with pytest.raises(ValueError, match="aligned"):
        save_feature_cache(tmp_path / "features.npz", cache)


def test_loading_rejects_corrupted_fingerprint(tmp_path) -> None:
    path = tmp_path / "features.npz"
    cache = _cache()
    np.savez_compressed(
        path,
        features=cache.features,
        probabilities=cache.probabilities,
        labels=cache.labels,
        sample_ids=cache.sample_ids,
        provenance_json=np.asarray(json.dumps(cache.provenance)),
        fingerprint=np.asarray("wrong"),
    )

    with pytest.raises(ValueError, match="corrupted"):
        load_feature_cache(path)
