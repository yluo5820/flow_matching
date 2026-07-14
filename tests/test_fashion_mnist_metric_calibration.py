import numpy as np
import torch
from torch import nn

from fm_lab.evaluation.cache import FeatureCache
from fm_lab.evaluation.features import extract_classifier_features
from fm_lab.evaluation.report import (
    evaluate_feature_caches,
    evaluate_reference_calibration,
)


class TinyFashionEvaluator(nn.Module):
    def forward(self, images: torch.Tensor):
        flat = images.flatten(1)
        features = torch.stack(
            (flat.mean(dim=1), flat.std(dim=1), flat[:, 0], flat[:, -1]),
            dim=1,
        )
        probabilities = torch.full((len(images), 10), 0.01, device=images.device)
        probabilities[:, 0] = 0.91
        return features, probabilities


def _reference_cache() -> FeatureCache:
    labels = np.repeat(np.arange(10), 4)
    features = np.stack((labels, np.tile(np.arange(4), 10)), axis=1).astype(np.float32)
    return FeatureCache(
        features=features,
        probabilities=np.eye(10, dtype=np.float32)[labels],
        labels=labels,
        sample_ids=np.asarray([f"real-{index}" for index in range(len(labels))]),
        provenance={"dataset": "fashion_mnist", "extractor": "tiny"},
    )


def test_reference_calibration_is_stratified_and_deterministic() -> None:
    first = evaluate_reference_calibration(
        _reference_cache(),
        seed=7,
        kid_subsets=1,
        kid_subset_size=20,
        recall_k=1,
        inception_splits=1,
    )
    second = evaluate_reference_calibration(
        _reference_cache(),
        seed=7,
        kid_subsets=1,
        kid_subset_size=20,
        recall_k=1,
        inception_splits=1,
    )

    assert first == second
    assert first["samples_per_half"] == 20
    assert set(first["metrics"]) == {"fid", "kid", "recall", "inception_score"}


def test_corruption_worsens_fid_and_does_not_improve_recall() -> None:
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(10), 4)
    images = np.empty((40, 1, 28, 28), dtype=np.float32)
    for index, label in enumerate(labels):
        images[index] = label / 9.0 + (index % 4) * 0.01
    corrupted = np.clip(images + rng.normal(0.0, 0.5, images.shape), 0.0, 1.0)
    evaluator = TinyFashionEvaluator()

    real = _extract(images, labels, evaluator, "real")
    clean = _extract(images.copy(), labels, evaluator, "clean")
    noisy = _extract(corrupted, labels, evaluator, "noisy")
    options = {
        "class_counts": [6000, 3596, 2156, 1292, 774, 464, 278, 166, 100, 60],
        "repeats": 1,
        "overall_samples": 40,
        "kid_subsets": 1,
        "kid_subset_size": 20,
        "recall_k": 2,
        "inception_splits": 1,
    }

    clean_report = evaluate_feature_caches(clean, real, **options)
    noisy_report = evaluate_feature_caches(noisy, real, **options)

    assert clean_report["metrics"]["fid"]["mean"] == 0.0
    assert noisy_report["metrics"]["fid"]["mean"] > 0.0
    assert noisy_report["metrics"]["recall"]["mean"] <= clean_report["metrics"]["recall"][
        "mean"
    ]


def _extract(
    images: np.ndarray,
    labels: np.ndarray,
    evaluator: nn.Module,
    split: str,
) -> FeatureCache:
    return extract_classifier_features(
        images,
        labels=labels,
        sample_ids=np.asarray([f"{split}-{index}" for index in range(len(labels))]),
        model=evaluator,
        batch_size=8,
        device=torch.device("cpu"),
        input_range=(0.0, 1.0),
        provenance={"dataset": "fashion_mnist", "split": split},
    )
