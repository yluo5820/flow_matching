from pathlib import Path

import numpy as np
import pytest
import torch

from fm_lab.diagnostics.mnist_eval import (
    FashionMNISTFeatureEvaluator,
    MNISTClassifier,
    _load_classifier_payload,
    _state_dict_sha256,
)
from fm_lab.evaluation.features import extract_classifier_features


def _metadata(**overrides):
    values = {
        "dataset": "fashion_mnist",
        "normalize": "minus_one_one",
        "test_accuracy": 0.93,
        "checkpoint_path": "classifier.pt",
        "weights_sha256": "abc123",
        "state_dict_sha256": "state123",
        "evaluator_version": 1,
        "architecture": "mnist_classifier_v1",
        "feature_dimension": 128,
        "class_order": list(range(10)),
    }
    values.update(overrides)
    return values


def test_mnist_classifier_exposes_penultimate_features() -> None:
    classifier = MNISTClassifier()
    images = torch.randn(5, 28 * 28)

    features = classifier.forward_features(images)
    logits = classifier(images)

    assert features.shape == (5, 128)
    assert logits.shape == (5, 10)


def test_fashion_evaluator_returns_features_and_probabilities() -> None:
    evaluator = FashionMNISTFeatureEvaluator(
        MNISTClassifier(),
        metadata=_metadata(),
        normalize="minus_one_one",
        minimum_accuracy=0.9,
    )

    features, probabilities = evaluator(torch.randn(4, 1, 28, 28))

    assert features.shape == (4, 128)
    assert probabilities.shape == (4, 10)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(4))
    assert evaluator.provenance["extractor"] == "fashion_mnist_classifier"


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (_metadata(dataset="mnist"), "dataset"),
        (_metadata(normalize="zero_one"), "normalization"),
        (_metadata(test_accuracy=0.85), "accuracy"),
        (_metadata(architecture="other"), "architecture"),
    ],
)
def test_fashion_evaluator_rejects_incompatible_metadata(metadata, message) -> None:
    with pytest.raises(ValueError, match=message):
        FashionMNISTFeatureEvaluator(
            MNISTClassifier(),
            metadata=metadata,
            normalize="minus_one_one",
            minimum_accuracy=0.9,
        )


def test_classifier_feature_extraction_preserves_classifier_normalization() -> None:
    evaluator = FashionMNISTFeatureEvaluator(
        MNISTClassifier(),
        metadata=_metadata(),
        normalize="minus_one_one",
        minimum_accuracy=0.9,
    )
    images = np.stack(
        [
            np.full((1, 28, 28), -1.0, dtype=np.float32),
            np.full((1, 28, 28), 1.0, dtype=np.float32),
        ]
    )

    result = extract_classifier_features(
        images,
        labels=np.array([0, 1]),
        sample_ids=np.array(["a", "b"]),
        model=evaluator,
        batch_size=2,
        device=torch.device("cpu"),
        input_range=(-1.0, 1.0),
        image_shape=(1, 28, 28),
        provenance=evaluator.provenance,
    )

    assert result.features.shape == (2, 128)
    assert result.probabilities.shape == (2, 10)
    assert result.provenance["preprocessing"] == "clamp_to_classifier_input_range"
    assert result.provenance["image_shape"] == [1, 28, 28]


def test_classifier_checkpoint_rejects_missing_contract_metadata(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    classifier = MNISTClassifier()
    torch.save({"model_state_dict": classifier.state_dict()}, path)

    with pytest.raises(ValueError, match="missing evaluator metadata"):
        _load_classifier_payload(
            classifier=MNISTClassifier(),
            checkpoint_path=path,
            device=torch.device("cpu"),
            dataset="fashion_mnist",
            normalize="minus_one_one",
        )


def test_classifier_checkpoint_validates_state_dict_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "classifier.pt"
    classifier = MNISTClassifier()
    payload = {
        "model_state_dict": classifier.state_dict(),
        "dataset": "fashion_mnist",
        "normalize": "minus_one_one",
        "evaluator_version": 1,
        "architecture": "mnist_classifier_v1",
        "feature_dimension": 128,
        "class_order": list(range(10)),
        "held_out_accuracy": 0.93,
        "state_dict_sha256": "incorrect",
        "steps": 100,
    }
    torch.save(payload, path)

    with pytest.raises(ValueError, match="state_dict_sha256"):
        _load_classifier_payload(
            classifier=MNISTClassifier(),
            checkpoint_path=path,
            device=torch.device("cpu"),
            dataset="fashion_mnist",
            normalize="minus_one_one",
        )

    payload["state_dict_sha256"] = _state_dict_sha256(classifier.state_dict())
    torch.save(payload, path)
    loaded = _load_classifier_payload(
        classifier=MNISTClassifier(),
        checkpoint_path=path,
        device=torch.device("cpu"),
        dataset="fashion_mnist",
        normalize="minus_one_one",
    )
    assert loaded["held_out_accuracy"] == 0.93
