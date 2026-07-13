import numpy as np
import pytest
import torch
from torch import nn

from fm_lab.evaluation.features import extract_inception_features
from fm_lab.evaluation.inception import ReferenceInceptionV3, sha256_file


class FakeInception(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[torch.Tensor] = []

    def forward(self, images: torch.Tensor):
        self.batches.append(images.detach().cpu())
        means = images.mean(dim=(1, 2, 3))
        features = torch.stack((means, means + 1.0), dim=1)
        probabilities = torch.stack((means, 1.0 - means), dim=1)
        return features, probabilities


def test_extract_features_converts_minus_one_one_and_batches() -> None:
    images = np.stack(
        [
            np.full((3, 2, 2), -1.0, dtype=np.float32),
            np.full((3, 2, 2), 1.0, dtype=np.float32),
            np.zeros((3, 2, 2), dtype=np.float32),
        ]
    )
    model = FakeInception()

    result = extract_inception_features(
        images,
        labels=np.array([0, 1, 0]),
        sample_ids=np.array(["a", "b", "c"]),
        model=model,
        batch_size=2,
        device=torch.device("cpu"),
        input_range=(-1.0, 1.0),
        image_shape=(3, 2, 2),
        provenance={"dataset": "fake"},
    )

    assert len(model.batches) == 2
    assert model.batches[0].min() == 0.0
    assert model.batches[0].max() == 1.0
    assert np.allclose(result.features[:, 0], [0.0, 1.0, 0.5])
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.provenance["input_range"] == [-1.0, 1.0]


def test_extract_features_accepts_flattened_cifar_images() -> None:
    images = np.zeros((2, 3 * 32 * 32), dtype=np.float32)

    result = extract_inception_features(
        images,
        labels=np.array([0, 1]),
        sample_ids=np.array(["0", "1"]),
        model=FakeInception(),
        batch_size=2,
        device=torch.device("cpu"),
        image_shape=(3, 32, 32),
        provenance={"dataset": "fake"},
    )

    assert result.features.shape == (2, 2)


def test_reference_inception_requires_exact_weight_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="TensorFlow-FID"):
        ReferenceInceptionV3(tmp_path / "missing.pth")


def test_sha256_file_hashes_weight_artifact(tmp_path) -> None:
    path = tmp_path / "weights.pth"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
