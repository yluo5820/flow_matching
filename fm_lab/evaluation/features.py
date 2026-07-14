"""Batched feature extraction into versioned evaluator caches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from fm_lab.evaluation.cache import FeatureCache


@torch.no_grad()
def extract_inception_features(
    images: np.ndarray | torch.Tensor,
    *,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    input_range: tuple[float, float] = (0.0, 1.0),
    image_shape: Sequence[int] = (3, 32, 32),
    provenance: dict[str, Any],
) -> FeatureCache:
    return _extract_features(
        images,
        labels=labels,
        sample_ids=sample_ids,
        model=model,
        batch_size=batch_size,
        device=device,
        input_range=input_range,
        image_shape=image_shape,
        provenance=provenance,
        rescale_to_zero_one=True,
        preprocessing="scale_to_0_1_then_tf_fid_inception",
    )


@torch.no_grad()
def extract_classifier_features(
    images: np.ndarray | torch.Tensor,
    *,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    input_range: tuple[float, float],
    image_shape: Sequence[int] = (1, 28, 28),
    provenance: dict[str, Any],
) -> FeatureCache:
    """Extract domain-classifier features without changing normalization."""

    return _extract_features(
        images,
        labels=labels,
        sample_ids=sample_ids,
        model=model,
        batch_size=batch_size,
        device=device,
        input_range=input_range,
        image_shape=image_shape,
        provenance=provenance,
        rescale_to_zero_one=False,
        preprocessing="clamp_to_classifier_input_range",
    )


def _extract_features(
    images: np.ndarray | torch.Tensor,
    *,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    input_range: tuple[float, float],
    image_shape: Sequence[int],
    provenance: dict[str, Any],
    rescale_to_zero_one: bool,
    preprocessing: str,
) -> FeatureCache:
    if batch_size < 1:
        raise ValueError("Feature extraction batch_size must be positive.")
    tensor = torch.as_tensor(images, dtype=torch.float32)
    shape = tuple(int(value) for value in image_shape)
    if tensor.ndim == 2:
        if tensor.shape[1] != int(np.prod(shape)):
            raise ValueError("Flattened image dimension does not match image_shape.")
        tensor = tensor.reshape(tensor.shape[0], *shape)
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != shape:
        raise ValueError("Images must have shape [N, C, H, W] or flattened equivalent.")
    low, high = (float(value) for value in input_range)
    if not low < high:
        raise ValueError("input_range must be increasing.")
    tensor = tensor.clamp(low, high)
    if rescale_to_zero_one:
        tensor = (tensor - low) / (high - low)

    features = []
    probabilities = []
    was_training = model.training
    model.to(device).eval()
    try:
        for offset in range(0, len(tensor), batch_size):
            batch_features, batch_probabilities = model(
                tensor[offset : offset + batch_size].to(device)
            )
            features.append(batch_features.detach().cpu().numpy())
            probabilities.append(batch_probabilities.detach().cpu().numpy())
    finally:
        model.train(was_training)
    resolved_provenance = dict(provenance)
    resolved_provenance.update(
        {
            "input_range": [low, high],
            "image_shape": list(shape),
            "preprocessing": preprocessing,
        }
    )
    return FeatureCache(
        features=np.concatenate(features, axis=0),
        probabilities=np.concatenate(probabilities, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=str),
        provenance=resolved_provenance,
    )
