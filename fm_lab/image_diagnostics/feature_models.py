"""Optional learned feature extractors for image datasets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from fm_lab.image_diagnostics.config import FeatureConfig
from fm_lab.image_diagnostics.save_utils import OptionalDependencyError

LOGGER = logging.getLogger("fm_lab.image_diagnostics")


class ImageFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, images: list[Any]) -> np.ndarray:
        """Return one feature row per image."""


def load_feature_model(config: FeatureConfig) -> ImageFeatureExtractor:
    """Load the optional learned feature extractor selected in the config."""

    if config.mode == "dinov2":
        return DINOv2FeatureExtractor(config)
    raise ValueError(f"Feature mode does not use a learned model: {config.mode}")


class DINOv2FeatureExtractor(ImageFeatureExtractor):
    """DINOv2 CLS-token image features."""

    def __init__(self, config: FeatureConfig) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise OptionalDependencyError(
                "DINOv2 features require Transformers. Install with: "
                'python -m pip install -e ".[image-embeddings]"'
            ) from exc

        self.torch = torch
        self.device = _resolve_device(config.device, torch)
        self.dtype = _resolve_dtype(config.dtype, self.device, torch)
        self.processor = AutoImageProcessor.from_pretrained(config.repo_id)
        self.model = AutoModel.from_pretrained(config.repo_id, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, images: list[Any]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        moved = {}
        for key, value in dict(inputs).items():
            value = value.to(self.device)
            if value.is_floating_point():
                value = value.to(self.dtype)
            moved[key] = value
        with self.torch.inference_mode():
            outputs = self.model(**moved)
        features = outputs.last_hidden_state[:, 0]
        return features.detach().float().cpu().numpy()


def _resolve_device(requested: str, torch: Any) -> Any:
    selected = requested.lower()
    if selected == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
    if selected.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA is unavailable; falling back to CPU.")
        selected = "cpu"
    if selected == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        LOGGER.warning("MPS is unavailable; falling back to CPU.")
        selected = "cpu"
    return torch.device(selected)


def _resolve_dtype(requested: str, device: Any, torch: Any) -> Any:
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    if device.type == "mps" and dtype == torch.bfloat16:
        return torch.float32
    return dtype
