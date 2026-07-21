"""Thin fm_lab interfaces around the vendored official ImbDiff-CM code.

This module deliberately delegates the CM model, training loss, endpoint
transfer, and DDIM update to the authors' released implementation under
``third_party/ImbDiff-CM``.  The adapter only translates flat fm_lab tensors
and context dictionaries to the official NCHW/y/use_cm interface.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class OfficialImbDiffCMComponents:
    """Classes imported from the exact vendored release."""

    cm_unet: type[nn.Module]
    cm_trainer: type[nn.Module]
    cm_sampler: type[nn.Module]


@lru_cache(maxsize=1)
def load_official_imbdiff_cm_components() -> OfficialImbDiffCMComponents:
    """Load the release package without copying or rewriting its equations."""

    repository_root = Path(__file__).resolve().parents[2]
    official_root = repository_root / "third_party" / "ImbDiff-CM"
    package_root = official_root / "imbdiff_cm"
    if not package_root.is_dir():
        raise FileNotFoundError(
            "The vendored ImbDiff-CM release is missing. Expected "
            f"{package_root}."
        )
    root_text = str(official_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    model_module = importlib.import_module("imbdiff_cm.model.model_cm")
    diffusion_module = importlib.import_module("imbdiff_cm.diffusion_cm")
    return OfficialImbDiffCMComponents(
        cm_unet=model_module.UNet_CM,
        cm_trainer=diffusion_module.GaussianDiffusionTrainer,
        # tools/sample_images.py imports this exact sampler for CM.
        cm_sampler=diffusion_module.GaussianDiffusionSamplerOld,
    )


_MISSING = object()


class OfficialImbDiffCMUNet(nn.Module):
    """Exact released ``UNet_CM`` behind the fm_lab model interface."""

    is_class_conditional = True
    is_official_imbdiff_cm = True

    def __init__(
        self,
        *,
        dim: int,
        image_shape: Sequence[int] = (3, 32, 32),
        timesteps: int = 1000,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 2, 2, 2),
        attention_levels: Sequence[int] = (1,),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int = 100,
        rank: int = 0,
        rank_ratio: float = 0.1,
        adapter_scale: float = 0.5,
        capacity_parts: Sequence[str] = ("up",),
        lora_alpha: float = 1.0,
        lora_mode: str = "ratio",
    ) -> None:
        super().__init__()
        shape = tuple(int(value) for value in image_shape)
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError("Official ImbDiff-CM requires RGB image_shape [3, H, W].")
        if int(dim) != shape[0] * shape[1] * shape[2]:
            raise ValueError(f"dim={dim} does not match image_shape={shape}.")
        if int(num_classes) < 1:
            raise ValueError("num_classes must be positive.")
        self.dim = int(dim)
        self.image_shape = shape
        self.num_classes = int(num_classes)
        self.num_timesteps = int(timesteps)
        self.capacity_parts = tuple(str(value) for value in capacity_parts)
        self.rank = int(rank)
        self.rank_ratio = float(rank_ratio)
        self.adapter_scale = float(adapter_scale)

        components = load_official_imbdiff_cm_components()
        self.network = components.cm_unet(
            T=self.num_timesteps,
            ch=int(base_channels),
            ch_mult=[int(value) for value in channel_multipliers],
            attn=[int(value) for value in attention_levels],
            num_res_blocks=int(num_res_blocks),
            dropout=float(dropout),
            cond=True,
            augm=False,
            num_class=self.num_classes,
            r=self.rank,
            lora_alpha=float(lora_alpha),
            r_ratio=self.rank_ratio,
            scaling=self.adapter_scale,
            lora_mode=str(lora_mode),
            lora_part=list(self.capacity_parts),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, Any] | None = None,
        *,
        y: torch.Tensor | None | object = _MISSING,
        augm: torch.Tensor | None = None,
        use_cm: bool = True,
    ) -> torch.Tensor:
        input_was_flat = x.ndim == 2
        if input_was_flat:
            image = x.reshape(x.shape[0], *self.image_shape)
        elif x.ndim == 4 and tuple(x.shape[1:]) == self.image_shape:
            image = x
        else:
            raise ValueError(
                "Official ImbDiff-CM input must be flat [N, D] or NCHW with "
                f"shape {self.image_shape}."
            )

        labels: torch.Tensor | None
        if y is _MISSING:
            # The released samplers pass y as the third positional argument,
            # while fm_lab passes a context dictionary.
            if isinstance(context, torch.Tensor):
                labels = context
                context = None
            elif isinstance(context, dict) and context.get("class_labels") is not None:
                labels = context["class_labels"].to(device=x.device, dtype=torch.long)
            elif context is None:
                labels = None
            else:
                raise ValueError("Official ImbDiff-CM requires class labels or y=None.")
        else:
            labels = y  # type: ignore[assignment]

        if labels is not None:
            labels = labels.to(device=x.device, dtype=torch.long)
            dropped = labels < 0
            if bool(dropped.all()):
                labels = None
            elif bool(dropped.any()):
                raise ValueError(
                    "The official model represents unconditional conditioning as y=None; "
                    "mixed per-sample label dropout is unsupported. Use batch dropout."
                )

        active_capacity = bool(use_cm)
        if isinstance(context, dict) and "use_capacity" in context:
            active_capacity = bool(context["use_capacity"])
        output = self.network(
            image,
            t.to(device=x.device, dtype=torch.long),
            y=labels,
            augm=augm,
            use_cm=active_capacity,
        )
        return output.reshape(x.shape[0], -1) if input_was_flat else output

    def capacity_metadata(self) -> dict[str, object]:
        adapter_layers = sum(
            module.__class__.__name__ == "Conv2d_LoRA" and getattr(module, "r", 0) > 0
            for module in self.network.modules()
        )
        return {
            "enabled": adapter_layers > 0,
            "rank": self.rank,
            "rank_ratio": self.rank_ratio,
            "adapter_scale": self.adapter_scale,
            "parts": list(self.capacity_parts),
            "adapter_layers": adapter_layers,
            "implementation": "official_imbdiff_cm",
        }


class OfficialImbDiffCMObjective:
    """Exact released CM trainer exposed as an fm_lab objective."""

    name = "official_imbdiff_cm"
    prediction_type = "epsilon"
    model_output = "source"
    loss_space = "source"
    is_discrete_diffusion = True
    uses_official_warmup = True
    uses_official_data_batches = True

    def __init__(
        self,
        *,
        class_counts: Sequence[int],
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cfg: bool = True,
        transfer_x0: bool = True,
        transfer_tr_tau: bool = False,
        transfer_mode: str = "t2h",
        transfer_tau: float = 1.0,
        consistency_weight: float = 1.0,
        diversity_weight: float = 0.2,
        image_shape: Sequence[int] = (3, 32, 32),
    ) -> None:
        counts = tuple(int(value) for value in class_counts)
        if not counts or any(value <= 0 for value in counts):
            raise ValueError("Official ImbDiff-CM requires positive class_counts.")
        self.class_counts = counts
        self.timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.cfg = bool(cfg)
        self.transfer_x0 = bool(transfer_x0)
        self.transfer_tr_tau = bool(transfer_tr_tau)
        self.transfer_mode = str(transfer_mode)
        self.transfer_tau = float(transfer_tau)
        self.consistency_weight = float(consistency_weight)
        self.diversity_weight = float(diversity_weight)
        self.image_shape = tuple(int(value) for value in image_shape)
        self._trainer: nn.Module | None = None
        self._trainer_model: nn.Module | None = None
        self._trainer_device: torch.device | None = None

    def __call__(
        self,
        *,
        model: nn.Module,
        path: Any,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        compute_diagnostics: bool = True,
        class_labels: torch.Tensor | None = None,
        original_class_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del path, x0, t, compute_diagnostics, class_labels
        if original_class_labels is None:
            raise ValueError("Official ImbDiff-CM requires original class labels.")
        clean = x1.reshape(x1.shape[0], *self.image_shape)
        trainer = self._trainer_for(model, clean.device)
        loss = trainer(
            clean,
            original_class_labels.to(device=clean.device, dtype=torch.long),
            augm=None,
            uncond_flag_out=False,
        )
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError("The official CM trainer must return one scalar loss tensor.")
        value = float(loss.detach().cpu())
        return loss, {"official_imbdiff_cm_loss": value, "loss": value}

    def _trainer_for(self, model: nn.Module, device: torch.device) -> nn.Module:
        if (
            self._trainer is not None
            and self._trainer_model is model
            and self._trainer_device == device
        ):
            return self._trainer
        components = load_official_imbdiff_cm_components()
        probabilities = torch.tensor(self.class_counts, dtype=torch.float32)
        probabilities = probabilities / probabilities.sum()
        label_weight = torch.pow(
            probabilities.unsqueeze(1) @ probabilities.unsqueeze(0),
            self.transfer_tau,
        )
        trainer = components.cm_trainer(
            model=model,
            beta_1=self.beta_start,
            beta_T=self.beta_end,
            T=self.timesteps,
            dataset=None,
            num_class=len(self.class_counts),
            cfg=self.cfg,
            weight=probabilities.unsqueeze(0),
            transfer_x0=self.transfer_x0,
            transfer_tr_tau=self.transfer_tr_tau,
            transfer_mode=self.transfer_mode,
            label_weight_tr=label_weight,
            w_con=self.consistency_weight,
            w_div=self.diversity_weight,
        ).to(device)
        self._trainer = trainer
        self._trainer_model = model
        self._trainer_device = device
        return trainer

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": "vendored_official_release",
            "prediction_type": self.prediction_type,
            "timesteps": self.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "cfg": self.cfg,
            "class_counts": list(self.class_counts),
            "transfer": {
                "transfer_x0": self.transfer_x0,
                "transfer_tr_tau": self.transfer_tr_tau,
                "transfer_mode": self.transfer_mode,
                "transfer_tau": self.transfer_tau,
            },
            "cm": {
                "consistency_weight": self.consistency_weight,
                "diversity_weight": self.diversity_weight,
            },
        }


@torch.no_grad()
def sample_official_imbdiff_cm(
    *,
    model: nn.Module,
    initial_noise: torch.Tensor,
    class_labels: torch.Tensor,
    timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    variance: str = "fixedlarge",
    omega: float = 1.5,
    method: str = "ddim",
    ddim_skip: int = 20,
    image_shape: Sequence[int] = (3, 32, 32),
) -> torch.Tensor:
    """Invoke the sampler used by the release's ``tools/sample_images.py``."""

    shape = tuple(int(value) for value in image_shape)
    flat_input = initial_noise.ndim == 2
    noise = (
        initial_noise.reshape(initial_noise.shape[0], *shape)
        if flat_input
        else initial_noise
    )
    if noise.ndim != 4 or tuple(noise.shape[1:]) != shape:
        raise ValueError(f"initial_noise must match image_shape={shape}.")
    if class_labels.shape != (noise.shape[0],):
        raise ValueError("class_labels must match the sampling batch.")
    components = load_official_imbdiff_cm_components()
    sampler = components.cm_sampler(
        model,
        float(beta_start),
        float(beta_end),
        int(timesteps),
        img_size=shape[-1],
        mean_type="epsilon",
        var_type=str(variance),
        w=float(omega),
        cond=True,
    ).to(noise.device)
    was_training = model.training
    model.eval()
    try:
        samples = sampler(
            noise,
            class_labels.to(device=noise.device, dtype=torch.long),
            method=str(method),
            skip=int(ddim_skip),
        )
    finally:
        model.train(was_training)
    return samples.reshape(samples.shape[0], -1) if flat_input else samples
