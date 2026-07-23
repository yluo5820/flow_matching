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
from torch.nn import functional as F


@dataclass(frozen=True)
class OfficialImbDiffCMComponents:
    """Classes imported from the exact vendored release."""

    unet: type[nn.Module]
    trainer: type[nn.Module]
    sampler: type[nn.Module]
    cm_unet: type[nn.Module]
    cm_trainer: type[nn.Module]
    cm_sampler: type[nn.Module]


@dataclass(frozen=True)
class OfficialImbDiffCMTerms:
    """Graph-connected CM terms from one exact pair of model evaluations."""

    noisy: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    conditioned_labels: torch.Tensor | None
    capacity_on: torch.Tensor
    capacity_off: torch.Tensor
    base_per_sample: torch.Tensor
    distance_per_sample: torch.Tensor
    coefficient_per_sample: torch.Tensor
    consistency_per_sample: torch.Tensor
    diversity_per_sample: torch.Tensor
    total_per_sample: torch.Tensor
    loss: torch.Tensor
    dropout_mode: str
    capacity_on_enabled: bool
    capacity_off_enabled: bool
    unconditional_batch: bool


# Backward-compatible diagnostic name used by the completed checkpoint probe.
OfficialImbDiffCMProbeTerms = OfficialImbDiffCMTerms


_CM_DROPOUT_MODES = frozenset({"independent", "paired", "disabled"})


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
    model_module = importlib.import_module("imbdiff_cm.model.model")
    cm_model_module = importlib.import_module("imbdiff_cm.model.model_cm")
    diffusion_module = importlib.import_module("imbdiff_cm.diffusion")
    cm_diffusion_module = importlib.import_module("imbdiff_cm.diffusion_cm")
    return OfficialImbDiffCMComponents(
        unet=model_module.UNet,
        trainer=diffusion_module.GaussianDiffusionTrainer,
        # tools/sample_images.py imports this exact sampler for OC.
        sampler=diffusion_module.GaussianDiffusionSamplerOld,
        cm_unet=cm_model_module.UNet_CM,
        cm_trainer=cm_diffusion_module.GaussianDiffusionTrainer,
        # tools/sample_images.py imports this exact sampler for CM.
        cm_sampler=cm_diffusion_module.GaussianDiffusionSamplerOld,
    )


_MISSING = object()


def _official_image_and_labels(
    x: torch.Tensor,
    *,
    image_shape: tuple[int, ...],
    context: dict[str, Any] | torch.Tensor | None,
    y: torch.Tensor | None | object,
) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
    input_was_flat = x.ndim == 2
    if input_was_flat:
        image = x.reshape(x.shape[0], *image_shape)
    elif x.ndim == 4 and tuple(x.shape[1:]) == image_shape:
        image = x
    else:
        raise ValueError(
            "Official ImbDiff input must be flat [N, D] or NCHW with "
            f"shape {image_shape}."
        )

    labels: torch.Tensor | None
    if y is _MISSING:
        if isinstance(context, torch.Tensor):
            labels = context
        elif isinstance(context, dict) and context.get("class_labels") is not None:
            labels = context["class_labels"]
        elif context is None:
            labels = None
        else:
            raise ValueError("Official ImbDiff requires class labels or y=None.")
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
    return image, labels, input_was_flat


class OfficialImbDiffUNet(nn.Module):
    """Exact released standard ``UNet`` used by DDPM, CBDM, and OC."""

    is_class_conditional = True
    is_official_imbdiff = True
    is_official_imbdiff_cm = False

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
    ) -> None:
        super().__init__()
        shape = tuple(int(value) for value in image_shape)
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError("Official ImbDiff requires RGB image_shape [3, H, W].")
        if int(dim) != shape[0] * shape[1] * shape[2]:
            raise ValueError(f"dim={dim} does not match image_shape={shape}.")
        if int(num_classes) < 1:
            raise ValueError("num_classes must be positive.")
        self.dim = int(dim)
        self.image_shape = shape
        self.num_classes = int(num_classes)
        self.num_timesteps = int(timesteps)

        components = load_official_imbdiff_cm_components()
        self.network = components.unet(
            T=self.num_timesteps,
            ch=int(base_channels),
            ch_mult=[int(value) for value in channel_multipliers],
            attn=[int(value) for value in attention_levels],
            num_res_blocks=int(num_res_blocks),
            dropout=float(dropout),
            cond=True,
            augm=False,
            num_class=self.num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: dict[str, Any] | torch.Tensor | None = None,
        *,
        y: torch.Tensor | None | object = _MISSING,
        augm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image, labels, input_was_flat = _official_image_and_labels(
            x,
            image_shape=self.image_shape,
            context=context,
            y=y,
        )
        output = self.network(
            image,
            t.to(device=x.device, dtype=torch.long),
            y=labels,
            augm=augm,
        )
        # The released trainers flatten predictions with ``view``. Preserve
        # that contract when fm_lab runs the U-Net in channels-last format.
        output = output.contiguous()
        return output.reshape(x.shape[0], -1) if input_was_flat else output

    def capacity_metadata(self) -> dict[str, object]:
        return {
            "enabled": False,
            "adapter_layers": 0,
            "implementation": "official_imbdiff",
        }


class OfficialImbDiffCMUNet(nn.Module):
    """Exact released ``UNet_CM`` behind the fm_lab model interface."""

    is_class_conditional = True
    is_official_imbdiff = True
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
        context: dict[str, Any] | torch.Tensor | None = None,
        *,
        y: torch.Tensor | None | object = _MISSING,
        augm: torch.Tensor | None = None,
        use_cm: bool = True,
    ) -> torch.Tensor:
        image, labels, input_was_flat = _official_image_and_labels(
            x,
            image_shape=self.image_shape,
            context=context,
            y=y,
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
        # diffusion_cm.py flattens h1/h2 with ``view``; channels-last outputs
        # therefore need an explicit contiguous interface boundary.
        output = output.contiguous()
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


class OfficialImbDiffObjective:
    """Paper/release discrete objectives behind one controlled interface.

    DDPM and OC delegate to the released standard trainer. Released CM,
    pure CM, and the capacity-only control delegate to the released CM
    trainer. CBDM is implemented directly from Eq. (4) of Qin et al. because
    ImbDiff-CM does not vendor or release a CBDM trainer.
    """

    name = "official_imbdiff"
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
        method: str,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cfg: bool = True,
        transfer_x0: bool | None = None,
        transfer_tr_tau: bool = False,
        transfer_mode: str = "t2h",
        transfer_tau: float = 1.0,
        consistency_weight: float = 1.0,
        diversity_weight: float = 0.2,
        cbdm_target_distribution: str = "train",
        cbdm_tau: float = 0.001,
        cbdm_gamma: float = 0.25,
        image_shape: Sequence[int] = (3, 32, 32),
    ) -> None:
        counts = tuple(int(value) for value in class_counts)
        if not counts or any(value <= 0 for value in counts):
            raise ValueError("Official ImbDiff requires positive class_counts.")
        normalized_method = str(method).lower().replace("-", "_")
        aliases = {
            "baseline": "ddpm",
            "cm": "released_cm",
            "cm_released": "released_cm",
            "pure": "pure_cm",
            "cm_pure": "pure_cm",
            "capacity_only": "oc_capacity_only",
            "oc_capacity": "oc_capacity_only",
        }
        normalized_method = aliases.get(normalized_method, normalized_method)
        supported = {
            "ddpm",
            "cbdm",
            "oc",
            "released_cm",
            "pure_cm",
            "oc_capacity_only",
        }
        if normalized_method not in supported:
            raise ValueError(
                "Official ImbDiff method must be ddpm, cbdm, oc, released_cm, "
                "pure_cm, or oc_capacity_only."
            )
        self.method = normalized_method
        self.name = f"official_imbdiff_{self.method}"
        self.class_counts = counts
        self.timesteps = int(timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.cfg = bool(cfg)
        canonical_transfer = {
            "ddpm": False,
            "cbdm": False,
            "oc": True,
            "released_cm": True,
            "pure_cm": False,
            "oc_capacity_only": True,
        }[self.method]
        if transfer_x0 is not None and bool(transfer_x0) != canonical_transfer:
            raise ValueError(
                f"Method {self.method!r} requires transfer_x0={canonical_transfer}."
            )
        self.transfer_x0 = canonical_transfer
        self.transfer_tr_tau = bool(transfer_tr_tau)
        self.transfer_mode = str(transfer_mode)
        self.transfer_tau = float(transfer_tau)
        self.consistency_weight = (
            0.0 if self.method == "oc_capacity_only" else float(consistency_weight)
        )
        self.diversity_weight = (
            0.0 if self.method == "oc_capacity_only" else float(diversity_weight)
        )
        if self.consistency_weight < 0 or self.diversity_weight < 0:
            raise ValueError("CM weights must be non-negative.")
        self.cbdm_target_distribution = str(cbdm_target_distribution).lower()
        if self.cbdm_target_distribution not in {"train", "sqrt", "uniform"}:
            raise ValueError(
                "CBDM target_distribution must be 'train', 'sqrt', or 'uniform'."
            )
        self.cbdm_tau = float(cbdm_tau)
        self.cbdm_gamma = float(cbdm_gamma)
        if self.cbdm_tau < 0 or self.cbdm_gamma < 0:
            raise ValueError("CBDM tau and gamma must be non-negative.")
        self.image_shape = tuple(int(value) for value in image_shape)
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps).double()
        self._sqrt_alpha_bars = torch.cumprod(1.0 - betas, dim=0).sqrt()
        self._sqrt_one_minus_alpha_bars = (
            1.0 - torch.cumprod(1.0 - betas, dim=0)
        ).sqrt()
        cbdm_counts = torch.tensor(self.class_counts, dtype=torch.float64)
        if self.cbdm_target_distribution == "sqrt":
            cbdm_counts = cbdm_counts.sqrt()
        elif self.cbdm_target_distribution == "uniform":
            cbdm_counts = torch.ones_like(cbdm_counts)
        self._cbdm_probabilities = (cbdm_counts / cbdm_counts.sum()).float()
        self._trainer: nn.Module | None = None
        self._trainer_model: nn.Module | None = None
        self._trainer_device: torch.device | None = None
        self._capture_training_terms = False
        self._last_training_terms: OfficialImbDiffCMTerms | None = None

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
        del path, x0, t, class_labels
        if original_class_labels is None:
            raise ValueError("Official ImbDiff requires original class labels.")
        model_uses_capacity = bool(getattr(model, "is_official_imbdiff_cm", False))
        if model_uses_capacity != self.uses_capacity_model:
            expected = "CM U-Net" if self.uses_capacity_model else "standard U-Net"
            raise ValueError(f"Method {self.method!r} requires the official {expected}.")
        clean = x1.reshape(x1.shape[0], *self.image_shape)
        labels = original_class_labels.to(device=clean.device, dtype=torch.long)
        if self.method == "cbdm":
            return self._cbdm_loss(model=model, clean=clean, labels=labels)
        if self.uses_capacity_model:
            terms = self.training_terms(
                model=model,
                clean=clean,
                labels=labels,
                dropout_mode="independent",
            )
            self._last_training_terms = (
                terms if self._capture_training_terms else None
            )
            loss = terms.loss
        else:
            trainer = self._trainer_for(model, clean.device)
            released_loss = trainer(
                clean,
                labels,
                augm=None,
                uncond_flag_out=False,
            )
            if isinstance(released_loss, tuple):
                if len(released_loss) != 2:
                    raise TypeError(
                        "The released standard trainer returned an invalid tuple."
                    )
                denoising, auxiliary = released_loss
                loss = denoising.mean() + auxiliary
            else:
                loss = released_loss
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise TypeError("The official trainer must resolve to one scalar loss tensor.")
        value = float(loss.detach().cpu())
        metrics = {
            "official_imbdiff_loss": value,
            f"official_imbdiff_{self.method}_loss": value,
            "loss": value,
        }
        if self.uses_capacity_model and compute_diagnostics:
            metrics.update(
                {
                    "cm_base_loss": float(terms.base_per_sample.detach().mean().cpu()),
                    "cm_auxiliary_loss": float(
                        (
                            terms.consistency_per_sample * self.consistency_weight
                            + terms.diversity_per_sample * self.diversity_weight
                        )
                        .detach()
                        .mean()
                        .cpu()
                    ),
                    "cm_branch_distance": float(
                        terms.distance_per_sample.detach().mean().cpu()
                    ),
                    "cm_coefficient_mean": float(
                        terms.coefficient_per_sample.detach().mean().cpu()
                    ),
                    "cm_unconditional_batch": float(terms.unconditional_batch),
                }
            )
        return loss, metrics

    def capture_next_training_terms(self) -> None:
        """Retain the next faithful CM graph for sparse in-loop diagnostics."""

        if not self.uses_capacity_model:
            raise ValueError("Training-term capture requires a CM capacity objective.")
        self._capture_training_terms = True
        self._last_training_terms = None

    def pop_captured_training_terms(self) -> OfficialImbDiffCMTerms:
        """Return and clear the graph retained by ``capture_next_training_terms``."""

        terms = self._last_training_terms
        self._capture_training_terms = False
        self._last_training_terms = None
        if terms is None:
            raise RuntimeError("The requested CM training terms were not captured.")
        return terms

    @property
    def uses_capacity_model(self) -> bool:
        return self.method in {"released_cm", "pure_cm", "oc_capacity_only"}

    @property
    def sampler_family(self) -> str:
        return "cm" if self.uses_capacity_model else "standard"

    def training_terms(
        self,
        *,
        model: nn.Module,
        clean: torch.Tensor,
        labels: torch.Tensor,
        dropout_mode: str = "independent",
    ) -> OfficialImbDiffCMTerms:
        """Sample and expose the exact graph-connected released-CM training terms.

        ``dropout_mode='independent'`` preserves the authors' two sequential
        forward calls and is the only mode used by the faithful objective.
        ``paired`` replays the first model RNG state for the general pass, while
        ``disabled`` temporarily evaluates both predictions with dropout off.
        """

        self._validate_cm_term_request(model, clean, labels)
        trainer = self._trainer_for(model, clean.device)
        batch_size = clean.shape[0]

        # Preserve Algorithm 1's stochastic operation order exactly.
        timesteps = torch.randint(
            self.timesteps,
            size=(batch_size,),
            device=clean.device,
        )
        noise = torch.randn_like(clean)
        signal, sigma = self._expanded_diffusion_coefficients(
            timesteps,
            clean,
        )
        noisy = signal * clean + sigma * noise

        conditioned_labels: torch.Tensor | None = labels
        unconditional_batch = False
        if self.cfg and bool(torch.rand(1)[0] < 0.1):
            conditioned_labels = None
            unconditional_batch = True

        capacity_on, capacity_off = self._cm_prediction_pair(
            model=model,
            noisy=noisy,
            timesteps=timesteps,
            conditioned_labels=conditioned_labels,
            dropout_mode=dropout_mode,
            capacity_on_enabled=True,
            capacity_off_enabled=False,
        )
        target = self._cm_target(
            trainer=trainer,
            noisy=noisy,
            clean=clean,
            labels=labels,
            timesteps=timesteps,
            noise=noise,
            signal=signal,
        )
        return self._assemble_cm_terms(
            trainer=trainer,
            noisy=noisy,
            target=target,
            labels=labels,
            timesteps=timesteps,
            conditioned_labels=conditioned_labels,
            capacity_on=capacity_on,
            capacity_off=capacity_off,
            dropout_mode=dropout_mode,
            capacity_on_enabled=True,
            capacity_off_enabled=False,
            unconditional_batch=unconditional_batch,
        )

    def probe_terms(
        self,
        *,
        model: nn.Module,
        clean: torch.Tensor,
        labels: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        transfer_seed: int | None = None,
        dropout_mode: str = "disabled",
        capacity_on_enabled: bool = True,
        capacity_off_enabled: bool = False,
    ) -> OfficialImbDiffCMTerms:
        """Expose the exact CM decomposition without sampling hidden randomness.

        The caller fixes the examples, discrete timesteps, and Gaussian noise.
        Classifier-free dropout is intentionally disabled because the mechanism
        probe compares class-frequency strata. When OC endpoint transfer is part
        of the method, ``transfer_seed`` fixes the release's multinomial draw.
        """

        self._validate_cm_term_request(model, clean, labels)
        if noise.shape != clean.shape:
            raise ValueError("CM probe noise must match the clean image tensor.")
        batch_size = clean.shape[0]
        if labels.shape != (batch_size,) or timesteps.shape != (batch_size,):
            raise ValueError("CM probe labels and timesteps must match the batch size.")
        labels = labels.to(device=clean.device, dtype=torch.long)
        timesteps = timesteps.to(device=clean.device, dtype=torch.long)
        if bool((timesteps < 0).any()) or bool((timesteps >= self.timesteps).any()):
            raise ValueError("CM probe timesteps are outside the diffusion schedule.")
        noise = noise.to(device=clean.device, dtype=clean.dtype)
        signal, sigma = self._expanded_diffusion_coefficients(timesteps, clean)
        noisy = signal * clean + sigma * noise

        capacity_on, capacity_off = self._cm_prediction_pair(
            model=model,
            noisy=noisy,
            timesteps=timesteps,
            conditioned_labels=labels,
            dropout_mode=dropout_mode,
            capacity_on_enabled=bool(capacity_on_enabled),
            capacity_off_enabled=bool(capacity_off_enabled),
        )
        trainer = self._trainer_for(model, clean.device)
        target = self._cm_target(
            trainer=trainer,
            noisy=noisy,
            clean=clean,
            labels=labels,
            timesteps=timesteps,
            noise=noise,
            signal=signal,
            transfer_seed=transfer_seed,
        )
        return self._assemble_cm_terms(
            trainer=trainer,
            noisy=noisy,
            target=target,
            labels=labels,
            timesteps=timesteps,
            conditioned_labels=labels,
            capacity_on=capacity_on,
            capacity_off=capacity_off,
            dropout_mode=dropout_mode,
            capacity_on_enabled=bool(capacity_on_enabled),
            capacity_off_enabled=bool(capacity_off_enabled),
            unconditional_batch=False,
        )

    def probe_inputs(
        self,
        *,
        model: nn.Module,
        clean: torch.Tensor,
        labels: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        transfer_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize fixed noisy inputs and the exact released training target."""

        self._validate_cm_term_request(model, clean, labels)
        if noise.shape != clean.shape:
            raise ValueError("CM probe noise must match the clean image tensor.")
        batch_size = clean.shape[0]
        if labels.shape != (batch_size,) or timesteps.shape != (batch_size,):
            raise ValueError("CM probe labels and timesteps must match the batch size.")
        labels = labels.to(device=clean.device, dtype=torch.long)
        timesteps = timesteps.to(device=clean.device, dtype=torch.long)
        if bool((timesteps < 0).any()) or bool((timesteps >= self.timesteps).any()):
            raise ValueError("CM probe timesteps are outside the diffusion schedule.")
        noise = noise.to(device=clean.device, dtype=clean.dtype)
        signal, sigma = self._expanded_diffusion_coefficients(timesteps, clean)
        noisy = signal * clean + sigma * noise

        trainer = self._trainer_for(model, clean.device)
        target = self._cm_target(
            trainer=trainer,
            noisy=noisy,
            clean=clean,
            labels=labels,
            timesteps=timesteps,
            noise=noise,
            signal=signal,
            transfer_seed=transfer_seed,
        )
        return noisy, target

    def _validate_cm_term_request(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if not self.uses_capacity_model:
            raise ValueError("CM terms require an official capacity model method.")
        if not bool(getattr(model, "is_official_imbdiff_cm", False)):
            raise ValueError("CM terms require the official CM U-Net.")
        if clean.ndim != 4 or tuple(clean.shape[1:]) != self.image_shape:
            raise ValueError(
                "CM clean images must be NCHW with shape "
                f"{self.image_shape}."
            )
        if labels.shape != (clean.shape[0],):
            raise ValueError("CM labels must match the batch size.")

    def _expanded_diffusion_coefficients(
        self,
        timesteps: torch.Tensor,
        clean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficient_shape = (clean.shape[0],) + (1,) * (clean.ndim - 1)
        signal = self._sqrt_alpha_bars.to(
            device=clean.device,
            dtype=clean.dtype,
        )[timesteps].reshape(coefficient_shape)
        sigma = self._sqrt_one_minus_alpha_bars.to(
            device=clean.device,
            dtype=clean.dtype,
        )[timesteps].reshape(coefficient_shape)
        return signal, sigma

    def _cm_prediction_pair(
        self,
        *,
        model: nn.Module,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        conditioned_labels: torch.Tensor | None,
        dropout_mode: str,
        capacity_on_enabled: bool,
        capacity_off_enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = str(dropout_mode).lower()
        if mode not in _CM_DROPOUT_MODES:
            raise ValueError(
                "CM dropout_mode must be independent, paired, or disabled."
            )

        def predict(use_capacity: bool) -> torch.Tensor:
            return model(
                noisy,
                timesteps,
                y=conditioned_labels,
                augm=None,
                use_cm=bool(use_capacity),
            )

        if mode == "disabled":
            was_training = model.training
            model.eval()
            try:
                return predict(capacity_on_enabled), predict(capacity_off_enabled)
            finally:
                model.train(was_training)

        if mode == "independent":
            return predict(capacity_on_enabled), predict(capacity_off_enabled)

        devices = _rng_cuda_devices(noisy)
        cpu_state = torch.random.get_rng_state()
        cuda_state = (
            torch.cuda.get_rng_state(devices[0])
            if devices
            else None
        )
        capacity_on = predict(capacity_on_enabled)
        # Enter the fork after the first pass so leaving it restores the RNG
        # stream to the state produced by that pass. Inside, replay the state
        # that existed immediately before the first pass.
        with torch.random.fork_rng(devices=devices):
            torch.random.set_rng_state(cpu_state)
            if devices and cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device=devices[0])
            capacity_off = predict(capacity_off_enabled)
        return capacity_on, capacity_off

    def _cm_target(
        self,
        *,
        trainer: nn.Module,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        labels: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
        signal: torch.Tensor,
        transfer_seed: int | None = None,
    ) -> torch.Tensor:
        if not self.transfer_x0:
            return noise

        sigma_t = torch.sqrt(
            torch.clamp(signal.reciprocal().square() - 1.0, min=0.0)
        )
        cx_t = clean + sigma_t * noise

        def transfer() -> torch.Tensor:
            if self.transfer_tr_tau:
                return trainer.do_transfer_x0_with_y(
                    noisy,
                    cx_t,
                    clean,
                    timesteps,
                    labels,
                    trainer.label_weight_tr,
                )
            target, _ = trainer.do_transfer_x0(
                noisy,
                cx_t,
                clean,
                timesteps,
                labels,
                return_transfer_label=True,
            )
            return target

        if transfer_seed is None:
            return transfer()
        devices = _rng_cuda_devices(clean)
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(transfer_seed))
            return transfer()

    def _assemble_cm_terms(
        self,
        *,
        trainer: nn.Module,
        noisy: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor,
        timesteps: torch.Tensor,
        conditioned_labels: torch.Tensor | None,
        capacity_on: torch.Tensor,
        capacity_off: torch.Tensor,
        dropout_mode: str,
        capacity_on_enabled: bool,
        capacity_off_enabled: bool,
        unconditional_batch: bool,
    ) -> OfficialImbDiffCMTerms:
        base_elements = F.mse_loss(capacity_on, target, reduction="none")
        base_per_sample = base_elements.flatten(1).mean(1)
        distance_per_sample = F.mse_loss(
            capacity_off.view(capacity_off.shape[0], -1),
            capacity_on.view(capacity_on.shape[0], -1),
            reduction="none",
        ).mean(dim=1)

        probabilities = trainer.weight.to(labels.device)
        inverse_probabilities = trainer.inverse_weight.to(labels.device)
        consistency_scale = probabilities[labels]
        diversity_scale = inverse_probabilities[labels]
        class_scale = float(len(self.class_counts))
        consistency_per_sample = (
            class_scale * consistency_scale * distance_per_sample
        )
        diversity_per_sample = (
            -class_scale * diversity_scale * distance_per_sample
        )
        coefficient_per_sample = class_scale * (
            consistency_scale * self.consistency_weight
            - diversity_scale * self.diversity_weight
        )
        total_per_sample = (
            base_per_sample + coefficient_per_sample * distance_per_sample
        )

        # Match the release's reduction order for exact scalar and gradient
        # compatibility; total_per_sample is the row-resolved diagnostic view.
        base_loss = base_elements.mean()
        auxiliary_loss = (
            (
                distance_per_sample
                * (
                    consistency_scale * self.consistency_weight
                    - diversity_scale * self.diversity_weight
                )
            ).mean()
            * class_scale
        )
        loss = base_loss + auxiliary_loss
        return OfficialImbDiffCMTerms(
            noisy=noisy,
            target=target,
            timesteps=timesteps,
            conditioned_labels=conditioned_labels,
            capacity_on=capacity_on,
            capacity_off=capacity_off,
            base_per_sample=base_per_sample,
            distance_per_sample=distance_per_sample,
            coefficient_per_sample=coefficient_per_sample,
            consistency_per_sample=consistency_per_sample,
            diversity_per_sample=diversity_per_sample,
            total_per_sample=total_per_sample,
            loss=loss,
            dropout_mode=str(dropout_mode).lower(),
            capacity_on_enabled=bool(capacity_on_enabled),
            capacity_off_enabled=bool(capacity_off_enabled),
            unconditional_batch=bool(unconditional_batch),
        )

    def _cbdm_loss(
        self,
        *,
        model: nn.Module,
        clean: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size = clean.shape[0]
        discrete_t = torch.randint(self.timesteps, (batch_size,), device=clean.device)
        noise = torch.randn_like(clean)
        coefficient_shape = (batch_size,) + (1,) * (clean.ndim - 1)
        signal = self._sqrt_alpha_bars.to(device=clean.device, dtype=clean.dtype)[
            discrete_t
        ].reshape(coefficient_shape)
        sigma = self._sqrt_one_minus_alpha_bars.to(
            device=clean.device, dtype=clean.dtype
        )[discrete_t].reshape(coefficient_shape)
        noisy = signal * clean + sigma * noise

        # Match the release's whole-batch CFG dropout and CPU RNG stream.
        dropped = self.cfg and bool(torch.rand(1)[0] < 0.1)
        conditioned_labels = None if dropped else labels
        prediction = model(noisy, discrete_t, y=conditioned_labels, augm=None)
        base_loss = F.mse_loss(prediction, noise)

        probabilities = self._cbdm_probabilities.to(device=clean.device)
        auxiliary_labels = torch.multinomial(
            probabilities,
            num_samples=batch_size,
            replacement=True,
        )
        auxiliary_prediction = model(
            noisy,
            discrete_t,
            y=auxiliary_labels,
            augm=None,
        )
        time_weight = self.cbdm_tau * discrete_t.to(dtype=prediction.dtype)
        regularizer_distance = (
            (prediction - auxiliary_prediction.detach()).square().flatten(1).mean(1)
        )
        commitment_distance = (
            (prediction.detach() - auxiliary_prediction).square().flatten(1).mean(1)
        )
        regularizer = (time_weight * regularizer_distance).mean()
        commitment = self.cbdm_gamma * (time_weight * commitment_distance).mean()
        loss = base_loss + regularizer + commitment
        return loss, {
            "official_imbdiff_loss": float(loss.detach().cpu()),
            "official_imbdiff_cbdm_loss": float(loss.detach().cpu()),
            "cbdm_base_loss": float(base_loss.detach().cpu()),
            "cbdm_regularizer": float(regularizer.detach().cpu()),
            "cbdm_commitment": float(commitment.detach().cpu()),
            "cbdm_unconditional_batch": float(dropped),
            "loss": float(loss.detach().cpu()),
        }

    def _trainer_for(self, model: nn.Module, device: torch.device) -> nn.Module:
        if (
            self._trainer is not None
            and self._trainer_model is model
            and self._trainer_device == device
        ):
            return self._trainer
        if self.method == "cbdm":
            raise RuntimeError("CBDM uses its paper-derived objective, not a release trainer.")
        components = load_official_imbdiff_cm_components()
        probabilities = torch.tensor(self.class_counts, dtype=torch.float32)
        probabilities = probabilities / probabilities.sum()
        label_weight = torch.pow(
            probabilities.unsqueeze(1) @ probabilities.unsqueeze(0),
            self.transfer_tau,
        )
        trainer_kwargs = {
            "model": model,
            "beta_1": self.beta_start,
            "beta_T": self.beta_end,
            "T": self.timesteps,
            "dataset": None,
            "num_class": len(self.class_counts),
            "cfg": self.cfg,
            "weight": probabilities.unsqueeze(0),
            "transfer_x0": self.transfer_x0,
            "transfer_tr_tau": self.transfer_tr_tau,
            "transfer_mode": self.transfer_mode,
            "label_weight_tr": label_weight,
        }
        if self.uses_capacity_model:
            trainer = components.cm_trainer(
                **trainer_kwargs,
                w_con=self.consistency_weight,
                w_div=self.diversity_weight,
            ).to(device)
        else:
            trainer = components.trainer(**trainer_kwargs).to(device)
        self._trainer = trainer
        self._trainer_model = model
        self._trainer_device = device
        return trainer

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": "vendored_official_release",
            "method": self.method,
            "model_family": "cm" if self.uses_capacity_model else "standard",
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
            "cbdm": {
                "target_distribution": self.cbdm_target_distribution,
                "tau": self.cbdm_tau,
                "gamma": self.cbdm_gamma,
                "implementation": "Qin_et_al_CVPR_2023_equation_4",
            },
        }


class OfficialImbDiffCMObjective(OfficialImbDiffObjective):
    """Backward-compatible name for the released CM recipe."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(method="released_cm", **kwargs)


def _rng_cuda_devices(reference: torch.Tensor) -> list[int]:
    if reference.device.type != "cuda":
        return []
    return [
        reference.device.index
        if reference.device.index is not None
        else torch.cuda.current_device()
    ]


@torch.no_grad()
def sample_official_imbdiff(
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
    sampler_family: str | None = None,
) -> torch.Tensor:
    """Invoke the standard or CM sampler used by ``tools/sample_images.py``."""

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
    if sampler_family is None:
        family = (
            "cm"
            if bool(getattr(model, "is_official_imbdiff_cm", False))
            else "standard"
        )
    else:
        family = str(sampler_family).lower()
    if family not in {"standard", "cm"}:
        raise ValueError("sampler_family must be 'standard' or 'cm'.")
    sampler_class = components.cm_sampler if family == "cm" else components.sampler
    sampler = sampler_class(
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


@torch.no_grad()
def sample_official_imbdiff_cm(**kwargs: Any) -> torch.Tensor:
    """Backward-compatible CM-sampler entry point."""

    kwargs["sampler_family"] = "cm"
    return sample_official_imbdiff(**kwargs)
