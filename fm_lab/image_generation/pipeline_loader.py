"""Diffusers pipeline loading for FLUX.2-klein."""

from __future__ import annotations

from typing import Any

from fm_lab.image_generation.config import ModelConfig


def load_flux_pipeline(model_config: ModelConfig) -> Any:
    """Load a configured FLUX.2-klein Diffusers pipeline.

    Current Black Forest Labs and Diffusers examples use
    ``Flux2KleinPipeline.from_pretrained(..., torch_dtype=...)``. The import is
    intentionally lazy so dry-runs and unit tests do not need Diffusers.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("torch is required for FLUX.2-klein generation.") from exc

    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "diffusers does not expose Flux2KleinPipeline. The FLUX.2-klein model card "
            "currently recommends installing Diffusers from source:\n"
            "  pip install git+https://github.com/huggingface/diffusers.git"
        ) from exc

    dtype = resolve_torch_dtype(model_config.dtype, torch)
    device = resolve_device(model_config.device, torch)
    load_kwargs: dict[str, Any] = {}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    if model_config.device_map is not None:
        load_kwargs["device_map"] = model_config.device_map

    pipe = Flux2KleinPipeline.from_pretrained(model_config.repo_id, **load_kwargs)

    if model_config.cpu_offload:
        pipe.enable_model_cpu_offload()
    elif model_config.device_map is None:
        pipe.to(device)

    if model_config.torch_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("model.torch_compile=true requires torch.compile support.")
        if hasattr(pipe, "transformer"):
            pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
        else:
            raise RuntimeError("Loaded pipeline does not expose a transformer to compile.")

    return pipe


def resolve_torch_dtype(dtype_name: str, torch_module: Any) -> Any:
    """Resolve a config dtype string to a torch dtype."""

    aliases = {
        "auto": None,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    key = dtype_name.lower()
    if key not in aliases:
        raise ValueError(f"Unsupported torch dtype: {dtype_name!r}")
    return aliases[key]


def resolve_device(device_name: str, torch_module: Any) -> str:
    """Resolve 'auto' and validate obvious device availability."""

    device = device_name.lower()
    if device == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if getattr(torch_module.backends, "mps", None) is not None:
            if torch_module.backends.mps.is_available():
                return "mps"
        return "cpu"
    if device.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(f"Configured device {device_name!r}, but CUDA is not available.")
    if device == "mps":
        mps_backend = getattr(torch_module.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("Configured device 'mps', but MPS is not available.")
    return device_name


def make_torch_generator(seed: int, device_name: str):
    """Create a deterministic torch.Generator for the target device when possible."""

    import torch

    device = resolve_device(device_name, torch)
    generator_device = device if str(device).startswith("cuda") else "cpu"
    try:
        return torch.Generator(device=generator_device).manual_seed(int(seed))
    except RuntimeError:
        return torch.Generator(device="cpu").manual_seed(int(seed))
