"""MNIST run evaluation helpers."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fm_lab.data import MNISTImages
from fm_lab.utils.config import ConfigError, load_config
from fm_lab.utils.logging import write_json


@dataclass
class MNISTEvalConfig:
    run_dir: Path
    output_dir: Path | None = None
    solver: str = "auto"
    nfe: int = 64
    max_samples: int = 256
    reference_samples: int = 2048
    nearest_neighbors: int = 16
    classifier_checkpoint: Path = Path("artifacts/mnist_classifier.pt")
    classifier_steps: int = 1000
    classifier_batch_size: int = 256
    classifier_eval_samples: int = 2048
    classifier_lr: float = 1.0e-3
    skip_classifier: bool = False
    device: torch.device = torch.device("cpu")


def evaluate_mnist_run(eval_config: MNISTEvalConfig) -> dict[str, Any]:
    """Evaluate generated samples from a completed MNIST run."""

    run_dir = eval_config.run_dir
    config = _load_run_config(run_dir)
    if str(config.get("data", {}).get("name", "")).lower() != "mnist":
        raise ConfigError(f"MNIST evaluation requires data.name: mnist, got {config.get('data')}.")

    output_dir = eval_config.output_dir or run_dir
    diagnostics_dir = output_dir / "diagnostics"
    plots_dir = output_dir / "plots"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sample_path = _resolve_sample_path(run_dir, solver=eval_config.solver, nfe=eval_config.nfe)
    target_path = run_dir / "samples" / "target_reference.npy"
    if not target_path.exists():
        raise ConfigError(f"Run is missing target reference samples: {target_path}")

    generated = torch.as_tensor(np.load(sample_path), dtype=torch.float32)[
        : eval_config.max_samples
    ]
    target_reference = torch.as_tensor(np.load(target_path), dtype=torch.float32)[
        : eval_config.max_samples
    ]
    mnist = _build_mnist_from_config(config, train=True, dequantize=False)
    image_range = tuple(float(value) for value in mnist.metadata()["image_value_range"])
    train_reference = mnist.sample(eval_config.reference_samples)

    metrics: dict[str, Any] = {
        "run_dir": str(run_dir),
        "sample_path": str(sample_path),
        "target_path": str(target_path),
        "n_generated": int(generated.shape[0]),
        "n_target_reference": int(target_reference.shape[0]),
        "n_train_reference": int(train_reference.shape[0]),
        "image_value_range": list(image_range),
        "pixel_stats": {
            "generated": _pixel_stats(generated, image_range),
            "target_reference": _pixel_stats(target_reference, image_range),
            "train_reference": _pixel_stats(train_reference, image_range),
        },
        "moment_gaps": _moment_gaps(generated=generated, reference=train_reference),
        "diversity": _diversity_stats(generated),
    }

    generated_clipped = generated.clamp(*image_range)
    nearest = _nearest_neighbor_stats(
        generated=generated_clipped,
        reference=train_reference,
        n_neighbors=eval_config.nearest_neighbors,
    )
    metrics["nearest_neighbors"] = nearest["metrics"]
    nearest_plot = plots_dir / f"mnist_nearest_neighbors_{sample_path.stem}.png"
    _plot_nearest_neighbors(
        generated=generated_clipped[nearest["generated_indices"]],
        neighbors=train_reference[nearest["neighbor_indices"]],
        distances=nearest["distances"],
        output_path=nearest_plot,
        image_range=image_range,
    )
    metrics["plots"] = {"nearest_neighbors": str(nearest_plot)}

    if not eval_config.skip_classifier:
        classifier_metrics = _evaluate_with_classifier(
            config=config,
            generated=generated_clipped,
            checkpoint_path=eval_config.classifier_checkpoint,
            steps=eval_config.classifier_steps,
            batch_size=eval_config.classifier_batch_size,
            eval_samples=eval_config.classifier_eval_samples,
            lr=eval_config.classifier_lr,
            device=eval_config.device,
        )
        metrics["classifier"] = classifier_metrics

    json_path = diagnostics_dir / f"mnist_eval_{sample_path.stem}.json"
    csv_path = diagnostics_dir / f"mnist_eval_{sample_path.stem}.csv"
    metrics["outputs"] = {"json": str(json_path), "csv": str(csv_path)}
    write_json(metrics, json_path)
    _write_flat_csv(metrics, csv_path)
    return metrics


class MNISTClassifier(nn.Module):
    """Small CNN classifier used for recognizability/diversity diagnostics."""

    def __init__(self, image_shape: tuple[int, int] = (28, 28)) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], 1, *self.image_shape))


def _evaluate_with_classifier(
    *,
    config: dict[str, Any],
    generated: torch.Tensor,
    checkpoint_path: Path,
    steps: int,
    batch_size: int,
    eval_samples: int,
    lr: float,
    device: torch.device,
) -> dict[str, Any]:
    train_data = _build_mnist_from_config(config, train=True, dequantize=False)
    test_data = _build_mnist_from_config(config, train=False, dequantize=False)
    checkpoint_path = _classifier_checkpoint_path(checkpoint_path, config)
    classifier = MNISTClassifier().to(device)

    trained_now = False
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        classifier.load_state_dict(payload["model_state_dict"])
        classifier_steps = int(payload.get("steps", 0))
    else:
        classifier_steps = steps
        _train_classifier(
            classifier=classifier,
            data=train_data,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": classifier.state_dict(),
                "steps": steps,
                "normalize": config.get("data", {}).get("normalize", "zero_one"),
            },
            checkpoint_path,
        )
        trained_now = True

    classifier.eval()
    test_accuracy = _classifier_accuracy(
        classifier=classifier,
        data=test_data,
        n_samples=eval_samples,
        batch_size=batch_size,
        device=device,
    )
    generated_metrics = _classifier_sample_metrics(
        classifier=classifier,
        samples=generated,
        batch_size=batch_size,
        device=device,
    )
    target_metrics = _classifier_sample_metrics(
        classifier=classifier,
        samples=test_data.sample(min(eval_samples, generated.shape[0])),
        batch_size=batch_size,
        device=device,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "trained_now": trained_now,
        "classifier_steps": classifier_steps,
        "test_accuracy": test_accuracy,
        "generated": generated_metrics,
        "target_reference": target_metrics,
        "uses_clipped_generated_samples": True,
    }


def _train_classifier(
    *,
    classifier: MNISTClassifier,
    data: MNISTImages,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    images = data.images
    labels = data.labels
    classifier.train()
    for _ in range(steps):
        indices = torch.randint(0, images.shape[0], (batch_size,))
        x = images[indices].to(device)
        y = labels[indices].to(device)
        loss = F.cross_entropy(classifier(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _classifier_accuracy(
    *,
    classifier: MNISTClassifier,
    data: MNISTImages,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> float:
    n_samples = min(n_samples, data.images.shape[0])
    indices = torch.randperm(data.images.shape[0])[:n_samples]
    correct = 0
    total = 0
    for start in range(0, n_samples, batch_size):
        batch_indices = indices[start : start + batch_size]
        x = data.images[batch_indices].to(device)
        y = data.labels[batch_indices].to(device)
        pred = classifier(x).argmax(dim=1)
        correct += int((pred == y).sum().cpu())
        total += int(y.numel())
    return correct / max(total, 1)


@torch.no_grad()
def _classifier_sample_metrics(
    *,
    classifier: MNISTClassifier,
    samples: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    probabilities = []
    for start in range(0, samples.shape[0], batch_size):
        x = samples[start : start + batch_size].to(device)
        probabilities.append(torch.softmax(classifier(x), dim=1).cpu())
    probs = torch.cat(probabilities, dim=0)
    confidence = probs.max(dim=1).values
    predicted = probs.argmax(dim=1)
    counts = torch.bincount(predicted, minlength=10).float()
    distribution = counts / counts.sum().clamp_min(1.0)
    entropy = float(-(distribution * distribution.clamp_min(1e-12).log()).sum() / math.log(10))
    return {
        "confidence_mean": float(confidence.mean()),
        "confidence_p10": float(confidence.quantile(0.10)),
        "confidence_p50": float(confidence.quantile(0.50)),
        "confidence_p90": float(confidence.quantile(0.90)),
        "predicted_digit_counts": [int(value) for value in counts.tolist()],
        "predicted_digit_distribution": [float(value) for value in distribution.tolist()],
        "predicted_digit_entropy_0_1": entropy,
        "max_digit_fraction": float(distribution.max()),
    }


def _build_mnist_from_config(
    config: dict[str, Any],
    *,
    train: bool,
    dequantize: bool,
) -> MNISTImages:
    data_config = config.get("data", {})
    return MNISTImages(
        root=data_config.get("root", "data/mnist"),
        train=train,
        download=bool(data_config.get("download", False)),
        normalize=str(data_config.get("normalize", "zero_one")),
        dequantize=dequantize,
    )


def _pixel_stats(samples: torch.Tensor, image_range: tuple[float, float]) -> dict[str, float]:
    low, high = image_range
    return {
        "min": float(samples.min()),
        "max": float(samples.max()),
        "mean": float(samples.mean()),
        "std": float(samples.std(unbiased=False)),
        "frac_below_range": float((samples < low).float().mean()),
        "frac_above_range": float((samples > high).float().mean()),
        "frac_out_of_range": float(((samples < low) | (samples > high)).float().mean()),
    }


def _moment_gaps(*, generated: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    gen_mean = generated.mean(dim=0)
    ref_mean = reference.mean(dim=0)
    gen_std = generated.std(dim=0, unbiased=False)
    ref_std = reference.std(dim=0, unbiased=False)
    return {
        "per_pixel_mean_l2": float((gen_mean - ref_mean).norm()),
        "per_pixel_mean_mae": float((gen_mean - ref_mean).abs().mean()),
        "per_pixel_std_l2": float((gen_std - ref_std).norm()),
        "per_pixel_std_mae": float((gen_std - ref_std).abs().mean()),
    }


def _diversity_stats(samples: torch.Tensor) -> dict[str, float]:
    if samples.shape[0] < 2:
        return {"pairwise_l2_p10": 0.0, "pairwise_l2_p50": 0.0, "pairwise_l2_mean": 0.0}
    distances = torch.pdist(samples)
    return {
        "pairwise_l2_p10": float(distances.quantile(0.10)),
        "pairwise_l2_p50": float(distances.quantile(0.50)),
        "pairwise_l2_mean": float(distances.mean()),
    }


def _nearest_neighbor_stats(
    *,
    generated: torch.Tensor,
    reference: torch.Tensor,
    n_neighbors: int,
) -> dict[str, Any]:
    distances = torch.cdist(generated, reference)
    nearest_distances, nearest_indices = distances.min(dim=1)
    order = torch.argsort(nearest_distances)[: min(n_neighbors, generated.shape[0])]
    return {
        "generated_indices": order,
        "neighbor_indices": nearest_indices[order],
        "distances": nearest_distances[order],
        "metrics": {
            "train_l2_mean": float(nearest_distances.mean()),
            "train_l2_p10": float(nearest_distances.quantile(0.10)),
            "train_l2_p50": float(nearest_distances.quantile(0.50)),
            "train_l2_p90": float(nearest_distances.quantile(0.90)),
            "train_l2_min": float(nearest_distances.min()),
        },
    }


def _plot_nearest_neighbors(
    *,
    generated: torch.Tensor,
    neighbors: torch.Tensor,
    distances: torch.Tensor,
    output_path: Path,
    image_range: tuple[float, float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pairs = generated.shape[0]
    vmin, vmax = image_range
    fig, axes = plt.subplots(n_pairs, 2, figsize=(3.0, 1.45 * n_pairs), squeeze=False)
    for idx in range(n_pairs):
        for col, values in enumerate((generated[idx], neighbors[idx])):
            axes[idx, col].imshow(
                values.reshape(28, 28).numpy().clip(vmin, vmax),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
            axes[idx, col].axis("off")
        axes[idx, 0].set_ylabel(f"{float(distances[idx]):.2f}", rotation=0, labelpad=18)
    axes[0, 0].set_title("generated", fontsize=9)
    axes[0, 1].set_title("nearest train", fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise ConfigError(f"Run directory is missing config.yaml: {run_dir}")
    return load_config(config_path)


def _resolve_sample_path(run_dir: Path, *, solver: str, nfe: int) -> Path:
    samples_dir = run_dir / "samples"
    if solver != "auto":
        sample_path = samples_dir / f"{solver}_nfe{nfe}.npy"
        if sample_path.exists():
            return sample_path
        raise ConfigError(f"Required generated sample file is missing: {sample_path}.")

    matches = sorted(samples_dir.glob(f"*_nfe{nfe}.npy"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ConfigError(f"No generated sample files found for nfe={nfe} in {samples_dir}.")
    raise ConfigError(
        f"Found multiple generated sample files for nfe={nfe} in {samples_dir}; "
        "pass --solver explicitly."
    )


def _classifier_checkpoint_path(path: Path, config: dict[str, Any]) -> Path:
    normalize = str(config.get("data", {}).get("normalize", "zero_one")).replace("/", "_")
    if path.suffix:
        return path.with_name(f"{path.stem}_{normalize}{path.suffix}")
    return path / f"mnist_classifier_{normalize}.pt"


def _write_flat_csv(metrics: dict[str, Any], path: Path) -> None:
    flat = _flatten(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in sorted(flat.items()):
            writer.writerow({"metric": key, "value": value})


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, list):
            flat[name] = ",".join(str(item) for item in value)
        else:
            flat[name] = value
    return flat
