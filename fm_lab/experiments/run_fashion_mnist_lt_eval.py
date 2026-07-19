"""Evaluate balanced conditional Fashion-MNIST long-tail generations."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from fm_lab.data import LongTailedFashionMNIST
from fm_lab.diagnostics.mnist_eval import (
    FashionMNISTFeatureEvaluator,
    load_or_train_fashion_mnist_classifier,
)
from fm_lab.evaluation.cache import FeatureCache, load_feature_cache, save_feature_cache
from fm_lab.evaluation.features import extract_classifier_features
from fm_lab.evaluation.report import (
    evaluate_feature_caches,
    evaluate_reference_calibration,
    write_evaluation_report,
)
from fm_lab.experiments.factory import resolve_device

_NUM_CLASSES = 10
_TRAIN_CLASS_COUNT = 6000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-cache")
    parser.add_argument("--real-cache")
    parser.add_argument("--generated-samples")
    parser.add_argument("--generated-labels")
    parser.add_argument("--generative-checkpoint")
    parser.add_argument("--generation-method")
    parser.add_argument("--sampler", default="euler")
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--generative-weights",
        choices=("raw", "ema"),
        default="raw",
        help="Checkpoint weight variant used to produce the generated arrays.",
    )
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--data-root", default="data/fashion_mnist")
    parser.add_argument(
        "--classifier-checkpoint",
        default="artifacts/fashion_mnist_lt_evaluator.pt",
    )
    parser.add_argument("--classifier-steps", type=int, default=1000)
    parser.add_argument("--classifier-eval-samples", type=int, default=10_000)
    parser.add_argument("--classifier-lr", type=float, default=1.0e-3)
    parser.add_argument("--minimum-accuracy", type=float, default=0.9)
    parser.add_argument("--normalize", default="minus_one_one")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--feature-cache-dir", default="features/fashion_mnist_lt")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imbalance-factor", type=float, default=0.01)
    parser.add_argument(
        "--class-ids",
        default=None,
        help=(
            "Comma-separated original Fashion-MNIST class IDs to evaluate. Labels are "
            "expected to be compact 0..K-1 in the generated arrays."
        ),
    )
    parser.add_argument(
        "--class-counts",
        default=None,
        help=(
            "Comma-separated unique training support for classes 0..9. Required for "
            "frequency rotations; otherwise the legacy monotone IR counts are used."
        ),
    )
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--overall-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--inception-splits", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    class_ids = _parse_class_ids(args.class_ids)
    generated, real = _resolve_feature_caches(args)
    _validate_cache_pair(generated, real)
    _validate_canonical_real_cache(
        real,
        class_ids=class_ids,
        samples_per_class=args.samples_per_class,
    )
    _validate_balanced_labels(
        generated.labels,
        samples_per_class=args.samples_per_class,
        num_classes=len(class_ids),
    )
    class_counts = _resolve_class_counts(
        args.class_counts,
        args.imbalance_factor,
        num_classes=len(class_ids),
    )
    report = evaluate_feature_caches(
        generated,
        real,
        class_counts=class_counts,
        repeats=args.repeats,
        overall_samples=args.overall_samples,
        seed=args.seed,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        recall_k=args.recall_k,
        inception_splits=args.inception_splits,
        require_balanced_generated=True,
        per_class_recall=True,
        conditional_diagnostics=True,
    )
    report["provenance"].update(
        {
            "benchmark": "fashion_mnist_lt",
            "samples_per_class": args.samples_per_class,
            "imbalance_factor": args.imbalance_factor,
            "class_counts": class_counts,
            "class_ids": list(class_ids),
            "reference_split": "official_test",
        }
    )
    report["reference_calibration"] = evaluate_reference_calibration(
        real,
        seed=args.seed,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        recall_k=args.recall_k,
        inception_splits=args.inception_splits,
    )
    paths = write_evaluation_report(report, args.output_dir)
    print(f"Wrote Fashion-MNIST long-tail metrics: {paths['json']}")
    return 0


def _resolve_feature_caches(args: argparse.Namespace) -> tuple[FeatureCache, FeatureCache]:
    class_ids = _parse_class_ids(args.class_ids)
    if args.generated_cache or args.real_cache:
        if not args.generated_cache or not args.real_cache:
            raise ValueError("Both --generated-cache and --real-cache are required together.")
        generated = load_feature_cache(args.generated_cache)
        real = load_feature_cache(args.real_cache)
        return (
            _subset_probability_cache(generated, class_ids),
            _subset_probability_cache(real, class_ids),
        )
    if not args.generated_samples or not args.generated_labels:
        raise ValueError(
            "Provide paired feature caches or both --generated-samples and --generated-labels."
        )
    if not args.generative_checkpoint or not args.generation_method:
        raise ValueError(
            "Array extraction requires --generative-checkpoint and --generation-method."
        )

    device = torch.device(resolve_device(args.device))
    classifier, metadata = load_or_train_fashion_mnist_classifier(
        data_root=args.data_root,
        normalize=args.normalize,
        download=args.download,
        checkpoint_path=Path(args.classifier_checkpoint),
        steps=args.classifier_steps,
        batch_size=args.batch_size,
        eval_samples=args.classifier_eval_samples,
        lr=args.classifier_lr,
        device=device,
    )
    evaluator = FashionMNISTFeatureEvaluator(
        classifier,
        metadata=metadata,
        normalize=args.normalize,
        minimum_accuracy=args.minimum_accuracy,
    )
    input_range = _image_range(args.normalize)
    cache_dir = Path(args.feature_cache_dir)

    generated_images = np.load(args.generated_samples)
    generated_labels = np.load(args.generated_labels)
    generated_provenance = dict(evaluator.provenance)
    generated_provenance.update(
        {
            "split": "generated",
            "source_samples": str(Path(args.generated_samples).resolve()),
            "source_labels": str(Path(args.generated_labels).resolve()),
            "source_samples_sha256": _sha256_file(Path(args.generated_samples)),
            "source_labels_sha256": _sha256_file(Path(args.generated_labels)),
            "generative_checkpoint_sha256": _sha256_file(
                Path(args.generative_checkpoint)
            ),
            "generative_weights": args.generative_weights,
            "generation_method": args.generation_method,
            "sampler": args.sampler,
            "nfe": args.nfe,
            "guidance_scale": args.guidance_scale,
            "generation_seed": args.generation_seed,
        }
    )
    generated = extract_classifier_features(
        generated_images,
        labels=generated_labels,
        sample_ids=np.arange(len(generated_images)).astype(str),
        model=evaluator,
        batch_size=args.batch_size,
        device=device,
        input_range=input_range,
        provenance=generated_provenance,
    )
    generated = _subset_probability_cache(generated, class_ids)
    save_feature_cache(
        cache_dir / f"generated_fashion_mnist_lt_{generated.fingerprint[:16]}.npz",
        generated,
    )

    real_dataset = LongTailedFashionMNIST(
        root=args.data_root,
        train=False,
        download=args.download,
        imbalance_type="balanced",
        imbalance_factor=1.0,
        normalize=args.normalize,
        class_ids=class_ids,
    )
    real_images, real_labels, real_ids = real_dataset.all_samples_with_labels()
    real_provenance = dict(evaluator.provenance)
    real_provenance.update(
        {
            "split": "official_test",
            "dataset_metadata": real_dataset.metadata(),
        }
    )
    real = extract_classifier_features(
        real_images,
        labels=real_labels.numpy(),
        sample_ids=real_ids,
        model=evaluator,
        batch_size=args.batch_size,
        device=device,
        input_range=input_range,
        provenance=real_provenance,
    )
    real = _subset_probability_cache(real, class_ids)
    save_feature_cache(cache_dir / "real_fashion_mnist_test.npz", real)
    return generated, real


def _validate_cache_pair(generated: FeatureCache, real: FeatureCache) -> None:
    for field in (
        "dataset",
        "extractor",
        "weights_sha256",
        "preprocessing",
        "image_shape",
        "normalize",
        "evaluator_version",
        "architecture",
        "feature_layer",
        "feature_dimension",
        "class_order",
        "minimum_accuracy",
        "test_accuracy",
    ):
        if generated.provenance.get(field) != real.provenance.get(field):
            raise ValueError(f"Feature cache provenance mismatch for {field}.")
    if generated.features.shape[1] != real.features.shape[1]:
        raise ValueError("Feature cache dimensions do not match.")
    if generated.features.shape[1] != int(generated.provenance["feature_dimension"]):
        raise ValueError("Feature cache dimension does not match evaluator provenance.")
    if float(real.provenance["test_accuracy"]) < float(
        real.provenance["minimum_accuracy"]
    ):
        raise ValueError("Cached evaluator accuracy is below its required threshold.")
    required_generation = {
        "source_samples_sha256",
        "source_labels_sha256",
        "generative_checkpoint_sha256",
        "generative_weights",
        "generation_method",
        "sampler",
        "nfe",
        "guidance_scale",
        "generation_seed",
    }
    missing = required_generation - set(generated.provenance)
    if missing:
        raise ValueError(f"Generated cache is missing protocol provenance: {sorted(missing)}")


def _validate_canonical_real_cache(
    real: FeatureCache,
    *,
    class_ids: tuple[int, ...],
    samples_per_class: int,
) -> None:
    if real.provenance.get("split") != "official_test":
        raise ValueError("Real cache must use split 'official_test'.")
    counts = np.bincount(real.labels, minlength=len(class_ids))
    if len(counts) != len(class_ids) or not np.array_equal(
        counts, np.full(len(class_ids), samples_per_class, dtype=np.int64)
    ):
        raise ValueError(
            "Real cache must contain the official test subset with equal samples per class."
        )
    try:
        sample_ids = real.sample_ids.astype(np.int64)
    except ValueError as exc:
        raise ValueError("Real cache sample identifiers must be official test indices.") from exc
    metadata = real.provenance.get("dataset_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Real cache is missing official dataset metadata.")
    expected_metadata = {
        "dataset": "fashion_mnist",
        "train": False,
        "n_images": samples_per_class * len(class_ids),
        "class_counts": [samples_per_class] * len(class_ids),
        "original_class_ids": list(class_ids),
        "subset_sha256": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
    }
    for field, value in expected_metadata.items():
        if metadata.get(field) != value:
            raise ValueError(f"Real cache dataset metadata mismatch for {field}.")


def _validate_balanced_labels(
    labels: np.ndarray,
    *,
    samples_per_class: int,
    num_classes: int,
) -> None:
    if samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least two.")
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    expected = np.full(num_classes, samples_per_class, dtype=np.int64)
    if len(counts) != num_classes or not np.array_equal(counts, expected):
        raise ValueError(
            f"Generated labels must contain exactly {samples_per_class} samples per class."
        )


def _long_tail_class_counts(imbalance_factor: float, *, num_classes: int) -> list[int]:
    if not 0.0 < imbalance_factor <= 1.0:
        raise ValueError("--imbalance-factor must be in (0, 1].")
    return [
        int(_TRAIN_CLASS_COUNT * imbalance_factor ** (class_id / (num_classes - 1.0)))
        for class_id in range(num_classes)
    ]


def _resolve_class_counts(
    serialized: str | None,
    imbalance_factor: float,
    *,
    num_classes: int = _NUM_CLASSES,
) -> list[int]:
    if serialized is None:
        return _long_tail_class_counts(imbalance_factor, num_classes=num_classes)
    try:
        counts = [int(value.strip()) for value in serialized.split(",")]
    except ValueError as exc:
        raise ValueError("--class-counts must be comma-separated integers.") from exc
    if len(counts) != num_classes or any(value <= 0 for value in counts):
        raise ValueError(
            f"--class-counts must contain {num_classes} positive integers."
        )
    return counts


def _parse_class_ids(serialized: str | None) -> tuple[int, ...]:
    if serialized is None or not str(serialized).strip():
        return tuple(range(_NUM_CLASSES))
    try:
        values = tuple(int(value.strip()) for value in str(serialized).split(","))
    except ValueError as exc:
        raise ValueError("--class-ids must be comma-separated integers.") from exc
    if len(values) < 3:
        raise ValueError("--class-ids must include at least three classes.")
    if len(set(values)) != len(values) or any(value < 0 or value >= _NUM_CLASSES for value in values):
        raise ValueError("--class-ids must be unique Fashion-MNIST class IDs in [0, 9].")
    return values


def _subset_probability_cache(cache: FeatureCache, class_ids: tuple[int, ...]) -> FeatureCache:
    if class_ids == tuple(range(_NUM_CLASSES)):
        return cache
    probabilities = np.asarray(cache.probabilities)
    if probabilities.shape[1] != _NUM_CLASSES:
        raise ValueError("Subset Fashion-MNIST evaluation requires ten classifier outputs.")
    labels = np.asarray(cache.labels, dtype=np.int64)
    if np.any(labels < 0) or np.any(labels >= len(class_ids)):
        raise ValueError("Subset cache labels must be compact 0..K-1 identifiers.")
    provenance = dict(cache.provenance)
    provenance["class_order"] = list(class_ids)
    provenance["probability_class_order"] = list(class_ids)
    return FeatureCache(
        features=cache.features,
        probabilities=probabilities[:, list(class_ids)],
        labels=labels,
        sample_ids=cache.sample_ids,
        provenance=provenance,
    )


def _image_range(normalize: str) -> tuple[float, float]:
    if normalize.lower() in {"minus_one_one", "-1_1", "centered"}:
        return (-1.0, 1.0)
    if normalize.lower() in {"zero_one", "01", "unit"}:
        return (0.0, 1.0)
    raise ValueError(f"Unsupported Fashion-MNIST normalization: {normalize}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
