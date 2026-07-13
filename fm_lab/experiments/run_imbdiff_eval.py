"""Evaluate cached ImbDiff CIFAR Inception features."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from fm_lab.evaluation.cache import load_feature_cache, save_feature_cache
from fm_lab.evaluation.report import evaluate_feature_caches, write_evaluation_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-cache")
    parser.add_argument("--real-cache")
    parser.add_argument("--generated-samples")
    parser.add_argument("--generated-labels")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"))
    parser.add_argument("--data-root")
    parser.add_argument("--weights", default="stats/pt_inception-2015-12-05-6726825d.pth")
    parser.add_argument("--feature-cache-dir", default="features/imbdiff")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--class-counts", required=True, help="JSON file containing counts by class id."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--overall-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--recall-k", type=int, default=3)
    parser.add_argument("--inception-splits", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generated, real = _resolve_feature_caches(args)
    class_counts = json.loads(Path(args.class_counts).read_text(encoding="utf-8"))
    if not isinstance(class_counts, list):
        raise ValueError("--class-counts must contain a JSON list.")
    report = evaluate_feature_caches(
        generated,
        real,
        class_counts=[int(value) for value in class_counts],
        repeats=args.repeats,
        overall_samples=args.overall_samples,
        seed=args.seed,
        kid_subsets=args.kid_subsets,
        kid_subset_size=args.kid_subset_size,
        recall_k=args.recall_k,
        inception_splits=args.inception_splits,
    )
    paths = write_evaluation_report(report, args.output_dir)
    print(f"Wrote ImbDiff metrics: {paths['json']}")
    return 0


def _resolve_feature_caches(args: argparse.Namespace):
    if args.generated_cache or args.real_cache:
        if not args.generated_cache or not args.real_cache:
            raise ValueError("Both --generated-cache and --real-cache are required together.")
        return load_feature_cache(args.generated_cache), load_feature_cache(args.real_cache)
    required = {
        "--generated-samples": args.generated_samples,
        "--generated-labels": args.generated_labels,
        "--dataset": args.dataset,
        "--data-root": args.data_root,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "Provide cached features or all extraction inputs; missing " + ", ".join(missing)
        )

    from fm_lab.data import ImbalancedCIFARImages
    from fm_lab.evaluation.features import extract_inception_features
    from fm_lab.evaluation.inception import ReferenceInceptionV3

    device = _resolve_device(args.device)
    model = ReferenceInceptionV3(args.weights)
    cache_dir = Path(args.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    generated_images = np.load(args.generated_samples)
    generated_labels = np.load(args.generated_labels)
    generated_provenance = {
        "dataset": args.dataset,
        "split": "generated",
        "extractor": "tf_fid_inception_v3",
        "weights_sha256": model.weights_sha256,
        "evaluator_version": 1,
        "source_samples": str(Path(args.generated_samples).resolve()),
        "source_labels": str(Path(args.generated_labels).resolve()),
    }
    generated = extract_inception_features(
        generated_images,
        labels=generated_labels,
        sample_ids=np.arange(len(generated_images)).astype(str),
        model=model,
        batch_size=args.batch_size,
        device=device,
        input_range=(-1.0, 1.0),
        provenance=generated_provenance,
    )
    save_feature_cache(cache_dir / f"generated_{args.dataset}.npz", generated)

    real_dataset = ImbalancedCIFARImages(
        dataset=args.dataset,
        root=args.data_root,
        train=True,
        download=args.download,
        imbalance_type="balanced",
        imbalance_factor=1.0,
        subset_seed=0,
        normalize="zero_one",
        horizontal_flip=False,
    )
    real_images, real_labels, real_ids = real_dataset.all_samples_with_labels()
    real_provenance = {
        "dataset": args.dataset,
        "split": "balanced_train",
        "extractor": "tf_fid_inception_v3",
        "weights_sha256": model.weights_sha256,
        "evaluator_version": 1,
        "dataset_metadata": real_dataset.metadata(),
    }
    real = extract_inception_features(
        real_images,
        labels=real_labels.numpy(),
        sample_ids=real_ids,
        model=model,
        batch_size=args.batch_size,
        device=device,
        provenance=real_provenance,
    )
    save_feature_cache(cache_dir / f"real_{args.dataset}_balanced_train.npz", real)
    return generated, real


def _resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    raise SystemExit(main())
