"""Evaluate completed MNIST image-generation runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fm_lab.diagnostics.mnist_eval import MNISTEvalConfig, evaluate_mnist_run
from fm_lab.experiments.factory import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a completed MNIST fm_lab run.")
    parser.add_argument("--run-dir", required=True, help="Completed MNIST training run directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to writing diagnostics into --run-dir.",
    )
    parser.add_argument("--solver", default="auto", help="Solver sample name, or auto.")
    parser.add_argument("--nfe", type=int, default=64, help="NFE suffix to evaluate.")
    parser.add_argument("--max-samples", type=int, default=256, help="Generated samples to score.")
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=2048,
        help="MNIST training samples used for moment and nearest-neighbor references.",
    )
    parser.add_argument(
        "--nearest-neighbors",
        type=int,
        default=16,
        help="Number of generated/nearest-train pairs to plot.",
    )
    parser.add_argument(
        "--classifier-checkpoint",
        default="artifacts/mnist_classifier.pt",
        help="Classifier checkpoint cache path. Normalization suffix is added automatically.",
    )
    parser.add_argument(
        "--classifier-steps",
        type=int,
        default=1000,
        help="Training steps when the classifier cache does not exist.",
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=256,
        help="Batch size for classifier training/evaluation.",
    )
    parser.add_argument(
        "--classifier-eval-samples",
        type=int,
        default=2048,
        help="Held-out MNIST samples used to estimate classifier accuracy.",
    )
    parser.add_argument("--classifier-lr", type=float, default=1.0e-3)
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Skip classifier recognizability/diversity metrics.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    result = evaluate_mnist_run(
        MNISTEvalConfig(
            run_dir=Path(args.run_dir),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            solver=args.solver,
            nfe=args.nfe,
            max_samples=args.max_samples,
            reference_samples=args.reference_samples,
            nearest_neighbors=args.nearest_neighbors,
            classifier_checkpoint=Path(args.classifier_checkpoint),
            classifier_steps=args.classifier_steps,
            classifier_batch_size=args.classifier_batch_size,
            classifier_eval_samples=args.classifier_eval_samples,
            classifier_lr=args.classifier_lr,
            skip_classifier=args.skip_classifier,
            device=torch.device(device),
        )
    )
    print(f"Wrote MNIST evaluation: {result['outputs']['json']}")


if __name__ == "__main__":
    main()
