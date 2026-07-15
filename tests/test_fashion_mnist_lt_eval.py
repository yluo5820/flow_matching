import hashlib

import numpy as np
import pytest

from fm_lab.evaluation.cache import FeatureCache, save_feature_cache
from fm_lab.experiments.run_fashion_mnist_lt_eval import _sha256_file, main, parse_args


def _cache(
    *,
    weights: str = "weights-a",
    samples_per_class: int = 2,
    split: str = "generated",
) -> FeatureCache:
    labels = np.repeat(np.arange(10), samples_per_class)
    features = np.zeros((len(labels), 128), dtype=np.float32)
    features[:, :2] = np.stack(
        (labels, np.tile(np.arange(samples_per_class), 10)), axis=1
    )
    probabilities = np.eye(10, dtype=np.float32)[labels] * 0.9 + 0.1 / 10
    sample_ids = np.arange(len(labels)).astype(str)
    subset_sha256 = hashlib.sha256(
        np.arange(len(labels), dtype=np.int64).tobytes()
    ).hexdigest()
    provenance = {
        "dataset": "fashion_mnist",
        "extractor": "fashion_mnist_classifier",
        "weights_sha256": weights,
        "preprocessing": "clamp_to_classifier_input_range",
        "image_shape": [1, 28, 28],
        "normalize": "minus_one_one",
        "evaluator_version": 1,
        "architecture": "mnist_classifier_v1",
        "feature_layer": "penultimate_128",
        "feature_dimension": 128,
        "class_order": list(range(10)),
        "minimum_accuracy": 0.9,
        "test_accuracy": 0.93,
        "split": split,
    }
    if split == "official_test":
        provenance["dataset_metadata"] = {
            "dataset": "fashion_mnist",
            "train": False,
            "n_images": len(labels),
            "class_counts": [samples_per_class] * 10,
            "subset_sha256": subset_sha256,
        }
    else:
        provenance.update(
            {
                "source_samples_sha256": "samples-hash",
                "source_labels_sha256": "labels-hash",
                "generative_checkpoint_sha256": "checkpoint-hash",
                "generative_weights": "raw",
                "generation_method": "flow_matching",
                "sampler": "euler",
                "nfe": 64,
                "guidance_scale": 2.0,
                "generation_seed": 0,
            }
        )
    return FeatureCache(
        features=features,
        probabilities=probabilities,
        labels=labels,
        sample_ids=sample_ids,
        provenance=provenance,
    )


def test_fashion_cli_uses_canonical_balanced_defaults() -> None:
    args = parse_args(["--output-dir", "report"])

    assert (
        args.classifier_checkpoint
        == "artifacts/fashion_mnist_lt_evaluator_minus_one_one.pt"
    )
    assert args.samples_per_class == 1000
    assert args.overall_samples == 10_000
    assert args.imbalance_factor == 0.01
    assert args.minimum_accuracy == 0.89
    assert args.guidance_scale == 1.0
    assert args.generative_weights == "raw"


def test_fashion_cli_requires_cache_paths_as_a_pair(tmp_path) -> None:
    path = tmp_path / "generated.npz"
    save_feature_cache(path, _cache())

    with pytest.raises(ValueError, match="Both --generated-cache and --real-cache"):
        main(
            [
                "--generated-cache",
                str(path),
                "--output-dir",
                str(tmp_path / "report"),
            ]
        )


def test_fashion_cli_rejects_mismatched_evaluator_provenance(tmp_path) -> None:
    generated_path = tmp_path / "generated.npz"
    real_path = tmp_path / "real.npz"
    save_feature_cache(generated_path, _cache(weights="generated"))
    save_feature_cache(
        real_path,
        _cache(weights="real", samples_per_class=1000, split="official_test"),
    )

    with pytest.raises(ValueError, match="weights_sha256"):
        main(
            [
                "--generated-cache",
                str(generated_path),
                "--real-cache",
                str(real_path),
                "--samples-per-class",
                "2",
                "--overall-samples",
                "20",
                "--output-dir",
                str(tmp_path / "report"),
            ]
        )


def test_fashion_cli_rejects_noncanonical_real_split(tmp_path) -> None:
    generated_path = tmp_path / "generated.npz"
    real_path = tmp_path / "real.npz"
    save_feature_cache(generated_path, _cache())
    save_feature_cache(real_path, _cache(samples_per_class=1000, split="train"))

    with pytest.raises(ValueError, match="official_test"):
        main(
            [
                "--generated-cache",
                str(generated_path),
                "--real-cache",
                str(real_path),
                "--samples-per-class",
                "2",
                "--overall-samples",
                "20",
                "--output-dir",
                str(tmp_path / "report"),
            ]
        )


def test_cached_fashion_cli_writes_balanced_report(tmp_path) -> None:
    generated_path = tmp_path / "generated.npz"
    real_path = tmp_path / "real.npz"
    save_feature_cache(generated_path, _cache())
    save_feature_cache(
        real_path,
        _cache(samples_per_class=1000, split="official_test"),
    )
    output_dir = tmp_path / "report"

    exit_code = main(
        [
            "--generated-cache",
            str(generated_path),
            "--real-cache",
            str(real_path),
            "--samples-per-class",
            "2",
            "--overall-samples",
            "20",
            "--output-dir",
            str(output_dir),
            "--repeats",
            "1",
            "--kid-subsets",
            "1",
            "--kid-subset-size",
            "10",
            "--recall-k",
            "1",
            "--inception-splits",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics.csv").exists()
    report_text = (output_dir / "metrics.json").read_text()
    assert "macro_classwise_fid" in report_text
    assert "reference_calibration" in report_text


def test_sha256_file_changes_when_generated_content_changes(tmp_path) -> None:
    path = tmp_path / "samples.npy"
    path.write_bytes(b"first")
    first = _sha256_file(path)
    path.write_bytes(b"second")

    assert _sha256_file(path) != first
