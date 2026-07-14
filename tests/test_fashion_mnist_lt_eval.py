import numpy as np
import pytest

from fm_lab.evaluation.cache import FeatureCache, save_feature_cache
from fm_lab.experiments.run_fashion_mnist_lt_eval import main, parse_args


def _cache(*, weights: str = "weights-a") -> FeatureCache:
    labels = np.repeat(np.arange(10), 2)
    features = np.stack((labels, np.tile(np.arange(2), 10)), axis=1).astype(np.float32)
    probabilities = np.eye(10, dtype=np.float32)[labels] * 0.9 + 0.1 / 10
    return FeatureCache(
        features=features,
        probabilities=probabilities,
        labels=labels,
        sample_ids=np.asarray([f"sample-{index}" for index in range(len(labels))]),
        provenance={
            "dataset": "fashion_mnist",
            "extractor": "fashion_mnist_classifier",
            "weights_sha256": weights,
            "preprocessing": "clamp_to_classifier_input_range",
            "image_shape": [1, 28, 28],
            "normalize": "minus_one_one",
        },
    )


def test_fashion_cli_uses_canonical_balanced_defaults() -> None:
    args = parse_args(["--output-dir", "report"])

    assert args.samples_per_class == 1000
    assert args.overall_samples == 10_000
    assert args.imbalance_factor == 0.01
    assert args.minimum_accuracy == 0.9


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
    save_feature_cache(real_path, _cache(weights="real"))

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


def test_cached_fashion_cli_writes_balanced_report(tmp_path) -> None:
    generated_path = tmp_path / "generated.npz"
    real_path = tmp_path / "real.npz"
    save_feature_cache(generated_path, _cache())
    save_feature_cache(real_path, _cache())
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
