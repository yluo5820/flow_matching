import json

import numpy as np
import pytest

from fm_lab.evaluation.cache import FeatureCache, save_feature_cache
from fm_lab.evaluation.report import evaluate_feature_caches, write_evaluation_report
from fm_lab.experiments.run_imbdiff_eval import main, parse_args


def _feature_cache(offset: float = 0.0) -> FeatureCache:
    labels = np.repeat(np.arange(3), 4)
    base = np.stack((labels, np.tile(np.arange(4), 3)), axis=1).astype(np.float32)
    probabilities = np.eye(3, dtype=np.float32)[labels] * 0.8 + 0.2 / 3
    return FeatureCache(
        features=base + offset,
        probabilities=probabilities,
        labels=labels,
        sample_ids=np.asarray([f"sample-{index}" for index in range(12)]),
        provenance={"dataset": "fake", "evaluator_version": 1, "offset": offset},
    )


def test_evaluation_report_contains_all_metrics_and_extensions() -> None:
    real = _feature_cache()
    generated = _feature_cache()

    report = evaluate_feature_caches(
        generated,
        real,
        class_counts=[100, 10, 1],
        repeats=2,
        overall_samples=12,
        seed=5,
        kid_subsets=2,
        kid_subset_size=6,
        recall_k=2,
        inception_splits=2,
    )

    assert set(report["metrics"]) == {
        "fid",
        "kid",
        "recall",
        "inception_score",
        "classwise_fid",
        "group_fid",
    }
    assert report["metrics"]["fid"]["mean"] == 0.0
    assert report["metrics"]["recall"]["mean"] == 1.0
    assert report["groups"] == {"many": [0], "medium": [1], "few": [2]}
    assert report["provenance"]["fid_kid_compatibility"] == "ImbDiff-CM reference"
    assert report["provenance"]["extended_metrics"] == [
        "recall",
        "inception_score",
        "classwise_fid",
        "group_fid",
    ]


def test_evaluation_report_and_cli_default_to_paper_recall_k_five() -> None:
    report = evaluate_feature_caches(
        _feature_cache(),
        _feature_cache(),
        class_counts=[100, 10, 1],
        repeats=1,
        overall_samples=12,
        kid_subsets=1,
        kid_subset_size=6,
        inception_splits=1,
    )

    assert report["provenance"]["recall_k"] == 5
    assert parse_args(["--class-counts", "counts.json", "--output-dir", "report"]).recall_k == 5


def test_screening_profile_skips_expensive_metrics() -> None:
    report = evaluate_feature_caches(
        _feature_cache(),
        _feature_cache(),
        class_counts=[100, 10, 1],
        repeats=1,
        overall_samples=12,
        kid_subsets=1,
        kid_subset_size=6,
        inception_splits=1,
        compute_recall=False,
        compute_classwise_fid=False,
    )

    assert set(report["metrics"]) == {"fid", "kid", "inception_score", "group_fid"}
    assert report["provenance"]["extended_metrics"] == [
        "inception_score",
        "group_fid",
    ]
    assert report["provenance"]["compute_recall"] is False
    assert report["provenance"]["compute_classwise_fid"] is False
    args = parse_args(
        [
            "--class-counts",
            "counts.json",
            "--output-dir",
            "report",
            "--skip-recall",
            "--skip-classwise-fid",
        ]
    )
    assert args.skip_recall is True
    assert args.skip_classwise_fid is True


def test_balanced_evaluation_rejects_unequal_generated_class_counts() -> None:
    generated = _feature_cache()
    generated.labels[-1] = 1

    with pytest.raises(ValueError, match="equal generated samples per class"):
        evaluate_feature_caches(
            generated,
            _feature_cache(),
            class_counts=[100, 10, 1],
            repeats=1,
            overall_samples=12,
            kid_subsets=1,
            kid_subset_size=6,
            recall_k=2,
            inception_splits=1,
            require_balanced_generated=True,
        )


def test_fashion_extensions_report_tail_and_conditional_diagnostics() -> None:
    report = evaluate_feature_caches(
        _feature_cache(),
        _feature_cache(),
        class_counts=[100, 10, 1],
        repeats=1,
        overall_samples=12,
        kid_subsets=1,
        kid_subset_size=6,
        recall_k=2,
        inception_splits=1,
        require_balanced_generated=True,
        per_class_recall=True,
        conditional_diagnostics=True,
    )

    assert report["metrics"]["macro_classwise_fid"]["mean"] == 0.0
    assert report["metrics"]["worst_class_fid"]["mean"] == 0.0
    assert set(report["metrics"]["classwise_recall"]) == {
        "class_0",
        "class_1",
        "class_2",
    }
    assert all(
        value["mean"] == 1.0
        for value in report["metrics"]["classwise_recall"].values()
    )
    assert report["conditional"]["requested_class_accuracy"] == 1.0
    assert report["conditional"]["mean_requested_class_probability"] == pytest.approx(
        0.8 + 0.2 / 3
    )
    assert report["conditional"]["predicted_class_histogram"] == [4, 4, 4]
    assert report["conditional"]["confusion_matrix"] == [
        [4, 0, 0],
        [0, 4, 0],
        [0, 0, 4],
    ]


def test_balanced_subsampling_keeps_equal_class_counts() -> None:
    labels = np.repeat(np.arange(3), 6)
    features = np.stack((labels, np.tile(np.arange(6), 3)), axis=1).astype(np.float32)
    cache = FeatureCache(
        features=features,
        probabilities=np.eye(3, dtype=np.float32)[labels],
        labels=labels,
        sample_ids=np.arange(len(labels)).astype(str),
        provenance={"dataset": "fake"},
    )

    report = evaluate_feature_caches(
        cache,
        cache,
        class_counts=[100, 10, 1],
        repeats=1,
        overall_samples=12,
        kid_subsets=1,
        kid_subset_size=6,
        recall_k=2,
        inception_splits=1,
        require_balanced_generated=True,
    )

    assert report["repeats"][0]["generated_class_counts"] == [4, 4, 4]


def test_balanced_subsampling_requires_divisible_sample_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        evaluate_feature_caches(
            _feature_cache(),
            _feature_cache(),
            class_counts=[100, 10, 1],
            repeats=1,
            overall_samples=11,
            kid_subsets=1,
            kid_subset_size=6,
            recall_k=2,
            inception_splits=1,
            require_balanced_generated=True,
        )


def test_write_report_creates_json_and_flat_csv(tmp_path) -> None:
    report = evaluate_feature_caches(
        _feature_cache(),
        _feature_cache(),
        class_counts=[100, 10, 1],
        repeats=1,
        overall_samples=12,
        seed=0,
        kid_subsets=1,
        kid_subset_size=6,
        recall_k=2,
        inception_splits=1,
    )

    paths = write_evaluation_report(report, tmp_path)

    assert json.loads(paths["json"].read_text())["metrics"]["fid"]["mean"] == 0.0
    assert "classwise_fid.class_0" in paths["csv"].read_text()


def test_cached_feature_cli_writes_report(tmp_path) -> None:
    generated_path = tmp_path / "generated.npz"
    real_path = tmp_path / "real.npz"
    save_feature_cache(generated_path, _feature_cache())
    save_feature_cache(real_path, _feature_cache())
    counts_path = tmp_path / "counts.json"
    counts_path.write_text(json.dumps([100, 10, 1]))
    output_dir = tmp_path / "report"

    exit_code = main(
        [
            "--generated-cache",
            str(generated_path),
            "--real-cache",
            str(real_path),
            "--class-counts",
            str(counts_path),
            "--output-dir",
            str(output_dir),
            "--repeats",
            "1",
            "--overall-samples",
            "12",
            "--kid-subsets",
            "1",
            "--kid-subset-size",
            "6",
            "--recall-k",
            "2",
            "--inception-splits",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "metrics.csv").exists()
