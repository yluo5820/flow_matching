import gzip
import json
import struct
from pathlib import Path

import numpy as np
import torch

from fm_lab.diagnostics.mnist_eval import MNISTClassifier, MNISTEvalConfig, evaluate_mnist_run
from fm_lab.utils.config import save_config


def test_mnist_classifier_forward_shape() -> None:
    classifier = MNISTClassifier()
    logits = classifier(torch.randn(5, 28 * 28))

    assert logits.shape == (5, 10)


def test_evaluate_mnist_run_writes_metrics_and_plots(tmp_path) -> None:
    data_root = tmp_path / "mnist"
    _write_fake_mnist(data_root, split="train", count=12)
    _write_fake_mnist(data_root, split="test", count=8)
    run_dir = tmp_path / "run"
    (run_dir / "samples").mkdir(parents=True)
    config = {
        "experiment": {"name": "fake_mnist"},
        "data": {
            "name": "mnist",
            "root": str(data_root),
            "train": True,
            "download": False,
            "normalize": "minus_one_one",
        },
        "sampling": {"nfe": 2},
    }
    save_config(config, run_dir / "config.yaml")
    target = torch.linspace(-1.0, 1.0, 8 * 28 * 28).reshape(8, 28 * 28).numpy()
    generated = np.clip(target + 0.05, -1.0, 1.0)
    np.save(run_dir / "samples" / "target_reference.npy", target)
    np.save(run_dir / "samples" / "euler_nfe2.npy", generated)

    result = evaluate_mnist_run(
        MNISTEvalConfig(
            run_dir=run_dir,
            solver="euler",
            nfe=2,
            max_samples=8,
            reference_samples=8,
            nearest_neighbors=3,
            skip_classifier=True,
        )
    )

    json_path = Path(result["outputs"]["json"])
    csv_path = Path(result["outputs"]["csv"])
    plot_path = Path(result["plots"]["nearest_neighbors"])

    assert json_path.exists()
    assert csv_path.exists()
    assert plot_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["pixel_stats"]["generated"]["frac_out_of_range"] == 0.0
    assert "classifier" not in payload
    assert payload["nearest_neighbors"]["train_l2_mean"] >= 0.0
    assert payload["outputs"]["json"] == str(json_path)


def _write_fake_mnist(root: Path, *, split: str, count: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    prefix = "train" if split == "train" else "t10k"
    images_path = root / f"{prefix}-images-idx3-ubyte.gz"
    labels_path = root / f"{prefix}-labels-idx1-ubyte.gz"
    images = np.zeros((count, 28, 28), dtype=np.uint8)
    for idx in range(count):
        images[idx, 4:24, 4 + idx % 10 : 6 + idx % 10] = 255
    labels = np.arange(count, dtype=np.uint8) % 10
    with gzip.open(images_path, "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, count, 28, 28))
        handle.write(images.reshape(-1).tobytes())
    with gzip.open(labels_path, "wb") as handle:
        handle.write(struct.pack(">II", 2049, count))
        handle.write(labels.tobytes())
