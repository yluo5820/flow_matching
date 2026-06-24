"""Locally compute the projections used by the full MNIST reference explorer."""

from __future__ import annotations

import json
import logging
import platform
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from sklearn.manifold import TSNE

from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import load_dataset

LOGGER = logging.getLogger("fm_lab.image_diagnostics.reference_projections")

METHOD_FILENAMES = {
    "umap": "mnist_embeddings.json",
    "tsne": "tsne_mnist_embeddings.json",
    "umap-min-dist-0.8": "md08_umap_mnist_embeddings.json",
}


def compute_mnist_reference_projections(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    methods: list[str],
    max_samples: int | None = None,
    overwrite: bool = False,
    n_jobs: int = -1,
) -> dict[str, object]:
    """Compute selected projections in the ordering used by the original explorer."""

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        InputConfig(
            type="mnist",
            dataset_root=str(dataset_root),
            split="all",
            order="mldata",
            thumbnail_mode="atlas",
            max_samples=max_samples,
            download=False,
        ),
        thumbnail_dir=None,
    )
    assert dataset.vectors is not None
    # fetch_mldata exposed MNIST pixel values on the original 0-255 scale.
    pixels = np.asarray(dataset.vectors * 255.0, dtype=np.float32)
    labels = dataset.metadata["label"].astype(int).to_numpy()
    _write_json(output_path / "mnist_labels.json", labels.tolist(), overwrite=overwrite)

    outputs: dict[str, str] = {}
    runtimes: dict[str, float] = {}
    for method in methods:
        path = output_path / METHOD_FILENAMES[method]
        if path.exists() and not overwrite:
            LOGGER.info("Keeping existing %s projection: %s", method, path)
            outputs[method] = str(path)
            continue
        LOGGER.info("Computing %s for %d MNIST samples.", method, len(pixels))
        started = time.perf_counter()
        coordinates = _compute_method(pixels, method=method, n_jobs=n_jobs)
        _write_json(path, coordinates.tolist(), overwrite=True)
        runtimes[method] = time.perf_counter() - started
        outputs[method] = str(path)
        LOGGER.info("Saved %s in %.2f seconds: %s", method, runtimes[method], path)

    manifest = {
        "dataset": "MNIST original",
        "samples": len(pixels),
        "features": pixels.shape[1],
        "ordering": "training stable-sorted by label, then test stable-sorted by label",
        "pixel_scale": [0.0, 255.0],
        "methods": methods,
        "parameters": {
            "umap": {
                "n_components": 2,
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "random_state": 42,
            },
            "tsne": {
                "n_components": 2,
                "perplexity": 30.0,
                "early_exaggeration": 12.0,
                "learning_rate": 200.0,
                "max_iter": 1000,
                "metric": "euclidean",
                "init": "random",
                "method": "barnes_hut",
                "angle": 0.5,
                "random_state": 0,
            },
            "umap-min-dist-0.8": {
                "n_components": 2,
                "n_neighbors": 15,
                "min_dist": 0.8,
                "metric": "euclidean",
                "random_state": 42,
            },
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "umap-learn": version("umap-learn"),
        },
        "runtime_seconds": runtimes,
        "outputs": outputs,
    }
    _write_json(output_path / "manifest.json", manifest, overwrite=True, indent=2)
    return manifest


def _compute_method(
    pixels: np.ndarray,
    *,
    method: str,
    n_jobs: int,
) -> np.ndarray:
    if method in {"umap", "umap-min-dist-0.8"}:
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError(
                'UMAP requires umap-learn. Install ".[image-diagnostics]".'
            ) from exc
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.8 if method == "umap-min-dist-0.8" else 0.1,
            metric="euclidean",
            random_state=42,
            n_jobs=1,
            verbose=True,
        )
        return np.asarray(reducer.fit_transform(pixels), dtype=np.float32)
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=30.0,
            early_exaggeration=12.0,
            learning_rate=200.0,
            max_iter=1000,
            n_iter_without_progress=300,
            min_grad_norm=1.0e-7,
            metric="euclidean",
            init="random",
            verbose=1,
            random_state=0,
            method="barnes_hut",
            angle=0.5,
            n_jobs=n_jobs,
        )
        return np.asarray(reducer.fit_transform(pixels), dtype=np.float32)
    raise ValueError(f"Unsupported reference projection method: {method}")


def _write_json(
    path: Path,
    value: object,
    *,
    overwrite: bool,
    indent: int | None = None,
) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(value, handle, indent=indent)
        handle.write("\n")
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
