from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from fm_lab.image_diagnostics.config import InputConfig
from fm_lab.image_diagnostics.dataset_loader import load_dataset


def test_voc2012_loader_reads_xml_labels_and_segmentation_metadata(tmp_path: Path) -> None:
    root = tmp_path / "VOCdevkit" / "VOC2012"
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()
    (root / "ImageSets" / "Main").mkdir(parents=True)
    (root / "SegmentationClass").mkdir()
    samples = [
        ("2007_000001", ["cat", "person"]),
        ("2007_000002", ["dog"]),
    ]
    for index, (sample_id, labels) in enumerate(samples):
        _write_image(root / "JPEGImages" / f"{sample_id}.jpg", value=index * 80)
        _write_annotation(root / "Annotations" / f"{sample_id}.xml", labels=labels)
    Image.fromarray(np.ones((20, 20), dtype=np.uint8), mode="L").save(
        root / "SegmentationClass" / "2007_000001.png"
    )
    (root / "ImageSets" / "Main" / "trainval.txt").write_text(
        "\n".join(sample_id for sample_id, _labels in samples),
        encoding="utf-8",
    )

    bundle = load_dataset(
        InputConfig(
            type="voc2012",
            dataset_root=str(tmp_path / "VOCdevkit"),
            split="trainval",
            image_size=32,
            thumbnail_mode="none",
        ),
        project_root=tmp_path,
    )

    assert bundle.vectors is not None
    assert bundle.vectors.shape == (2, 32 * 32 * 3)
    assert bundle.image_shape == (32, 32, 3)
    assert bundle.total_rows == 2
    assert set(bundle.metadata["label"]) == {"cat", "dog"}
    first = bundle.metadata.iloc[0]
    assert first["object_classes"] == "cat,person"
    assert bool(first["has_segmentation_mask"]) is True


def _write_image(path: Path, *, value: int) -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    image[..., 0] = value
    image[8:32, 10:40, 1] = 255 - value
    Image.fromarray(image, mode="RGB").save(path)


def _write_annotation(path: Path, *, labels: list[str]) -> None:
    objects = "\n".join(
        f"<object><name>{label}</name></object>"
        for label in labels
    )
    path.write_text(
        f"""
<annotation>
  <size><width>50</width><height>40</height><depth>3</depth></size>
  {objects}
</annotation>
""".strip(),
        encoding="utf-8",
    )
