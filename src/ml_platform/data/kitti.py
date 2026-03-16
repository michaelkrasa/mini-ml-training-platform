"""Helpers for reading KITTI-style image and label directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml_platform.constants import DEFAULT_IMAGE_SIZE, KITTI_CLASS_NAMES, SUPPORTED_IMAGE_SUFFIXES


@dataclass(frozen=True)
class KittiRecord:
    sample_id: str
    image_path: Path
    label_path: Path
    labels: tuple[str, ...]
    target: tuple[float, ...]


def _ordered_unique(labels: Iterable[str], class_names: Sequence[str]) -> tuple[str, ...]:
    label_set = set(labels)
    return tuple(class_name for class_name in class_names if class_name in label_set)


def encode_labels(labels: Iterable[str], class_names: Sequence[str] = KITTI_CLASS_NAMES) -> tuple[float, ...]:
    label_set = set(labels)
    return tuple(1.0 if class_name in label_set else 0.0 for class_name in class_names)


def parse_kitti_label_file(
    label_path: Path, class_names: Sequence[str] = KITTI_CLASS_NAMES
) -> tuple[str, ...]:
    labels: list[str] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            class_name = stripped.split()[0]
            if class_name == "DontCare":
                continue
            if class_name in class_names:
                labels.append(class_name)
    return _ordered_unique(labels, class_names)


def preprocess_image(
    image: Image.Image, image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    resized = image.convert("RGB").resize(image_size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(chw)


def discover_records(
    images_dir: Path,
    labels_dir: Path,
    class_names: Sequence[str] = KITTI_CLASS_NAMES,
) -> list[KittiRecord]:
    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        return []

    missing_labels: list[str] = []
    records: list[KittiRecord] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(label_path.name)
            continue
        labels = parse_kitti_label_file(label_path, class_names=class_names)
        records.append(
            KittiRecord(
                sample_id=image_path.stem,
                image_path=image_path,
                label_path=label_path,
                labels=labels,
                target=encode_labels(labels, class_names=class_names),
            )
        )

    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise FileNotFoundError(
            f"Missing label files for {len(missing_labels)} samples. First few: {preview}"
        )
    return records


class KittiScenePresenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """A lightweight perception task built from KITTI labels.

    Each sample predicts a multi-hot vector indicating whether each class is present
    anywhere in the scene. This keeps training fast while preserving a perception-style
    dataset contract.
    """

    def __init__(
        self,
        records: Sequence[KittiRecord],
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.records = list(records)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            features = preprocess_image(image, image_size=self.image_size)
        target = torch.tensor(record.target, dtype=torch.float32)
        return features, target, record.sample_id
