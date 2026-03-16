"""Download and convert common benchmark datasets into the platform dataset contract."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image
from torchvision import datasets

from ml_platform.config import get_settings
from ml_platform.constants import SINGLE_LABEL_TASK
from ml_platform.data.profile import DatasetProfile, write_dataset_profile
from ml_platform.utils.io import write_json


@dataclass(frozen=True)
class ValidationDatasetSpec:
    key: str
    display_name: str
    description: str
    classes: tuple[str, ...]
    builder: Callable[[Path, bool, bool], object]


def _sanitize_label_token(name: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    if not token:
        raise ValueError(f"Could not derive a safe class token from {name!r}.")
    return token


def _cifar10_builder(root: Path, train: bool, download: bool) -> datasets.CIFAR10:
    return datasets.CIFAR10(root=root, train=train, download=download)


def _fashion_mnist_builder(
    root: Path, train: bool, download: bool
) -> datasets.FashionMNIST:
    return datasets.FashionMNIST(root=root, train=train, download=download)


def _mnist_builder(root: Path, train: bool, download: bool) -> datasets.MNIST:
    return datasets.MNIST(root=root, train=train, download=download)


SUPPORTED_VALIDATION_DATASETS = {
    "cifar10": ValidationDatasetSpec(
        key="cifar10",
        display_name="CIFAR-10",
        description="Widely used object classification benchmark with ten natural image classes.",
        classes=(
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ),
        builder=_cifar10_builder,
    ),
    "fashion-mnist": ValidationDatasetSpec(
        key="fashion-mnist",
        display_name="Fashion-MNIST",
        description="Widely used grayscale image classification benchmark for clothing categories.",
        classes=(
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ),
        builder=_fashion_mnist_builder,
    ),
    "mnist": ValidationDatasetSpec(
        key="mnist",
        display_name="MNIST",
        description="Classic handwritten digit recognition benchmark.",
        classes=tuple(str(index) for index in range(10)),
        builder=_mnist_builder,
    ),
}


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for existing in path.iterdir():
        if existing.name == ".gitkeep":
            continue
        if existing.is_file():
            existing.unlink()


def _export_split(
    dataset: object,
    split_name: str,
    limit: int | None,
    image_dir: Path,
    label_dir: Path,
    class_names: tuple[str, ...],
) -> int:
    total = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(total):
        image, target = dataset[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        label_token = class_names[int(target)]
        sample_id = f"{split_name}_{index:06d}"
        image.convert("RGB").save(image_dir / f"{sample_id}.png")
        (label_dir / f"{sample_id}.txt").write_text(
            f"{label_token}\n",
            encoding="utf-8",
        )
    return total


def prepare_validation_dataset(
    dataset_key: str,
    max_train_samples: int | None = None,
    max_test_samples: int = 0,
) -> dict[str, object]:
    if dataset_key not in SUPPORTED_VALIDATION_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_VALIDATION_DATASETS))
        raise ValueError(f"Unsupported dataset {dataset_key!r}. Supported values: {supported}")

    settings = get_settings()
    settings.ensure_directories()
    spec = SUPPORTED_VALIDATION_DATASETS[dataset_key]
    class_names = tuple(_sanitize_label_token(name) for name in spec.classes)

    _clear_directory(settings.images_dir)
    _clear_directory(settings.labels_dir)
    _clear_directory(settings.manifests_dir)

    raw_root = settings.artifacts_root / "dataset_cache" / dataset_key
    raw_root.mkdir(parents=True, exist_ok=True)

    train_dataset = spec.builder(raw_root, True, True)
    train_count = _export_split(
        dataset=train_dataset,
        split_name="train",
        limit=max_train_samples,
        image_dir=settings.images_dir,
        label_dir=settings.labels_dir,
        class_names=class_names,
    )

    test_count = 0
    if max_test_samples != 0:
        test_dataset = spec.builder(raw_root, False, True)
        limit = None if max_test_samples < 0 else max_test_samples
        test_count = _export_split(
            dataset=test_dataset,
            split_name="test",
            limit=limit,
            image_dir=settings.images_dir,
            label_dir=settings.labels_dir,
            class_names=class_names,
        )

    profile = DatasetProfile(
        dataset_name=spec.key,
        source=f"torchvision:{spec.display_name}",
        description=spec.description,
        class_names=class_names,
        task_type=SINGLE_LABEL_TASK,
    )
    write_dataset_profile(settings, profile)

    summary = {
        "dataset_name": spec.key,
        "display_name": spec.display_name,
        "description": spec.description,
        "task_type": SINGLE_LABEL_TASK,
        "original_class_names": list(spec.classes),
        "class_names": list(class_names),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "raw_cache_root": str(raw_root),
        "dataset_root": str(settings.dataset_root),
        "export_counts": {
            "train": train_count,
            "test": test_count,
            "total": train_count + test_count,
        },
    }
    write_json(settings.manifests_dir / "source_dataset_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_VALIDATION_DATASETS),
        required=True,
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Use 0 to skip the official test split, or -1 to export all test samples.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = prepare_validation_dataset(
        dataset_key=args.dataset,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    print(summary)


if __name__ == "__main__":
    main()
