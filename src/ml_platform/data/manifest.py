"""Dataset manifest generation for reproducible training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from typing import Sequence

from ml_platform.config import PlatformSettings
from ml_platform.constants import KITTI_CLASS_NAMES, MULTI_LABEL_TASK
from ml_platform.data.kitti import KittiRecord, discover_records
from ml_platform.data.profile import resolve_dataset_profile
from ml_platform.utils.hashing import sha256_file, stable_json_digest
from ml_platform.utils.io import read_json, write_json


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image_path: str
    label_path: str
    labels: tuple[str, ...]
    target: tuple[float, ...]
    split: str
    image_sha256: str
    label_sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    dataset_root: str
    dataset_name: str
    task_type: str
    fingerprint: str
    created_at: str
    class_names: tuple[str, ...]
    seed: int
    val_ratio: float
    samples: tuple[DatasetSample, ...]

    def split_counts(self) -> dict[str, int]:
        counts = {"train": 0, "val": 0}
        for sample in self.samples:
            counts[sample.split] = counts.get(sample.split, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["sample_count"] = len(self.samples)
        payload["split_counts"] = self.split_counts()
        return payload


def _choose_val_ids(sample_ids: Sequence[str], seed: int, val_ratio: float) -> set[str]:
    if len(sample_ids) < 2:
        return set()
    ids = list(sample_ids)
    Random(seed).shuffle(ids)
    val_count = max(1, int(round(len(ids) * val_ratio)))
    val_count = min(val_count, len(ids) - 1)
    return set(ids[:val_count])


def build_dataset_manifest(
    settings: PlatformSettings,
    class_names: Sequence[str] | None = None,
    seed: int | None = None,
    val_ratio: float = 0.2,
    persist: bool = True,
) -> DatasetManifest:
    dataset_profile = resolve_dataset_profile(settings)
    resolved_seed = settings.training_seed if seed is None else seed
    resolved_class_names = (
        tuple(class_names) if class_names is not None else dataset_profile.class_names
    )
    records = discover_records(
        images_dir=settings.images_dir,
        labels_dir=settings.labels_dir,
        class_names=resolved_class_names,
    )
    if not records:
        raise RuntimeError(
            f"No dataset records found under {settings.images_dir} and {settings.labels_dir}."
        )

    val_ids = _choose_val_ids([record.sample_id for record in records], resolved_seed, val_ratio)
    samples: list[DatasetSample] = []
    fingerprint_payload: list[dict[str, object]] = []
    for record in records:
        image_sha256 = sha256_file(record.image_path)
        label_sha256 = sha256_file(record.label_path)
        split = "val" if record.sample_id in val_ids else "train"
        sample = DatasetSample(
            sample_id=record.sample_id,
            image_path=str(record.image_path.relative_to(settings.dataset_root)),
            label_path=str(record.label_path.relative_to(settings.dataset_root)),
            labels=record.labels,
            target=record.target,
            split=split,
            image_sha256=image_sha256,
            label_sha256=label_sha256,
        )
        samples.append(sample)
        fingerprint_payload.append(
            {
                "sample_id": sample.sample_id,
                "labels": sample.labels,
                "target": sample.target,
                "image_sha256": image_sha256,
                "label_sha256": label_sha256,
            }
        )

    fingerprint = stable_json_digest(
        {
            "class_names": resolved_class_names,
            "dataset_name": dataset_profile.dataset_name,
            "seed": resolved_seed,
            "samples": fingerprint_payload,
            "task_type": dataset_profile.task_type,
            "val_ratio": val_ratio,
        }
    )
    manifest = DatasetManifest(
        dataset_root=str(settings.dataset_root),
        dataset_name=dataset_profile.dataset_name,
        task_type=dataset_profile.task_type,
        fingerprint=fingerprint,
        created_at=datetime.now(timezone.utc).isoformat(),
        class_names=resolved_class_names,
        seed=resolved_seed,
        val_ratio=val_ratio,
        samples=tuple(samples),
    )

    if persist:
        versioned_path = settings.manifests_dir / f"dataset-manifest-{fingerprint[:12]}.json"
        latest_path = settings.manifests_dir / "latest.json"
        payload = manifest.to_dict()
        write_json(versioned_path, payload)
        write_json(latest_path, payload)
    return manifest


def load_manifest(path: Path) -> DatasetManifest:
    payload = read_json(path)
    samples = tuple(
        DatasetSample(
            sample_id=sample["sample_id"],
            image_path=sample["image_path"],
            label_path=sample["label_path"],
            labels=tuple(sample["labels"]),
            target=tuple(sample["target"]),
            split=sample["split"],
            image_sha256=sample["image_sha256"],
            label_sha256=sample["label_sha256"],
        )
        for sample in payload["samples"]
    )
    return DatasetManifest(
        dataset_root=payload["dataset_root"],
        dataset_name=payload.get("dataset_name", "kitti_scene_presence"),
        task_type=payload.get("task_type", MULTI_LABEL_TASK),
        fingerprint=payload["fingerprint"],
        created_at=payload["created_at"],
        class_names=tuple(payload["class_names"]),
        seed=payload["seed"],
        val_ratio=payload["val_ratio"],
        samples=samples,
    )


def manifest_records(
    manifest: DatasetManifest,
    dataset_root: Path,
    split: str | None = None,
) -> list[KittiRecord]:
    records: list[KittiRecord] = []
    for sample in manifest.samples:
        if split is not None and sample.split != split:
            continue
        records.append(
            KittiRecord(
                sample_id=sample.sample_id,
                image_path=dataset_root / sample.image_path,
                label_path=dataset_root / sample.label_path,
                labels=sample.labels,
                target=sample.target,
            )
        )
    return records
