"""Dataset metadata used to keep training and inference dataset-aware."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ml_platform.config import PlatformSettings
from ml_platform.constants import KITTI_CLASS_NAMES, MULTI_LABEL_TASK
from ml_platform.utils.io import read_json, write_json


@dataclass(frozen=True)
class DatasetProfile:
    dataset_name: str
    source: str
    description: str
    class_names: tuple[str, ...]
    task_type: str = MULTI_LABEL_TASK

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def dataset_profile_path(settings: PlatformSettings) -> Path:
    return settings.manifests_dir / "dataset_profile.json"


def default_dataset_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="kitti_scene_presence",
        source="manual",
        description="KITTI-style perception labels interpreted as scene-level class presence.",
        class_names=KITTI_CLASS_NAMES,
    )


def load_dataset_profile(path: Path) -> DatasetProfile:
    payload = read_json(path)
    return DatasetProfile(
        dataset_name=payload["dataset_name"],
        source=payload["source"],
        description=payload["description"],
        class_names=tuple(payload["class_names"]),
        task_type=payload.get("task_type", MULTI_LABEL_TASK),
    )


def resolve_dataset_profile(settings: PlatformSettings) -> DatasetProfile:
    path = dataset_profile_path(settings)
    if not path.exists():
        return default_dataset_profile()
    return load_dataset_profile(path)


def write_dataset_profile(settings: PlatformSettings, profile: DatasetProfile) -> Path:
    path = dataset_profile_path(settings)
    write_json(path, profile.to_dict())
    return path
