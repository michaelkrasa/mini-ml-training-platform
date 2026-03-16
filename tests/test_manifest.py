from pathlib import Path

from PIL import Image

from ml_platform.config import PlatformSettings
from ml_platform.data.manifest import build_dataset_manifest, manifest_records


def _settings(root: Path) -> PlatformSettings:
    dataset_root = root / "dataset"
    artifacts_root = root / "artifacts"
    mlflow_root = root / "mlflow"
    mlartifacts_root = root / "mlartifacts"
    training_output_root = artifacts_root / "training"
    settings = PlatformSettings(
        project_root=root,
        dataset_root=dataset_root,
        images_dir=dataset_root / "images",
        labels_dir=dataset_root / "labels",
        manifests_dir=dataset_root / "manifests",
        artifacts_root=artifacts_root,
        mlflow_root=mlflow_root,
        mlartifacts_root=mlartifacts_root,
        mlflow_tracking_uri="http://127.0.0.1:5000",
        mlflow_experiment_name="test-experiment",
        mlflow_model_name="test-model",
        mlflow_model_alias="champion",
        model_refresh_interval_seconds=30,
        training_seed=7,
        training_output_root=training_output_root,
    )
    settings.ensure_directories()
    return settings


def _write_sample(dataset_root: Path, sample_id: str, labels: list[str]) -> None:
    image_path = dataset_root / "images" / f"{sample_id}.png"
    label_path = dataset_root / "labels" / f"{sample_id}.txt"
    Image.new("RGB", (32, 32), color=(40, 70, 100)).save(image_path)
    label_path.write_text(
        "\n".join(
            f"{label} 0.0 0 0.0 4 4 20 20 0 0 0 0 0 0 0" for label in labels
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_dataset_manifest_is_deterministic(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_sample(settings.dataset_root, "000000", ["Car"])
    _write_sample(settings.dataset_root, "000001", ["Pedestrian", "Cyclist"])
    _write_sample(settings.dataset_root, "000002", ["Truck"])

    manifest_a = build_dataset_manifest(settings, val_ratio=0.34, seed=7, persist=True)
    manifest_b = build_dataset_manifest(settings, val_ratio=0.34, seed=7, persist=False)

    assert manifest_a.fingerprint == manifest_b.fingerprint
    assert manifest_a.split_counts() == {"train": 2, "val": 1}
    assert (settings.manifests_dir / "latest.json").exists()

    train_records = manifest_records(
        manifest=manifest_a,
        dataset_root=settings.dataset_root,
        split="train",
    )
    assert len(train_records) == 2
