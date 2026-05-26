from ml_platform.data.manifest import DatasetManifest, DatasetSample
from ml_platform.data.mining import rank_manifest_samples


def _sample(sample_id: str, labels: tuple[str, ...], split: str = "train") -> DatasetSample:
    return DatasetSample(
        sample_id=sample_id,
        image_path=f"images/{sample_id}.png",
        label_path=f"labels/{sample_id}.txt",
        labels=labels,
        target=(1.0, 0.0, 0.0),
        split=split,
        image_sha256="image",
        label_sha256="label",
    )


def test_rank_manifest_samples_prioritizes_rare_multi_object_examples() -> None:
    manifest = DatasetManifest(
        dataset_root="/tmp/dataset",
        dataset_name="kitti_scene_presence",
        task_type="multi_label_scene_presence",
        fingerprint="abc123",
        created_at="2026-05-26T00:00:00Z",
        class_names=("Car", "Truck", "Cyclist"),
        seed=7,
        val_ratio=0.2,
        samples=(
            _sample("common-1", ("Car",)),
            _sample("common-2", ("Car",)),
            _sample("rare", ("Truck", "Cyclist")),
        ),
    )

    ranked = rank_manifest_samples(manifest)

    assert ranked[0].sample_id == "rare"
    assert ranked[0].score > ranked[1].score
    assert "multi_object=0.150" in ranked[0].reasons
