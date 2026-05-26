from ml_platform.deployment.gates import PromotionPolicy, evaluate_promotion
from ml_platform.deployment.package import build_rollout_manifest


def test_evaluate_promotion_requires_all_checks_to_pass() -> None:
    summary = {
        "run_id": "run-1",
        "model_version": "3",
        "dataset_fingerprint": "abcdef123456",
        "history": [
            {
                "epoch": 1,
                "val_loss": 0.42,
                "val_macro_f1": 0.71,
                "val_label_accuracy": 0.83,
            }
        ],
    }

    report = evaluate_promotion(
        summary,
        PromotionPolicy(
            max_val_loss=0.5,
            min_val_macro_f1=0.7,
            min_val_label_accuracy=0.8,
        ),
    )

    assert report["passed"] is True
    assert len(report["checks"]) == 3


def test_build_rollout_manifest_uses_model_version_and_dataset_fingerprint() -> None:
    summary = {
        "run_id": "run-1",
        "model_version": "9",
        "model_alias": "champion",
        "model_name": "kitti-presence-classifier",
        "dataset_name": "kitti_scene_presence",
        "dataset_fingerprint": "abcdef1234567890",
        "history": [{"epoch": 1, "val_loss": 0.2}],
    }

    manifest = build_rollout_manifest(
        summary,
        image_repository="registry.example.com/perception",
        policy=PromotionPolicy(max_val_loss=0.3),
    )

    assert manifest["image"] == "registry.example.com/perception:v9-abcdef123456"
    assert manifest["promotion_gate"]["passed"] is True
