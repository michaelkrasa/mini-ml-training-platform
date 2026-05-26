"""Create a model image rollout manifest from a training summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_platform.config import get_settings
from ml_platform.deployment.gates import PromotionPolicy, evaluate_promotion
from ml_platform.utils.io import read_json, write_json


def build_rollout_manifest(
    summary: dict[str, Any],
    *,
    image_repository: str,
    policy: PromotionPolicy | None = None,
) -> dict[str, object]:
    fingerprint = str(summary["dataset_fingerprint"])
    version = str(summary.get("model_version", "unregistered"))
    tag = f"v{version}-{fingerprint[:12]}"
    gate_report = evaluate_promotion(summary, policy) if policy is not None else None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "image": f"{image_repository}:{tag}",
        "image_repository": image_repository,
        "image_tag": tag,
        "run_id": summary.get("run_id"),
        "model_version": summary.get("model_version"),
        "model_alias": summary.get("model_alias"),
        "model_name": summary.get("model_name"),
        "dataset_name": summary.get("dataset_name"),
        "dataset_fingerprint": fingerprint,
        "git_commit": summary.get("git_commit"),
        "training_config": summary.get("training_config", {}),
        "promotion_gate": gate_report,
        "rollout_targets": ["local-inference", "batch-validation"],
    }


def write_rollout_manifest(
    summary_path: Path,
    output_path: Path,
    *,
    image_repository: str,
    policy: PromotionPolicy | None = None,
) -> dict[str, object]:
    manifest = build_rollout_manifest(
        read_json(summary_path),
        image_repository=image_repository,
        policy=policy,
    )
    write_json(output_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=settings.artifacts_root / "deployment" / "rollout-manifest.json",
    )
    parser.add_argument(
        "--image-repository",
        type=str,
        default="mini-ml-platform/scene-presence",
    )
    parser.add_argument("--max-val-loss", type=float, default=None)
    parser.add_argument("--min-val-macro-f1", type=float, default=None)
    parser.add_argument("--min-val-label-accuracy", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = None
    if any(
        threshold is not None
        for threshold in (
            args.max_val_loss,
            args.min_val_macro_f1,
            args.min_val_label_accuracy,
        )
    ):
        policy = PromotionPolicy(
            max_val_loss=args.max_val_loss,
            min_val_macro_f1=args.min_val_macro_f1,
            min_val_label_accuracy=args.min_val_label_accuracy,
        )
    report = write_rollout_manifest(
        args.summary_path,
        args.output_path,
        image_repository=args.image_repository,
        policy=policy,
    )
    print(report)


if __name__ == "__main__":
    main()
