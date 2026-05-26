"""Active-learning style data mining over dataset manifests."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from ml_platform.config import get_settings
from ml_platform.data.manifest import DatasetManifest, DatasetSample, load_manifest
from ml_platform.utils.io import write_json


@dataclass(frozen=True)
class MiningCandidate:
    sample_id: str
    image_path: str
    labels: tuple[str, ...]
    split: str
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _label_counts(manifest: DatasetManifest) -> dict[str, int]:
    counts = {class_name: 0 for class_name in manifest.class_names}
    for sample in manifest.samples:
        for label in sample.labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def score_sample(
    sample: DatasetSample,
    label_counts: dict[str, int],
    *,
    train_split_bonus: float = 0.25,
) -> MiningCandidate:
    rarity_score = sum(1.0 / max(label_counts.get(label, 0), 1) for label in sample.labels)
    multi_object_bonus = max(len(sample.labels) - 1, 0) * 0.15
    split_bonus = train_split_bonus if sample.split == "train" else 0.0
    score = rarity_score + multi_object_bonus + split_bonus

    reasons = [f"rarity={rarity_score:.3f}"]
    if multi_object_bonus:
        reasons.append(f"multi_object={multi_object_bonus:.3f}")
    if split_bonus:
        reasons.append(f"train_split={split_bonus:.3f}")

    return MiningCandidate(
        sample_id=sample.sample_id,
        image_path=sample.image_path,
        labels=sample.labels,
        split=sample.split,
        score=round(score, 6),
        reasons=tuple(reasons),
    )


def rank_manifest_samples(manifest: DatasetManifest) -> list[MiningCandidate]:
    counts = _label_counts(manifest)
    candidates = [score_sample(sample, counts) for sample in manifest.samples]
    return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.sample_id))


def write_mining_report(
    manifest_path: Path,
    output_path: Path,
    *,
    limit: int = 25,
) -> dict[str, object]:
    manifest = load_manifest(manifest_path)
    candidates = rank_manifest_samples(manifest)[:limit]
    report = {
        "dataset_fingerprint": manifest.fingerprint,
        "dataset_name": manifest.dataset_name,
        "strategy": "class_rarity_plus_multi_object_density",
        "candidate_count": len(candidates),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=settings.manifests_dir / "latest.json",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=settings.artifacts_root / "mining" / "active-learning-candidates.json",
    )
    parser.add_argument("--limit", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = write_mining_report(args.manifest_path, args.output_path, limit=args.limit)
    print(report)


if __name__ == "__main__":
    main()
