"""Model selection gates for closed-loop deployment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml_platform.utils.io import read_json, write_json


@dataclass(frozen=True)
class PromotionPolicy:
    max_val_loss: float | None = None
    min_val_macro_f1: float | None = None
    min_val_label_accuracy: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _best_epoch(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        raise ValueError("Training summary does not contain epoch history.")
    return min(history, key=lambda row: row.get("val_loss", row.get("train_loss", float("inf"))))


def evaluate_promotion(summary: dict[str, Any], policy: PromotionPolicy) -> dict[str, object]:
    best = _best_epoch(summary.get("history", []))
    checks: list[dict[str, object]] = []

    def add_check(name: str, passed: bool, observed: float | None, threshold: float) -> None:
        checks.append(
            {
                "name": name,
                "passed": passed,
                "observed": observed,
                "threshold": threshold,
            }
        )

    if policy.max_val_loss is not None:
        observed = best.get("val_loss", best.get("train_loss"))
        add_check(
            "max_val_loss",
            observed is not None and float(observed) <= policy.max_val_loss,
            None if observed is None else float(observed),
            policy.max_val_loss,
        )
    if policy.min_val_macro_f1 is not None:
        observed = best.get("val_macro_f1", best.get("train_macro_f1"))
        add_check(
            "min_val_macro_f1",
            observed is not None and float(observed) >= policy.min_val_macro_f1,
            None if observed is None else float(observed),
            policy.min_val_macro_f1,
        )
    if policy.min_val_label_accuracy is not None:
        observed = best.get("val_label_accuracy", best.get("train_label_accuracy"))
        add_check(
            "min_val_label_accuracy",
            observed is not None and float(observed) >= policy.min_val_label_accuracy,
            None if observed is None else float(observed),
            policy.min_val_label_accuracy,
        )

    passed = bool(checks) and all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "policy": policy.to_dict(),
        "best_epoch": best,
        "checks": checks,
        "run_id": summary.get("run_id"),
        "model_version": summary.get("model_version"),
        "dataset_fingerprint": summary.get("dataset_fingerprint"),
    }


def write_promotion_report(
    summary_path: Path,
    output_path: Path,
    policy: PromotionPolicy,
) -> dict[str, object]:
    report = evaluate_promotion(read_json(summary_path), policy)
    write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-val-loss", type=float, default=None)
    parser.add_argument("--min-val-macro-f1", type=float, default=None)
    parser.add_argument("--min-val-label-accuracy", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = write_promotion_report(
        args.summary_path,
        args.output_path,
        PromotionPolicy(
            max_val_loss=args.max_val_loss,
            min_val_macro_f1=args.min_val_macro_f1,
            min_val_label_accuracy=args.min_val_label_accuracy,
        ),
    )
    print(report)


if __name__ == "__main__":
    main()
