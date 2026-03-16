"""Metrics for the multi-label scene presence task."""

from __future__ import annotations

import torch


def batch_classification_stats(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    label_accuracy = (predictions == targets).float().mean().item()
    exact_match = (predictions == targets).all(dim=1).float().mean().item()

    true_positive = (predictions * targets).sum(dim=0)
    predicted_positive = predictions.sum(dim=0)
    actual_positive = targets.sum(dim=0)

    precision = true_positive / (predicted_positive + 1e-8)
    recall = true_positive / (actual_positive + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    active_classes = (predicted_positive + actual_positive) > 0
    macro_f1 = f1[active_classes].mean().item() if active_classes.any() else 0.0

    stats = {
        "label_accuracy": label_accuracy,
        "exact_match": exact_match,
        "macro_f1": macro_f1,
    }
    row_sums = targets.sum(dim=1)
    if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        top1_accuracy = (
            logits.argmax(dim=1) == targets.argmax(dim=1)
        ).float().mean().item()
        stats["top1_accuracy"] = top1_accuracy
    return stats
