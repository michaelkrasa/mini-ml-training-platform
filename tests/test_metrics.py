import torch

from ml_platform.training.metrics import batch_classification_stats


def test_batch_classification_stats_reports_perfect_predictions() -> None:
    logits = torch.tensor(
        [
            [9.0, -9.0, 6.0],
            [8.0, -5.0, -6.0],
            [-4.0, 7.0, -7.0],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    stats = batch_classification_stats(logits, targets)

    assert stats["label_accuracy"] == 1.0
    assert stats["exact_match"] == 1.0
    assert stats["macro_f1"] == 1.0
