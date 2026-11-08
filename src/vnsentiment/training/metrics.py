"""Accuracy and macro-F1 computed with plain torch."""

import torch


def compute_metrics(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int = 3
) -> dict:
    if preds.numel() == 0 or labels.numel() == 0:
        raise ValueError("preds and labels must not be empty")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    f1_per_class = []
    for label in range(num_classes):
        predicted = preds == label
        actual = labels == label
        true_positive = (predicted & actual).sum().item()
        false_positive = (predicted & ~actual).sum().item()
        false_negative = (~predicted & actual).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_per_class.append(
            2 * true_positive / denominator if denominator else 0.0
        )

    return {
        "accuracy": (preds == labels).sum().item() / labels.numel(),
        "macro_f1": sum(f1_per_class) / num_classes,
        "f1_per_class": tuple(f1_per_class),
    }
