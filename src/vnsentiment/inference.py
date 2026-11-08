"""Local inference for the trained sentiment checkpoint."""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from vnsentiment.training.factory import build_sentiment_model

_LABELS = ("Neg", "Neu", "Pos")


class SentimentPredictor:
    def __init__(
        self, model: nn.Module, labels: Sequence[str] = _LABELS
    ):
        self.model = model.eval()
        self.labels = tuple(labels)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        labels: Sequence[str] = _LABELS,
    ) -> "SentimentPredictor":
        payload = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=True
        )
        model = build_sentiment_model(device, num_labels=len(labels))
        model.load_state_dict(payload["model_state"])
        return cls(model, labels)

    def predict(
        self, texts: Sequence[str], batch_size: int = 8
    ) -> list[dict]:
        rows = list(texts)
        if not rows:
            raise ValueError("texts must not be empty")
        if any(
            not isinstance(text, str) or not text.strip() for text in rows
        ):
            raise ValueError("texts must be non-empty segmented strings")

        results = []
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                probabilities = self.model(chunk).softmax(dim=-1).cpu()
                for row in probabilities:
                    label_id = int(row.argmax().item())
                    results.append(
                        {
                            "label_id": label_id,
                            "label": self.labels[label_id],
                            "probabilities": {
                                label: float(row[index])
                                for index, label in enumerate(self.labels)
                            },
                        }
                    )
        return results
