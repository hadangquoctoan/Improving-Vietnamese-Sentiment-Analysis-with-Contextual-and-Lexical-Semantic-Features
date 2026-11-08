"""Build the full sentiment model from local artifacts."""

from pathlib import Path

import torch

from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.projection import LexiconProjection
from vnsentiment.models.sentiment import SentimentModel
from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from vnsentiment.phobert_branch.tokenizer import load_phobert_tokenizer


def build_sentiment_model(
    device: str | torch.device = "cpu",
    lexicon_path: str | Path | None = None,
    num_labels: int = 3,
) -> SentimentModel:
    model = SentimentModel(
        phobert=PhoBertBranch(
            load_phobert_tokenizer(),
            PhoBertEncoder(device=device),
        ),
        lexicon=LexiconBranch(LexiconStore.load(lexicon_path), LexiconProjection()),
        gate=LearnableGate(),
        classifier=SentimentClassifier(num_labels=num_labels),
    )
    return model.to(device)
