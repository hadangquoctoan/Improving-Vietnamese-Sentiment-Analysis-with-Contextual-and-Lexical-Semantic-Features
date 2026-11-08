import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from vnsentiment.inference import SentimentPredictor
from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.projection import LexiconProjection
from vnsentiment.models.sentiment import SentimentModel
from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from tests.helpers.fake_encoder import FakeEncoder
from tests.helpers.fake_tokenizer import FakeTokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_fixture.json"
PIECES = {
    "tốt": [10, 11],
    "này": [12],
    "không": [13, 14],
    "tệ": [15],
}


def _build_model(store):
    return SentimentModel(
        phobert=PhoBertBranch(
            FakeTokenizer(PIECES),
            PhoBertEncoder(model=FakeEncoder(hidden_size=4), device="cpu"),
        ),
        lexicon=LexiconBranch(store, LexiconProjection(hidden_dim=4)),
        gate=LearnableGate(hidden_dim=4, bias_init=0.0),
        classifier=SentimentClassifier(hidden_dim=4),
    )


class SentimentPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LexiconStore(
            json.loads(FIXTURE.read_text(encoding="utf-8"))
        )

    def setUp(self):
        torch.manual_seed(0)
        self.model = _build_model(self.store)
        self.predictor = SentimentPredictor(self.model)

    def test_predict_returns_label_and_normalized_probabilities(self):
        results = self.predictor.predict(["tốt này không tệ", "tệ"])

        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn(result["label"], ("Neg", "Neu", "Pos"))
            probabilities = result["probabilities"]
            self.assertEqual(set(probabilities), {"Neg", "Neu", "Pos"})
            self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)
            best = max(probabilities, key=probabilities.get)
            self.assertEqual(result["label"], best)

    def test_predict_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            self.predictor.predict([])
        with self.assertRaises(ValueError):
            self.predictor.predict(["   "])

    def test_from_checkpoint_restores_predictions(self):
        expected = self.predictor.predict(["tốt này không tệ"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            torch.save({"model_state": self.model.state_dict()}, checkpoint)
            with mock.patch(
                "vnsentiment.inference.build_sentiment_model",
                side_effect=lambda device: _build_model(self.store),
            ):
                loaded = SentimentPredictor.from_checkpoint(
                    checkpoint, device="cpu"
                )

        actual = loaded.predict(["tốt này không tệ"])
        self.assertEqual(actual[0]["label"], expected[0]["label"])
        self.assertAlmostEqual(
            actual[0]["probabilities"]["Neg"],
            expected[0]["probabilities"]["Neg"],
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
