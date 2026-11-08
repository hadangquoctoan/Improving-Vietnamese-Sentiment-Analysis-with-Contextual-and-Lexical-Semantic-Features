import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from vnsentiment.lexicon.store import LexiconStore
from vnsentiment.models.fusion.classifier import SentimentClassifier
from vnsentiment.models.fusion.gate import LearnableGate
from vnsentiment.models.fusion.lexicon_branch import LexiconBranch
from vnsentiment.models.fusion.projection import LexiconProjection
from vnsentiment.models.sentiment import SentimentModel
from vnsentiment.phobert_branch.branch import PhoBertBranch
from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from vnsentiment.training import evaluate
from vnsentiment.training.evaluate import evaluate_test
from tests.helpers.fake_encoder import FakeEncoder
from tests.helpers.fake_tokenizer import FakeTokenizer

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_fixture.json"
PIECES = {
    "tốt": [10, 11],
    "này": [12],
    "không": [13, 14],
    "tệ": [15],
}
SAMPLES = [
    {"text": "tốt này không tệ", "label": 2},
    {"text": "tệ", "label": 0},
    {"text": "sản_phẩm này tốt", "label": 2},
    {"text": "không tốt", "label": 0},
    {"text": "tốt ,", "label": 2},
]


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


class EvaluateTestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LexiconStore(
            json.loads(FIXTURE.read_text(encoding="utf-8"))
        )

    def setUp(self):
        torch.manual_seed(0)
        self.model = _build_model(self.store)

    def test_existing_report_blocks_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                evaluate_test("missing.pt", report)

    def test_report_contains_metrics_provenance_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            report_path = Path(directory) / "report.json"
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps({"splits": {"test": {"output_sha256": "abc"}}}),
                encoding="utf-8",
            )
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "epoch": 3,
                    "dev_metrics": {
                        "accuracy": 0.5,
                        "macro_f1": 0.4,
                        "f1_per_class": (0.1, 0.2, 0.9),
                    },
                    "seed": 7,
                    "config": {"batch_size": 8},
                },
                checkpoint,
            )

            def fake_load_split(split, allow_test=False):
                if split == "test":
                    self.assertTrue(allow_test)
                    return SAMPLES
                return SAMPLES[:2]

            with mock.patch.object(
                evaluate,
                "build_sentiment_model",
                side_effect=lambda device, lexicon_path=None: _build_model(self.store),
            ), mock.patch.object(
                evaluate, "load_split", side_effect=fake_load_split
            ), mock.patch.object(
                evaluate, "_DEFAULT_LEXICON_PATH", FIXTURE
            ), mock.patch.object(
                evaluate, "_MANIFEST_PATH", manifest
            ):
                report = evaluate_test(
                    checkpoint, report_path, device="cpu", batch_size=2
                )

            self.assertTrue(report_path.is_file())
            self.assertEqual(report["checkpoint"]["epoch"], 3)
            self.assertEqual(report["checkpoint"]["seed"], 7)
            self.assertEqual(
                report["checkpoint"]["sha256"],
                hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            self.assertEqual(report["data"]["rows"], len(SAMPLES))
            self.assertEqual(
                report["data"]["manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertIn("accuracy", report["metrics"])
            self.assertIn("macro_f1", report["metrics"])
            self.assertEqual(len(report["metrics"]["f1_per_class"]), 3)
            self.assertIsInstance(report["metrics"]["f1_per_class"], list)
            self.assertEqual(report["coverage"]["test"], 1.0)
            self.assertIsNone(report["runtime"]["gpu"])
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["run_id"],
                report["run_id"],
            )


if __name__ == "__main__":
    unittest.main()
