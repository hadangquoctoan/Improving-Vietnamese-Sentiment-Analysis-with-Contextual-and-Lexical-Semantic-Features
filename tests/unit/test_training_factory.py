import unittest
from pathlib import Path
from unittest import mock

import torch

from vnsentiment.phobert_branch.encoder import PhoBertEncoder
from vnsentiment.training import factory
from tests.helpers.fake_encoder import FakeEncoder
from tests.helpers.fake_tokenizer import FakeTokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_expansion"


class FactoryDeviceTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_all_parameters_follow_requested_device(self):
        def encoder_factory(device):
            return PhoBertEncoder(
                model=FakeEncoder(hidden_size=768), device=device
            )

        with mock.patch.object(
            factory,
            "load_phobert_tokenizer",
            return_value=FakeTokenizer({}),
        ), mock.patch.object(
            factory, "PhoBertEncoder", side_effect=encoder_factory
        ):
            model = factory.build_sentiment_model("cuda")

        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.device.type, "cuda", name)

    def test_lexicon_path_and_binary_head(self):
        def encoder_factory(device):
            return PhoBertEncoder(
                model=FakeEncoder(hidden_size=768), device=device
            )

        with mock.patch.object(
            factory,
            "load_phobert_tokenizer",
            return_value=FakeTokenizer({}),
        ), mock.patch.object(
            factory, "PhoBertEncoder", side_effect=encoder_factory
        ):
            model = factory.build_sentiment_model(
                "cpu",
                lexicon_path=FIXTURES / "base_composite.json",
                num_labels=2,
            )

        self.assertEqual(model.classifier.fc.out_features, 2)
        self.assertEqual(len(model.lexicon.store), 7)


if __name__ == "__main__":
    unittest.main()
