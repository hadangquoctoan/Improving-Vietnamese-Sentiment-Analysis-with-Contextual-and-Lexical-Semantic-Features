import unittest
from pathlib import Path

import numpy as np

from lexicon_build.expansion.fasttext import VectorStore, propagate_fasttext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_expansion"
VEC = FIXTURES / "tiny.vec"


class VectorStoreTests(unittest.TestCase):
    def test_stream_keeps_only_needed_keys(self):
        store = VectorStore.stream(VEC, {"tốt", "sản", "phẩm"})
        self.assertEqual(len(store), 3)
        self.assertIsNotNone(store.vector("tốt"))
        self.assertIsNone(store.vector("hay"))

    def test_compound_vector_is_mean_and_normalized(self):
        store = VectorStore.stream(VEC, {"sản", "phẩm"})
        vector = store.vector("sản_phẩm")
        expected = np.array([0.5, 0.0, 0.5], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(vector, expected, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_missing_syllable_returns_none(self):
        store = VectorStore.stream(VEC, {"sản"})
        self.assertIsNone(store.vector("sản_x"))


class FastTextPropagationTests(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore.stream(
            VEC, {"tốt", "xấu", "hay", "dở", "sản", "phẩm"}
        )

    def test_positive_neighbor_propagates(self):
        expanded, logs = propagate_fasttext(
            self.store, ["hay"], {"tốt": 0.5}, min_neighbors=1
        )
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].key, "hay")
        self.assertAlmostEqual(expanded[0].value, 0.25, places=6)
        self.assertEqual(expanded[0].source, "fasttext_nn")

    def test_single_neighbor_is_rejected(self):
        expanded, logs = propagate_fasttext(self.store, ["hay"], {"tốt": 0.5})
        self.assertEqual(expanded, [])
        self.assertIn("low_neighbors", {row["reason"] for row in logs})

    def test_low_similarity_is_rejected(self):
        expanded, logs = propagate_fasttext(self.store, ["dở"], {"tốt": 0.5})
        self.assertEqual(expanded, [])
        self.assertIn("low_similarity", {row["reason"] for row in logs})

    def test_low_sign_agreement_is_rejected(self):
        expanded, logs = propagate_fasttext(
            self.store, ["hay"], {"tốt": 0.5, "xấu": -0.5}
        )
        self.assertEqual(expanded, [])
        self.assertIn("low_sign_agreement", {row["reason"] for row in logs})

    def test_low_score_after_shrink_is_rejected(self):
        expanded, logs = propagate_fasttext(
            self.store, ["hay"], {"tốt": 0.5}, min_neighbors=1, shrink=0.1
        )
        self.assertEqual(expanded, [])
        self.assertIn("low_score", {row["reason"] for row in logs})

    def test_missing_vector_is_logged(self):
        expanded, logs = propagate_fasttext(self.store, ["xyz"], {"tốt": 0.5})
        self.assertEqual(expanded, [])
        self.assertIn("no_vector", {row["reason"] for row in logs})


if __name__ == "__main__":
    unittest.main()
