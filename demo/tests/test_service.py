"""Real-asset tests for the demo API. Skipped when checkpoints or the JDK are missing.

Run from the repo root (JAVA_HOME must point at a JDK):
    python -m pytest demo/tests -q
"""

import os
import unittest

import torch

from demo.api.service import CHECKPOINT_DIR, SPECS, DemoService, InputError

_READY = (CHECKPOINT_DIR / SPECS["uit"].checkpoint).is_file() and bool(os.environ.get("JAVA_HOME"))


@unittest.skipUnless(_READY, "needs artifacts/checkpoints/best_fused.pt and JAVA_HOME")
class DemoServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = DemoService()

    def test_step_by_step_forward_matches_model_forward(self):
        model = self.service._models["uit"]
        segmented = self.service.prepare("Giảng viên dạy rất nhiệt tình nhưng phòng học hơi nóng")
        with torch.no_grad():
            for lexicon, mode in ((True, "fused"), (False, "contextual")):
                ours = self.service.forward(model, segmented, lexicon)[0]
                reference = model([segmented], mode=mode)
                torch.testing.assert_close(ours, reference, rtol=0, atol=0)

    def test_response_contract(self):
        result = self.service.predict("Thầy cô hướng dẫn rất tận tâm", "uit", True)
        for key in ("label", "label_key", "label_id", "probs", "segmented", "truncated",
                    "tokens", "coverage", "model", "latency_ms"):
            self.assertIn(key, result)
        self.assertAlmostEqual(sum(result["probs"].values()), 1.0, places=4)
        self.assertEqual(result["label_key"], max(result["probs"], key=result["probs"].get))
        self.assertEqual(result["coverage"]["words"], len(result["segmented"].split()))
        self.assertFalse(result["truncated"])

    def test_lexicon_words_have_gate_and_original_polarity_scale(self):
        result = self.service.predict("Thầy cô hướng dẫn rất tận tâm")
        hits = [token for token in result["tokens"] if token["in_lexicon"]]
        self.assertTrue(hits, "expected at least one lexicon word")
        for token in hits:
            self.assertGreater(token["gate"], 0.0)
            self.assertLess(token["gate"], 1.0)
            polarity = token["dims"]["polarity"]
            if polarity is not None:
                self.assertGreaterEqual(polarity, -1.0)
                self.assertLessEqual(polarity, 1.0)

    def test_out_of_lexicon_word_has_no_gate(self):
        result = self.service.predict("xyzqwv tốt")
        oov = result["tokens"][0]
        self.assertFalse(oov["in_lexicon"])
        self.assertIsNone(oov["gate"])
        self.assertIsNone(oov["dims"])

    def test_lexicon_off_has_no_gates(self):
        result = self.service.predict("Thầy cô hướng dẫn rất tận tâm", lexicon=False)
        self.assertTrue(all(token["gate"] is None for token in result["tokens"]))

    def test_long_text_is_flagged_truncated(self):
        text = " ".join(["xyzqwv"] * 140)  # 979 chars, under MAX_CHARS, but several subwords per word
        self.assertLessEqual(len(text), 1000)
        result = self.service.predict(text)
        self.assertTrue(result["truncated"])
        self.assertLess(sum(token["subwords"] > 0 for token in result["tokens"]), 140)

    def test_invalid_inputs_raise_input_error(self):
        for text in ("", "   ", "a" * 1001):
            with self.assertRaises(InputError):
                self.service.predict(text)
        with self.assertRaises(InputError):
            self.service.predict("tốt", model_id="khong-co-model-nay")

    def test_binary_aivivn_model_returns_two_labels(self):
        result = self.service.predict("Sản phẩm dùng rất tốt, giao hàng nhanh", "aivivn")
        self.assertEqual(set(result["probs"]), {"neg", "pos"})
        self.assertIn(result["label_key"], ("neg", "pos"))
        self.assertAlmostEqual(sum(result["probs"].values()), 1.0, places=4)
        self.assertEqual(result["model"]["checkpoint"], "aivivn2019_best.pt")

    def test_model_list_reports_labels_and_recommendation(self):
        listed = {m["id"]: m for m in self.service.models()}
        self.assertEqual(set(listed), set(SPECS))
        self.assertEqual(listed["aivivn"]["labels"], ["neg", "pos"])
        self.assertEqual(listed["uit"]["labels"], ["neg", "neu", "pos"])
        self.assertEqual({k for k, m in listed.items() if m["recommended"]}, {"uit", "vlsp", "aivivn"})

    def test_at_most_three_models_stay_loaded(self):
        for model_id in ("vlsp", "aivivn", "uit-alt", "uit"):
            if (CHECKPOINT_DIR / SPECS[model_id].checkpoint).is_file():
                self.service.predict("Thầy cô hướng dẫn rất tận tâm", model_id)
                self.assertLessEqual(len(self.service.loaded()), 3)
        self.assertEqual(self.service.loaded()[-1], "uit")  # most recently used is kept


@unittest.skipUnless(_READY, "needs artifacts/checkpoints/best_fused.pt and JAVA_HOME")
class ApiTests(unittest.TestCase):
    def test_http_endpoints(self):
        try:
            from fastapi.testclient import TestClient
        except Exception as error:  # httpx missing
            self.skipTest(f"TestClient unavailable: {error}")
        from demo.api.app import app

        with TestClient(app) as client:
            self.assertEqual(client.get("/api/health").json()["status"], "ok")
            self.assertEqual({m["id"] for m in client.get("/api/models").json()}, set(SPECS))
            binary = client.post("/api/predict", json={"text": "Hàng kém chất lượng", "model": "aivivn"})
            self.assertEqual(binary.status_code, 200)
            self.assertEqual(set(binary.json()["probs"]), {"neg", "pos"})
            ok = client.post("/api/predict", json={"text": "Sản phẩm không tệ lắm"})
            self.assertEqual(ok.status_code, 200)
            self.assertIn(ok.json()["label_key"], ("neg", "neu", "pos"))
            bad = client.post("/api/predict", json={"text": "   "})
            self.assertEqual(bad.status_code, 400)
            self.assertIn("error", bad.json())
            broken = client.post(
                "/api/predict", content=b'{"text": ', headers={"Content-Type": "application/json"}
            )
            self.assertIn(broken.status_code, (400, 422))
            self.assertIn("error", broken.json())
            missing = client.post("/api/predict", json={"model": "uit"})
            self.assertEqual(missing.status_code, 422)
            self.assertIn("error", missing.json())
            cors = client.options(
                "/api/predict",
                headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
            )
            self.assertEqual(cors.headers.get("access-control-allow-origin"), "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
