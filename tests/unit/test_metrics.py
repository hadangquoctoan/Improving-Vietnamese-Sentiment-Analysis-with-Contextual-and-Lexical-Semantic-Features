import unittest

import torch

from vnsentiment.training.metrics import compute_metrics


class ComputeMetricsTests(unittest.TestCase):
    def test_hand_computed_imbalanced_case(self):
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 2, 2, 2, 2])
        preds = torch.tensor([0, 0, 0, 0, 2, 1, 2, 2, 2, 0])

        metrics = compute_metrics(preds, labels)

        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["macro_f1"], 0.85)
        self.assertEqual(len(metrics["f1_per_class"]), 3)
        self.assertAlmostEqual(metrics["f1_per_class"][0], 0.8)
        self.assertAlmostEqual(metrics["f1_per_class"][1], 1.0)
        self.assertAlmostEqual(metrics["f1_per_class"][2], 0.75)

    def test_class_absent_from_labels_and_preds_scores_zero(self):
        labels = torch.tensor([0, 2])
        preds = torch.tensor([0, 2])

        metrics = compute_metrics(preds, labels)

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["f1_per_class"][1], 0.0)
        self.assertAlmostEqual(metrics["macro_f1"], 2.0 / 3.0)

    def test_binary_num_classes(self):
        labels = torch.tensor([0, 0, 1, 1])
        preds = torch.tensor([0, 1, 1, 1])

        metrics = compute_metrics(preds, labels, num_classes=2)

        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertEqual(len(metrics["f1_per_class"]), 2)
        self.assertAlmostEqual(metrics["f1_per_class"][0], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["f1_per_class"][1], 0.8)
        self.assertAlmostEqual(metrics["macro_f1"], (2.0 / 3.0 + 0.8) / 2)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            compute_metrics(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
