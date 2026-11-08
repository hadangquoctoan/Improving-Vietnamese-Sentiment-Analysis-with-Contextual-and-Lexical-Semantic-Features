import unittest

from vnsentiment.training.vlsp_split import stratified_split


def _samples():
    return [
        {"text": f"{label}-{index}", "label": label}
        for label in range(3)
        for index in range(10)
    ]


class StratifiedSplitTests(unittest.TestCase):
    def test_split_is_balanced_and_disjoint(self):
        fit, dev = stratified_split(_samples(), dev_ratio=0.2, seed=42)

        self.assertEqual(len(fit), 24)
        self.assertEqual(len(dev), 6)
        for label in range(3):
            self.assertEqual(sum(row["label"] == label for row in dev), 2)
            self.assertEqual(sum(row["label"] == label for row in fit), 8)
        self.assertFalse(
            {row["text"] for row in fit} & {row["text"] for row in dev}
        )

    def test_split_is_deterministic(self):
        first = stratified_split(_samples(), dev_ratio=0.2, seed=7)
        second = stratified_split(_samples(), dev_ratio=0.2, seed=7)
        self.assertEqual(first, second)

    def test_empty_samples_raise(self):
        with self.assertRaises(ValueError):
            stratified_split([], dev_ratio=0.1, seed=42)

    def test_invalid_ratio_raises(self):
        for ratio in (0.0, 1.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                stratified_split(_samples(), dev_ratio=ratio, seed=42)


if __name__ == "__main__":
    unittest.main()
