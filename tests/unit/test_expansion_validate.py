import json
import unittest
from pathlib import Path

from lexicon_build.expansion.validate import (
    assert_no_empty_entries,
    assert_non_destructive,
    assert_provenance_complete,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_expansion"


def _entries():
    payload = json.loads((FIXTURES / "base_composite.json").read_text(encoding="utf-8"))
    return payload["entries"]


class ValidateTests(unittest.TestCase):
    def test_non_destructive_detects_changed_active_value(self):
        base = _entries()
        merged = {key: {"values": list(item["values"]), "mask": list(item["mask"])} for key, item in base.items()}
        merged["tốt"]["values"][0] = 0.1
        with self.assertRaises(AssertionError):
            assert_non_destructive(base, merged)

    def test_provenance_complete_detects_missing_row(self):
        base = _entries()
        merged = {key: {"values": list(item["values"]), "mask": list(item["mask"])} for key, item in base.items()}
        merged["hay"] = {"values": [0.875] + [0.0] * 11, "mask": [1] + [0] * 11}
        with self.assertRaises(AssertionError):
            assert_provenance_complete(base, merged, [])

    def test_provenance_complete_accepts_matching_row(self):
        base = _entries()
        merged = {key: {"values": list(item["values"]), "mask": list(item["mask"])} for key, item in base.items()}
        merged["hay"] = {"values": [0.875] + [0.0] * 11, "mask": [1] + [0] * 11}
        assert_provenance_complete(
            base,
            merged,
            [{"key": "hay", "dim": 0, "decision": "accepted"}],
        )

    def test_no_empty_entries(self):
        with self.assertRaises(AssertionError):
            assert_no_empty_entries({"x": {"values": [0.0] * 12, "mask": [0] * 12}})


if __name__ == "__main__":
    unittest.main()
