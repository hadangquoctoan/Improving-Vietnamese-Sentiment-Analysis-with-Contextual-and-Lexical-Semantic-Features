import json
import unittest
from pathlib import Path

from lexicon_build.expansion.merge import merge_entries
from lexicon_build.expansion.synonym import ExpandedValue

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_expansion"


def _entries():
    payload = json.loads((FIXTURES / "base_composite.json").read_text(encoding="utf-8"))
    return payload["entries"]


def _value(key, value, source="wordnet_syn"):
    return ExpandedValue(
        key=key,
        dim=0,
        value=value,
        source=source,
        confidence=1.0,
        seeds=("tốt",),
        seed_values=(0.875,),
        detail="test",
    )


class MergeTests(unittest.TestCase):
    def test_new_key_gets_polarity_only(self):
        merged, logs = merge_entries(_entries(), [_value("hay", 0.875)], set())
        self.assertEqual(merged["hay"]["values"][0], 0.875)
        self.assertEqual(merged["hay"]["mask"][0], 1)
        self.assertEqual(merged["hay"]["mask"][1:], [0] * 11)
        self.assertEqual(logs[0]["decision"], "accepted")

    def test_existing_masked_dim_is_filled_without_touching_others(self):
        merged, _ = merge_entries(_entries(), [_value("cũ", 0.875)], set())
        self.assertEqual(merged["cũ"]["values"][0], 0.875)
        self.assertEqual(merged["cũ"]["mask"][0], 1)
        self.assertEqual(merged["cũ"]["values"][1], 0.4)
        self.assertEqual(merged["cũ"]["mask"][1], 1)

    def test_active_value_is_never_rewritten(self):
        merged, logs = merge_entries(_entries(), [_value("tốt", 0.1)], set())
        self.assertEqual(merged["tốt"]["values"][0], 0.875)
        self.assertEqual(logs[0]["decision"], "rejected")
        self.assertEqual(logs[0]["reason"], "already_active")

    def test_audited_key_requires_opt_in(self):
        merged, logs = merge_entries(_entries(), [_value("cũ", 0.875)], {"cũ"})
        self.assertEqual(merged["cũ"]["mask"][0], 0)
        self.assertEqual(logs[0]["reason"], "audited_conflict")

        merged, logs = merge_entries(
            _entries(), [_value("cũ", 0.875)], {"cũ"}, fill_conflicts=True
        )
        self.assertEqual(merged["cũ"]["mask"][0], 1)
        self.assertEqual(logs[0]["reason"], "audited_override")

    def test_source_precedence_prefers_wordnet(self):
        expanded = [
            _value("hay", 0.9, source="fasttext_nn"),
            _value("hay", 0.5, source="wordnet_syn"),
        ]
        merged, logs = merge_entries(_entries(), expanded, set())
        self.assertEqual(merged["hay"]["values"][0], 0.5)
        self.assertIn("source_precedence", {row["reason"] for row in logs})


if __name__ == "__main__":
    unittest.main()
