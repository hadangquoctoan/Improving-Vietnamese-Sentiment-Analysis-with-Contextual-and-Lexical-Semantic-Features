import json
import unittest
from pathlib import Path

from lexicon_build.expansion.canonical import build_canonicalizer
from lexicon_build.expansion.seeds import (
    load_audited_conflicts,
    load_seeds,
    load_stoplist,
)
from lexicon_build.expansion.wordnet import SynonymGroup, load_groups
from tests.helpers.fake_segmenter import FakeSegmenter, FIXTURE_MAPPING

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_expansion"
WORDNET = FIXTURES / "wordnet"


def _base_entries():
    payload = json.loads((FIXTURES / "base_composite.json").read_text(encoding="utf-8"))
    return payload["entries"]


class SeedTests(unittest.TestCase):
    def setUp(self):
        self.canon = build_canonicalizer(FakeSegmenter(FIXTURE_MAPPING))

    def test_seed_selection_uses_strength_filter(self):
        seeds = load_seeds(_base_entries(), min_abs=0.5)
        self.assertEqual(
            seeds,
            {"tốt": 0.875, "xấu": -0.625, "vui": 0.5, "buồn": -0.5},
        )

    def test_seed_with_vad_conflict_is_excluded(self):
        entries = {
            "tốt": {
                "values": [0.875, 0.9] + [0.0] * 10,
                "mask": [1, 1] + [0] * 10,
            },
            "nhàn_nhã": {
                "values": [-0.625, 0.806] + [0.0] * 10,
                "mask": [1, 1] + [0] * 10,
            },
        }
        seeds = load_seeds(entries)
        self.assertIn("tốt", seeds)
        self.assertNotIn("nhàn_nhã", seeds)

    def test_audited_is_separate_from_stoplist(self):
        audited = load_audited_conflicts(FIXTURES / "conflicts_vswn.tsv")
        self.assertEqual(audited, {"tốt", "cũ"})
        stoplist = load_stoplist(WORDNET, self.canon)
        self.assertNotIn("tốt", stoplist)
        for word in ("rất", "và", "của", "này"):
            self.assertIn(word, stoplist)


class SynonymPropagationTests(unittest.TestCase):
    def setUp(self):
        self.entries = _base_entries()
        self.seeds = load_seeds(self.entries, min_abs=0.5)
        self.audited = load_audited_conflicts(FIXTURES / "conflicts_vswn.tsv")
        self.canon = build_canonicalizer(FakeSegmenter(FIXTURE_MAPPING))
        self.stoplist = load_stoplist(WORDNET, self.canon)

    def _propagate(self, groups, *, fill_conflicts=False):
        from lexicon_build.expansion.synonym import propagate_synonyms

        return propagate_synonyms(
            groups,
            self.seeds,
            self.entries,
            self.audited,
            self.stoplist,
            fill_conflicts=fill_conflicts,
        )

    def test_propagates_from_single_seed(self):
        expanded, _ = self._propagate([SynonymGroup("adj", ("tốt", "hay"))])
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].key, "hay")
        self.assertEqual(expanded[0].value, 0.875)
        self.assertEqual(expanded[0].source, "wordnet_syn")

    def test_sign_conflict_skips_group(self):
        groups = [SynonymGroup("adj", ("vui", "hào_hứng"))]
        seeds = dict(self.seeds, hào_hứng=-0.5)
        from lexicon_build.expansion.synonym import propagate_synonyms

        expanded, logs = propagate_synonyms(
            groups, seeds, self.entries, self.audited, self.stoplist
        )
        self.assertEqual(expanded, [])
        self.assertIn("sign_conflict", {row["reason"] for row in logs})

    def test_audited_target_is_skipped_by_default(self):
        groups = [SynonymGroup("adj", ("tốt", "cũ"))]
        expanded, logs = self._propagate(groups)
        self.assertEqual(expanded, [])
        self.assertIn("audited_conflict", {row["reason"] for row in logs})

    def test_audited_target_can_be_filled_with_flag(self):
        groups = [SynonymGroup("adj", ("tốt", "cũ"))]
        expanded, _ = self._propagate(groups, fill_conflicts=True)
        self.assertEqual([(item.key, item.value) for item in expanded], [("cũ", 0.875)])

    def test_active_target_is_never_rewritten(self):
        groups = [SynonymGroup("adj", ("tốt", "hay"))]
        expanded, logs = self._propagate(groups)
        self.assertEqual([item.key for item in expanded], ["hay"])
        self.assertIn("already_active", {row["reason"] for row in logs})
        self.assertNotIn("tốt", {item.key for item in expanded})

    def test_stoplist_target_is_skipped(self):
        groups = [SynonymGroup("adj", ("vui", "rất"))]
        expanded, logs = self._propagate(groups)
        self.assertEqual(expanded, [])
        self.assertIn("stoplist", {row["reason"] for row in logs})

    def test_multi_group_aggregation_uses_median(self):
        groups = [
            SynonymGroup("adj", ("vui", "hay")),
            SynonymGroup("adj", ("tốt", "hay")),
        ]
        seeds = dict(self.seeds)
        expanded, logs = self._propagate(groups)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].key, "hay")
        self.assertEqual(expanded[0].value, 0.6875)
        self.assertIn("spread=0.375", expanded[0].detail)

    def test_multi_group_disagreement_is_rejected(self):
        groups = [
            SynonymGroup("adj", ("vui", "hay")),
            SynonymGroup("adj", ("buồn", "hay")),
        ]
        expanded, logs = self._propagate(groups)
        self.assertEqual(expanded, [])
        self.assertIn("multi_group_disagreement", {row["reason"] for row in logs})

    def test_vad_conflict_is_rejected(self):
        groups = [SynonymGroup("adj", ("xấu", "rực_rỡ"))]
        expanded, logs = self._propagate(groups)
        self.assertEqual(expanded, [])
        self.assertIn("vad_conflict", {row["reason"] for row in logs})


if __name__ == "__main__":
    unittest.main()
