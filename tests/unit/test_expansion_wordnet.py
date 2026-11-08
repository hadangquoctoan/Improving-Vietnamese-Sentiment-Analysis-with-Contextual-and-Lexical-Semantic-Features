import unittest
from pathlib import Path

from lexicon_build.expansion.canonical import build_canonicalizer
from lexicon_build.expansion.wordnet import build_key_file_map, load_groups
from tests.helpers.fake_segmenter import FakeSegmenter, FIXTURE_MAPPING

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_expansion"
WORDNET = FIXTURES / "wordnet"


class WordNetLoaderTests(unittest.TestCase):
    def setUp(self):
        self.canon = build_canonicalizer(FakeSegmenter(FIXTURE_MAPPING))

    def test_load_groups_dedupes_and_logs_dropped(self):
        groups, dropped = load_groups(WORDNET, ("adj",), self.canon)
        self.assertEqual(
            [(group.source_file, group.terms) for group in groups],
            [
                ("adj", ("tốt", "hay")),
                ("adj", ("xấu", "dở")),
                ("adj", ("vui", "hào_hứng")),
                ("adj", ("khô_cằn", "khô_hạn")),
            ],
        )
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["raw_term"], "sự quan tâm quá mức")
        self.assertEqual(dropped[0]["source"], "adj")

    def test_build_key_file_map_includes_adv(self):
        mapping = build_key_file_map(
            WORDNET, ("adj", "adv", "verb.emotion"), self.canon
        )
        self.assertEqual(mapping["hay"], ("adj", "adv"))
        self.assertEqual(mapping["rất"], ("adv",))
        self.assertEqual(mapping["yêu"], ("verb.emotion",))
        self.assertNotIn("sự quan tâm quá mức", mapping)


if __name__ == "__main__":
    unittest.main()
