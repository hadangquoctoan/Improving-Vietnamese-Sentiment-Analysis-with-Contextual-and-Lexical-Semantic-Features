import unittest
from pathlib import Path

from lexicon_build.expansion.canonical import build_canonicalizer
from tests.helpers.fake_segmenter import FakeSegmenter, FIXTURE_MAPPING

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_expansion"


class CanonicalizerTests(unittest.TestCase):
    def test_normalizes_and_rejects_multitoken(self):
        canon = build_canonicalizer(FakeSegmenter(FIXTURE_MAPPING))
        self.assertEqual(canon("  Tốt  "), "tốt")
        self.assertEqual(canon("hào hứng"), "hào_hứng")
        self.assertIsNone(canon("sự quan tâm quá mức"))
        self.assertIsNone(canon("   "))

    def test_cache_reuses_and_persists(self):
        cache_path = FIXTURES / "_cache_test.json"
        if cache_path.exists():
            cache_path.unlink()
        segmenter = FakeSegmenter(FIXTURE_MAPPING)
        canon = build_canonicalizer(segmenter, cache_path)
        self.assertEqual(canon("tốt"), "tốt")
        self.assertEqual(canon("tốt"), "tốt")
        self.assertEqual(segmenter.calls, 1)
        canon.flush()

        second = FakeSegmenter(FIXTURE_MAPPING)
        cached = build_canonicalizer(second, cache_path)
        self.assertEqual(cached("tốt"), "tốt")
        self.assertEqual(second.calls, 0)
        cache_path.unlink()


if __name__ == "__main__":
    unittest.main()
