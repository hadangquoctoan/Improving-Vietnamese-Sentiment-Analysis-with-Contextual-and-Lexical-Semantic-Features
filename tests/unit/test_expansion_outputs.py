import json
import unittest
from pathlib import Path

from lexicon_build.expansion.coverage import subset_coverage
from lexicon_build.expansion.outputs import ensure_output_absent, write_outputs

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lexicon_expansion"


def _entries():
    payload = json.loads((FIXTURES / "base_composite.json").read_text(encoding="utf-8"))
    return payload["entries"]


def _payload():
    return {
        "source": {"name": "fixture", "version": "0.3.0", "parent_version": "0.2.0"},
        "features": [
            "polarity", "valence", "arousal", "dominance",
            "joy", "sadness", "anger", "fear",
            "disgust", "surprise", "trust", "anticipation",
        ],
        "entries": _entries(),
    }


class OutputTests(unittest.TestCase):
    def test_ensure_output_absent_raises(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "composite_lexicon.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                ensure_output_absent(output)

    def test_write_outputs_is_byte_deterministic(self):
        import tempfile

        expansion = {
            "key": "hay",
            "dim": 0,
            "value": 0.875,
            "source": "wordnet_syn",
            "confidence": 1.0,
            "seeds": ("tốt",),
            "seed_values": (0.875,),
            "detail": "test",
            "decision": "accepted",
            "reason": "accepted",
        }
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for directory in (first, second):
                write_outputs(
                    Path(directory),
                    _payload(),
                    expansions=[expansion],
                    conflicts=[],
                    coverage={"train": {}},
                    dropped=[],
                    readme_lines=["# test"],
                )
            for name in (
                "composite_lexicon.json",
                "coverage_expansion.json",
                "expansions.tsv",
                "conflicts_expansion.tsv",
                "dropped_multitoken.tsv",
                "README.source.md",
            ):
                self.assertEqual(
                    (Path(first) / name).read_bytes(),
                    (Path(second) / name).read_bytes(),
                    name,
                )

    def test_expansions_tsv_joins_seed_values(self):
        import tempfile

        expansion = {
            "key": "hay",
            "dim": 0,
            "value": 0.875,
            "source": "wordnet_syn",
            "confidence": 1.0,
            "seeds": ("tốt", "vui"),
            "seed_values": (0.875, 0.5),
            "detail": "test",
        }
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(
                Path(tmp),
                _payload(),
                expansions=[expansion],
                conflicts=[],
                coverage={},
                dropped=[],
                readme_lines=["# test"],
            )
            text = (Path(tmp) / "expansions.tsv").read_text(encoding="utf-8")
            self.assertIn("key\tdim\tvalue\tsource", text)
            self.assertIn("hay\t0\t0.875\twordnet_syn\t1.000000\ttốt,vui\t0.875000,0.500000", text)


class SubsetCoverageTests(unittest.TestCase):
    def test_subset_coverage_uses_subset_denominator(self):
        entries = {
            "tốt": {"values": [0.0] * 12, "mask": [1] + [0] * 11},
        }
        result = subset_coverage(
            {"tốt": 2, "hay": 1, "xyz": 3},
            entries,
            {"tốt", "hay"},
        )
        self.assertEqual(result["tokens_total"], 3)
        self.assertEqual(result["tokens_covered"], 2)
        self.assertEqual(result["token_coverage_pct"], 66.67)
        self.assertEqual(result["types_total"], 2)
        self.assertEqual(result["type_coverage_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
