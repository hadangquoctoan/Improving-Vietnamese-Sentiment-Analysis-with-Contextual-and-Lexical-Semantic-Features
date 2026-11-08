import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vnsentiment.training import vlsp_data


def _write_split(directory, split, rows):
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path = Path(directory) / f"{split}.jsonl"
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(directory, split, sha256):
    manifest = {"splits": {split: {"output_sha256": sha256}}}
    (Path(directory) / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class LoadVlsp2016Tests(unittest.TestCase):
    def test_real_train_has_expected_rows(self):
        self.assertEqual(len(vlsp_data.load_vlsp2016("train")), 5100)

    def test_test_split_is_locked_by_default(self):
        with self.assertRaises(ValueError):
            vlsp_data.load_vlsp2016("test")

    def test_unknown_split_raises(self):
        with self.assertRaises(ValueError):
            vlsp_data.load_vlsp2016("dev")

    def test_loads_verified_split(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"text": "tốt", "label": 2},
                {"text": "tệ", "label": 0},
            ]
            sha = _write_split(directory, "train", rows)
            _write_manifest(directory, "train", sha)
            with mock.patch.object(vlsp_data, "_DATA_DIR", Path(directory)):
                loaded = vlsp_data.load_vlsp2016("train")

        self.assertEqual(loaded, rows)

    def test_sha_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_split(directory, "train", [{"text": "tốt", "label": 2}])
            _write_manifest(directory, "train", "0" * 64)
            with mock.patch.object(vlsp_data, "_DATA_DIR", Path(directory)):
                with self.assertRaises(ValueError):
                    vlsp_data.load_vlsp2016("train")

    def test_label_outside_range_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            sha = _write_split(
                directory, "train", [{"text": "tốt", "label": 3}]
            )
            _write_manifest(directory, "train", sha)
            with mock.patch.object(vlsp_data, "_DATA_DIR", Path(directory)):
                with self.assertRaises(ValueError):
                    vlsp_data.load_vlsp2016("train")


if __name__ == "__main__":
    unittest.main()
