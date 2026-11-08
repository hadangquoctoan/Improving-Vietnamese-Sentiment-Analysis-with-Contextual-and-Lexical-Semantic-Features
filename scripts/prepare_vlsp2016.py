#!/usr/bin/env python3
"""Normalize and segment user-provided VLSP2016 CSVs into processed JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnsentiment.preprocessing.collapse import normalize_basic  # noqa: E402
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter  # noqa: E402

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
RAW_DIR = ROOT / "data" / "raw" / "vlsp2016"
OUT_DIR = ROOT / "data" / "processed" / "vlsp2016"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Class", "Data"]:
            raise ValueError(f"{path.name}: expected header Class,Data")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            label = (row.get("Class") or "").strip().lower()
            text = (row.get("Data") or "").strip()
            if label not in LABEL_MAP:
                raise ValueError(f"{path.name}:{line_number}: bad label {label!r}")
            if not text:
                raise ValueError(f"{path.name}:{line_number}: empty text")
            rows.append((label, text))
    return rows


def _segment(rows, segmenter, split: str) -> list[dict]:
    processed = []
    for index, (label, text) in enumerate(rows, start=1):
        normalized = normalize_basic(text)
        segmented = normalize_basic(segmenter.segment(normalized))
        processed.append({"text": segmented, "label": LABEL_MAP[label]})
        if index % 500 == 0:
            print(f"{split}: {index}/{len(rows)}", flush=True)
    return processed


def _stats(rows: list[dict]) -> dict:
    lengths = [len(row["text"].split()) for row in rows]
    return {
        "rows": len(rows),
        "label_counts": {
            str(label): count
            for label, count in sorted(Counter(row["label"] for row in rows).items())
        },
        "tokens_min": min(lengths),
        "tokens_mean": round(sum(lengths) / len(lengths), 3),
        "tokens_max": max(lengths),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _duplicate_count(rows: list[dict]) -> int:
    return len(rows) - len({row["text"] for row in rows})


def _uit_raw_texts() -> set[str]:
    texts = set()
    for split in ("train", "dev", "test"):
        path = ROOT / "data" / "raw" / "uit-vsfc" / f"{split}_sents.txt"
        if path.is_file():
            texts.update(
                normalize_basic(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return texts


def _parse_args():
    parser = argparse.ArgumentParser(description="Prepare VLSP2016 JSONL.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    return parser.parse_args()


def _write_outputs(processed, out_dir):
    outputs = {}
    for split, rows in processed.items():
        path = out_dir / f"{split}.jsonl"
        _write_jsonl(path, rows)
        stats = _stats(rows)
        outputs[split] = {
            **{key: value for key, value in stats.items() if key != "rows"},
            "rows": len(rows),
            "output": str(path.relative_to(ROOT)).replace("\\", "/"),
            "output_sha256": _sha256(path),
            "output_bytes": path.stat().st_size,
        }
    return outputs


def _build_manifest(raw, processed, raw_dir, outputs):
    train_texts = {row["text"] for row in processed["train"]}
    test_texts = {row["text"] for row in processed["test"]}
    raw_test_texts = {normalize_basic(text) for _, text in raw["test"]}
    uit_texts = _uit_raw_texts()
    return {
        "dataset": "VLSP2016 (user-provided balanced 3-class CSV)",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "raw_dir": str(raw_dir.relative_to(ROOT)).replace("\\", "/"),
            "files": {
                f"{split}.csv": {
                    "sha256": _sha256(raw_dir / f"{split}.csv"),
                    "rows": len(raw[split]),
                }
                for split in ("train", "test")
            },
            "note": "User-provided; URL and license not supplied; do not redistribute.",
        },
        "normalization": ["Unicode NFC", "lowercase", "whitespace collapse"],
        "segmentation": {
            "tool": "VnCoreNLP 1.2 (py_vncorenlp)",
            "annotator": "wseg",
        },
        "label_map": LABEL_MAP,
        "splits": {
            split: {
                "rows": outputs[split]["rows"],
                "label_counts": outputs[split]["label_counts"],
                "tokens_min": outputs[split]["tokens_min"],
                "tokens_mean": outputs[split]["tokens_mean"],
                "tokens_max": outputs[split]["tokens_max"],
                "output": outputs[split]["output"],
                "output_sha256": outputs[split]["output_sha256"],
                "output_bytes": outputs[split]["output_bytes"],
            }
            for split in ("train", "test")
        },
        "integrity": {
            "train_duplicate_texts": _duplicate_count(processed["train"]),
            "test_duplicate_texts": _duplicate_count(processed["test"]),
            "train_test_exact_overlap": len(train_texts & test_texts),
            "test_overlap_uit_raw": len(raw_test_texts & uit_texts),
        },
        "notes": [
            "No deduplication; source order preserved.",
            "Test split is for one-shot cross-domain evaluation only.",
            "Model max length 256; longer texts are truncated at word boundaries.",
        ],
    }


def main() -> int:
    args = _parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw = {
        split: _read_rows(raw_dir / f"{split}.csv")
        for split in ("train", "test")
    }
    segmenter = VnCoreNLPSegmenter()
    processed = {split: _segment(raw[split], segmenter, split) for split in raw}
    outputs = _write_outputs(processed, out_dir)
    manifest = _build_manifest(raw, processed, raw_dir, outputs)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["integrity"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
