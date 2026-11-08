#!/usr/bin/env python3
"""Normalize and segment AIVIVN 2019 CSVs into processed JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnsentiment.preprocessing.collapse import normalize_basic  # noqa: E402
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter  # noqa: E402

LABEL_MAP = {"negative": 0, "positive": 1}
RAW_DIR = ROOT / "data" / "raw" / "aivivn2019"
OUT_DIR = ROOT / "data" / "processed" / "aivivn2019"
DEV_RATIO = 0.1
SEED = 42


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "comment", "label"]:
            raise ValueError(f"{path.name}: expected header id,comment,label")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            text = (row.get("comment") or "").strip()
            if label not in {"0", "1"}:
                raise ValueError(f"{path.name}:{line_number}: bad label {label!r}")
            if not text:
                raise ValueError(f"{path.name}:{line_number}: empty comment")
            rows.append({"id": (row.get("id") or "").strip(), "text": text, "label": int(label)})
    return rows


def _segment(rows: list[dict], segmenter, split: str) -> list[dict]:
    processed = []
    for index, row in enumerate(rows, start=1):
        normalized = normalize_basic(row["text"])
        segmented = normalize_basic(segmenter.segment(normalized))
        processed.append({"text": segmented, "label": row["label"]})
        if index % 500 == 0:
            print(f"{split}: {index}/{len(rows)}", flush=True)
    return processed


def _stratified_split(rows: list[dict], dev_ratio: float, seed: int):
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    rng = random.Random(seed)
    train, dev = [], []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        dev_count = max(1, round(len(items) * dev_ratio))
        dev.extend(items[:dev_count])
        train.extend(items[dev_count:])
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


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


def _processed_texts(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    texts = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        texts.add(json.loads(line)["text"].replace("_", " "))
    return texts


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
    parser = argparse.ArgumentParser(description="Prepare AIVIVN 2019 JSONL.")
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
    dev_texts = {row["text"] for row in processed["dev"]}
    test_texts = {row["text"] for row in processed["test"]}
    test_raw_texts = {normalize_basic(row["text"]) for row in raw["test"]}
    test_ids_all_train_prefix = all(
        row["id"].startswith("train_") for row in raw["test"]
    )
    return {
        "dataset": "AIVIVN 2019 (user-provided binary CSV; test likely split from train)",
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
        "segmentation": {"tool": "VnCoreNLP 1.2 (py_vncorenlp)", "annotator": "wseg"},
        "label_map": LABEL_MAP,
        "split_protocol": {
            "dev_ratio": DEV_RATIO,
            "seed": SEED,
            "stratified": True,
            "source_train_rows": len(processed["train"]) + len(processed["dev"]),
        },
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
            for split in ("train", "dev", "test")
        },
        "integrity": {
            "train_duplicate_texts": _duplicate_count(processed["train"]),
            "dev_duplicate_texts": _duplicate_count(processed["dev"]),
            "test_duplicate_texts": _duplicate_count(processed["test"]),
            "train_dev_exact_overlap": len(train_texts & dev_texts),
            "train_test_exact_overlap": len(train_texts & test_texts),
            "dev_test_exact_overlap": len(dev_texts & test_texts),
            "test_overlap_uit_raw": len(test_raw_texts & _uit_raw_texts()),
            "test_overlap_vlsp_test": len(
                {row["text"].replace("_", " ") for row in processed["test"]}
                & _processed_texts(ROOT / "data" / "processed" / "vlsp2016" / "test.jsonl")
            ),
            "test_ids_all_train_prefix": test_ids_all_train_prefix,
        },
        "notes": [
            "No deduplication; source order preserved except stratified train/dev split.",
            "AIVIVN test IDs all use the train_ prefix; 10 exact texts overlap train.",
            "Treat the provided test split as a held-out split from the same corpus, not the official AIVIVN public test.",
            "Binary head: 0=negative, 1=positive; do not mix with the 3-class UIT/VLSP head.",
        ],
    }


def main() -> int:
    args = _parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    raw = {split: _read_rows(raw_dir / f"{split}.csv") for split in ("train", "test")}
    segmenter = VnCoreNLPSegmenter()
    train_rows = _segment(raw["train"], segmenter, "train")
    test_rows = _segment(raw["test"], segmenter, "test")
    train_rows, dev_rows = _stratified_split(train_rows, DEV_RATIO, SEED)
    processed = {"train": train_rows, "dev": dev_rows, "test": test_rows}
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
