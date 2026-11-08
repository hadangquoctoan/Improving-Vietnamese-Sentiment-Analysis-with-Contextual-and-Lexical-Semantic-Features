#!/usr/bin/env python3
"""Benchmark a checkpoint on the user-provided VLSP2016 test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from vnsentiment.inference import SentimentPredictor  # noqa: E402
from vnsentiment.training.metrics import compute_metrics  # noqa: E402
from vnsentiment.training.vlsp_data import load_vlsp2016  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best.pt"
DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "vlsp2016_test_best.json"
MANIFEST = ROOT / "data" / "processed" / "vlsp2016" / "manifest.json"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime(device: str) -> dict:
    resolved = torch.device(device)
    runtime = {
        "device": str(resolved),
        "torch": torch.__version__,
        "gpu": None,
        "vram_bytes": None,
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index or 0
        runtime["gpu"] = torch.cuda.get_device_name(index)
        runtime["vram_bytes"] = torch.cuda.get_device_properties(
            index
        ).total_memory
    return runtime


def _confusion(preds, labels) -> list[list[int]]:
    matrix = [[0, 0, 0] for _ in range(3)]
    for true, pred in zip(labels.tolist(), preds.tolist()):
        matrix[true][pred] += 1
    return matrix


def _parse_args():
    parser = argparse.ArgumentParser(
        description="One-shot VLSP2016 test benchmark of a checkpoint."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-dataset", default="UIT-VSFC")
    parser.add_argument("--note", action="append", default=[])
    return parser.parse_args()


def _run_predictions(checkpoint, device, batch_size):
    samples = load_vlsp2016("test", allow_test=True)
    texts = [sample["text"] for sample in samples]
    labels = torch.tensor([sample["label"] for sample in samples])
    predictor = SentimentPredictor.from_checkpoint(checkpoint, device)
    results = predictor.predict(texts, batch_size=batch_size)
    preds = torch.tensor([result["label_id"] for result in results])
    return samples, texts, labels, preds


def _build_report(
    checkpoint, payload, samples, texts, metrics, matrix, device,
    training_dataset, extra_notes
):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "run_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{checkpoint.stem}",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "epoch": payload.get("epoch"),
            "dev_metrics": payload.get("dev_metrics"),
            "seed": payload.get("seed"),
            "config": payload.get("config"),
        },
        "dataset": {
            "name": manifest["dataset"],
            "manifest_sha256": _sha256(MANIFEST),
            "test_output_sha256": manifest["splits"]["test"]["output_sha256"],
            "raw_test_sha256": manifest["source"]["files"]["test.csv"]["sha256"],
            "rows": len(samples),
            "label_counts": manifest["splits"]["test"]["label_counts"],
            "label_map": manifest["label_map"],
            "integrity": manifest["integrity"],
            "max_length": 256,
            "texts_over_256_tokens": sum(
                len(text.split()) > 256 for text in texts
            ),
        },
        "runtime": _runtime(device),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "f1_per_class": list(metrics["f1_per_class"]),
        },
        "confusion_matrix": matrix,
        "notes": [
            f"Checkpoint training data: {training_dataset}.",
            "Checkpoint dev_metrics are from its own training protocol.",
            "One-shot evaluation on this test; no tuning from this result.",
            "Class order: negative=0, neutral=1, positive=2.",
            *extra_notes,
        ],
    }


def _print_summary(checkpoint, payload, samples, metrics, matrix, path):
    print(f"checkpoint: {checkpoint} (epoch {payload.get('epoch')})")
    print(f"test rows: {len(samples)}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"macro-F1: {metrics['macro_f1']:.4f}")
    print(f"F1 per class: {list(metrics['f1_per_class'])}")
    print(f"confusion (rows=true, cols=pred): {matrix}")
    print(f"report: {path}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError(f"benchmark report already exists: {report_path}")

    device = _resolve_device(args.device)
    payload = torch.load(
        checkpoint, map_location="meta", weights_only=True
    )
    samples, texts, labels, preds = _run_predictions(
        checkpoint, device, args.batch_size
    )
    metrics = compute_metrics(preds, labels)
    matrix = _confusion(preds, labels)
    report = _build_report(
        checkpoint,
        payload,
        samples,
        texts,
        metrics,
        matrix,
        device,
        args.training_dataset,
        args.note,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print_summary(
        checkpoint, payload, samples, metrics, matrix, report_path
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
