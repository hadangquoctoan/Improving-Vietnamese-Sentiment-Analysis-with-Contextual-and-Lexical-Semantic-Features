"""One-shot test evaluation with provenance and lexicon coverage."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from vnsentiment.contracts.train_batch import TrainBatch
from vnsentiment.training.data import load_split
from vnsentiment.training.factory import build_sentiment_model
from vnsentiment.training.metrics import compute_metrics

_DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "lexicons"
    / "composite_12d"
    / "v0.2.0"
    / "composite_lexicon.json"
)
_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "processed" / "manifest.json"
)


def evaluate_test(
    checkpoint_path: str | Path,
    report_path: str | Path,
    device: str = "cpu",
    batch_size: int = 8,
    lexicon_path: str | Path | None = None,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError(f"test report already exists: {report_path}")
    resolved_lexicon = (
        Path(lexicon_path) if lexicon_path is not None else _DEFAULT_LEXICON_PATH
    )

    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model = build_sentiment_model(device, lexicon_path=resolved_lexicon)
    model.load_state_dict(payload["model_state"])
    model.eval()

    test_samples = load_split("test", allow_test=True)
    metrics = _evaluate_samples(model, test_samples, batch_size)
    report = _build_report(
        checkpoint_path,
        payload,
        test_samples,
        model.lexicon.store,
        metrics,
        device,
        resolved_lexicon,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _build_report(
    checkpoint_path, payload, test_samples, store, metrics, device, lexicon
) -> dict:
    return {
        "run_id": _run_id(checkpoint_path),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "epoch": payload.get("epoch"),
            "dev_metrics": payload.get("dev_metrics"),
            "seed": payload.get("seed"),
            "config": payload.get("config"),
        },
        "data": {
            "split": "test",
            "rows": len(test_samples),
            "manifest_sha256": _sha256(_MANIFEST_PATH),
            "split_sha256": _split_hashes(),
        },
        "lexicon": {
            "path": str(lexicon),
            "sha256": _sha256(lexicon),
        },
        "runtime": _runtime(device),
        "metrics": _metrics_dict(metrics),
        "coverage": {
            "train": _coverage(load_split("train"), store),
            "dev": _coverage(load_split("dev"), store),
            "test": _coverage(test_samples, store),
        },
    }


def _evaluate_samples(model, samples, batch_size):
    predictions = []
    labels = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = TrainBatch.from_samples(chunk)
            logits = model(batch.texts)
            predictions.append(logits.argmax(dim=-1).cpu())
            labels.append(batch.labels)
    return compute_metrics(torch.cat(predictions), torch.cat(labels))


def _coverage(samples, store) -> float:
    matched = 0
    total = 0
    for sample in samples:
        for token in sample["text"].split():
            if not any(char.isalnum() for char in token):
                continue
            total += 1
            if store.lookup(token) is not None:
                matched += 1
    return matched / total if total else 0.0


def _metrics_dict(metrics) -> dict:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "f1_per_class": list(metrics["f1_per_class"]),
    }


def _split_hashes() -> dict:
    payload = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        split: info["output_sha256"]
        for split, info in payload["splits"].items()
    }


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _run_id(checkpoint_path) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{Path(checkpoint_path).stem}"
