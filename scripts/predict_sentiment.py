#!/usr/bin/env python3
"""Predict Neg/Neu/Pos for raw Vietnamese feedback with the trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnsentiment.inference import SentimentPredictor  # noqa: E402
from vnsentiment.preprocessing.collapse import normalize_basic  # noqa: E402
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best_fused.pt"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Dự đoán Neg/Neu/Pos bằng checkpoint best_fused.pt."
    )
    parser.add_argument("text", nargs="*", help="Câu cần dự đoán.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--segmented",
        action="store_true",
        help="Input đã tách từ bằng dấu gạch dưới (bỏ qua VnCoreNLP).",
    )
    return parser.parse_args()


def _prepare(text: str, segmenter, already_segmented: bool) -> str:
    normalized = normalize_basic(text)
    if not normalized:
        raise ValueError("câu rỗng sau chuẩn hoá")
    if already_segmented:
        return normalized
    return normalize_basic(segmenter.segment(normalized))


def _print_result(text: str, segmented: str, result: dict) -> None:
    probabilities = " ".join(
        f"{label}={value:.4f}"
        for label, value in result["probabilities"].items()
    )
    print(f"\n> {text}")
    print(f"  segmented: {segmented}")
    print(f"  label: {result['label']} ({result['label_id']})")
    print(f"  probs: {probabilities}")


def _run(texts, predictor, segmenter, already_segmented):
    prepared = [_prepare(text, segmenter, already_segmented) for text in texts]
    results = predictor.predict(prepared)
    for text, segmented, result in zip(texts, prepared, results):
        _print_result(text, segmented, result)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()
    try:
        predictor = SentimentPredictor.from_checkpoint(
            args.checkpoint, device=_resolve_device(args.device)
        )
    except Exception as error:
        print(f"Lỗi nạp checkpoint: {error}", file=sys.stderr)
        return 1

    print(
        "Lưu ý: checkpoint train trên UIT-VSFC (phản hồi sinh viên); "
        "câu khác miền có thể kém tin cậy."
    )
    segmenter = None if args.segmented else VnCoreNLPSegmenter()
    try:
        if args.text:
            _run(args.text, predictor, segmenter, args.segmented)
            return 0

        print("Nhập câu cần dự đoán (q để thoát).")
        while True:
            try:
                text = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if text.strip().lower() in {"q", "quit", "exit"}:
                return 0
            if not text.strip():
                continue
            _run([text], predictor, segmenter, args.segmented)
    except Exception as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
