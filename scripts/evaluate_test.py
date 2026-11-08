#!/usr/bin/env python3
"""Evaluate the chosen checkpoint on UIT-VSFC test exactly once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vnsentiment.training.evaluate import evaluate_test  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best_fused.pt"
DEFAULT_REPORT = ROOT / "artifacts" / "reports" / "test_eval.json"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="One-shot test evaluation; refuses to overwrite a report."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--lexicon-path",
        default=None,
        help="Composite JSON; mặc định dùng v0.2.0 của LexiconStore.",
    )
    args = parser.parse_args()

    report = evaluate_test(
        args.checkpoint,
        args.report,
        device=_resolve_device(args.device),
        batch_size=args.batch_size,
        lexicon_path=args.lexicon_path,
    )
    metrics = report["metrics"]
    print(f"test accuracy: {metrics['accuracy']:.4f}")
    print(f"test macro-F1: {metrics['macro_f1']:.4f}")
    print(f"test F1 per class: {metrics['f1_per_class']}")
    print(f"coverage train/dev/test: {report['coverage']}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
