#!/usr/bin/env python3
"""Train the fused sentiment model on UIT-VSFC train/dev."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from vnsentiment.training.data import load_split  # noqa: E402
from vnsentiment.training.factory import build_sentiment_model  # noqa: E402
from vnsentiment.training.loop import Trainer  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "best_fused.pt"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune PhoBERT + lexicon fusion; never touches test."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 epoch on 200 train samples, no checkpoint.",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--lexicon-path",
        default=None,
        help="Composite JSON; mặc định dùng v0.2.0 của LexiconStore.",
    )
    args = parser.parse_args()
    if not 1 <= args.epochs <= 10:
        parser.error("--epochs must be between 1 and 10")
    return args


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    train_samples = load_split("train")
    dev_samples = load_split("dev")
    epochs = args.epochs
    checkpoint = Path(args.checkpoint)
    if args.smoke:
        train_samples = train_samples[:200]
        epochs = 1
        checkpoint = None
        print("smoke: 200 train samples, 1 epoch, no checkpoint")

    model = build_sentiment_model(device, lexicon_path=args.lexicon_path)
    trainer = Trainer(
        model,
        device=device,
        batch_size=args.batch_size,
        accum_steps=args.accum,
        max_epochs=epochs,
        seed=args.seed,
        checkpoint_path=checkpoint,
        verbose=True,
    )
    summary = trainer.fit(train_samples, dev_samples)

    print(f"best epoch: {summary['best_epoch']}")
    print(f"best dev macro-F1: {summary['best_metrics']['macro_f1']:.4f}")
    if checkpoint is not None:
        print(f"checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
