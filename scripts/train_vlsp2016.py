#!/usr/bin/env python3
"""Fine-tune the fused model on VLSP2016 train with a stratified dev split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from vnsentiment.training.factory import build_sentiment_model  # noqa: E402
from vnsentiment.training.loop import Trainer  # noqa: E402
from vnsentiment.training.vlsp_data import load_vlsp2016  # noqa: E402
from vnsentiment.training.vlsp_split import stratified_split  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "vlsp2016_best.pt"
SPLIT_MANIFEST = ROOT / "artifacts" / "splits" / "vlsp2016_train_dev.json"
DATA_MANIFEST = ROOT / "data" / "processed" / "vlsp2016" / "manifest.json"


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune PhoBERT + lexicon fusion on VLSP2016 train."
    )
    parser.add_argument(
        "--smoke", action="store_true", help="200 train / 100 dev, 1 epoch."
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--lr-fusion", type=float, default=5e-5)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--lexicon-path",
        default=None,
        help="Composite JSON; mặc định dùng v0.2.0 của LexiconStore.",
    )
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20:
        parser.error("--epochs must be between 1 and 20")
    return args


def _label_counts(samples) -> dict[str, int]:
    return {
        str(label): count
        for label, count in sorted(
            Counter(row["label"] for row in samples).items()
        )
    }


def _save_split_manifest(fit, dev, args) -> None:
    data_manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    SPLIT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_MANIFEST.write_text(
        json.dumps(
            {
                "source": "data/processed/vlsp2016/train.jsonl",
                "source_sha256": data_manifest["splits"]["train"]["output_sha256"],
                "seed": args.seed,
                "dev_ratio": args.dev_ratio,
                "train_rows": len(fit),
                "dev_rows": len(dev),
                "train_label_counts": _label_counts(fit),
                "dev_label_counts": _label_counts(dev),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = _parse_args()
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    print(f"device: {device}")

    samples = load_vlsp2016("train")
    fit, dev = stratified_split(samples, dev_ratio=args.dev_ratio, seed=args.seed)
    checkpoint = Path(args.checkpoint)
    epochs = args.epochs
    if args.smoke:
        fit, dev = fit[:200], dev[:100]
        epochs = 1
        checkpoint = None
        print("smoke: 200 train / 100 dev, 1 epoch, no checkpoint")
    else:
        _save_split_manifest(fit, dev, args)
        print(f"train/dev rows: {len(fit)}/{len(dev)}")
        print(f"split manifest: {SPLIT_MANIFEST}")

    model = build_sentiment_model(device, lexicon_path=args.lexicon_path)
    trainer = Trainer(
        model,
        device=device,
        lr_encoder=2e-5,
        lr_fusion=args.lr_fusion,
        weight_decay=0.01,
        batch_size=args.batch_size,
        accum_steps=args.accum,
        max_epochs=epochs,
        patience=3,
        seed=args.seed,
        checkpoint_path=checkpoint,
        verbose=True,
    )
    summary = trainer.fit(fit, dev)

    print(f"best epoch: {summary['best_epoch']}")
    print(f"best dev macro-F1: {summary['best_metrics']['macro_f1']:.4f}")
    if checkpoint is not None:
        print(f"checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
