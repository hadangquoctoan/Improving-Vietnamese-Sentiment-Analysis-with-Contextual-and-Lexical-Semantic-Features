#!/usr/bin/env python3
"""Generate benchmark figures (PNG + SVG) for slides."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"
FIGURES = REPORTS / "figures"
LOG = ROOT / "artifacts" / "logs" / "train_aivivn.log"

REPORT_FILES = {
    "UIT-VSFC": REPORTS / "test_eval.json",
    "VLSP zero-shot": REPORTS / "vlsp2016_test_best.json",
    "VLSP fine-tuned": REPORTS / "vlsp2016_test_finetuned.json",
    "AIVIVN (binary)": REPORTS / "aivivn2019_test.json",
}
CLASS_NAMES = ("Negative", "Neutral", "Positive")
BINARY_NAMES = ("Negative", "Positive")
COLORS = ("#2E86AB", "#A23B72", "#F18F01", "#3B7A57")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _save(fig, name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            output_dir / f"{name}.{suffix}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def _model_rows():
    rows = []
    for name, path in REPORT_FILES.items():
        payload = _load(path)
        if payload is None:
            continue
        rows.append((name, payload["metrics"], payload.get("confusion_matrix")))
    return rows


def _macro_f1_chart(rows, output_dir):
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    labels = [row[0] for row in rows]
    macro = [row[1]["macro_f1"] * 100 for row in rows]
    accuracy = [row[1]["accuracy"] * 100 for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color=COLORS[0])
    bars2 = ax.bar(x + width / 2, macro, width, label="Macro-F1", color=COLORS[1])
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)")
    ax.set_title("Sentiment benchmark comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_macro_f1", output_dir)


def _per_class_chart(rows, output_dir):
    three = [row for row in rows if len(row[1]["f1_per_class"]) == 3]
    binary = [row for row in rows if len(row[1]["f1_per_class"]) == 2]
    columns = int(bool(three)) + int(bool(binary))
    fig, axes = plt.subplots(1, max(columns, 1), figsize=(11, 4.4), squeeze=False)
    axes = axes[0]
    index = 0
    if three:
        ax = axes[index]; index += 1
        names = [row[0] for row in three]
        x = np.arange(len(names))
        width = 0.8 / 3
        for class_index, class_name in enumerate(CLASS_NAMES):
            values = [row[1]["f1_per_class"][class_index] * 100 for row in three]
            ax.bar(x + (class_index - 1) * width, values, width, label=class_name)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 105)
        ax.set_ylabel("F1 (%)")
        ax.set_title("Per-class F1 (3-class models)")
        ax.legend(
            frameon=False, ncol=3, fontsize=8,
            loc="upper center", bbox_to_anchor=(0.5, -0.12),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    if binary:
        ax = axes[index]
        names = [row[0] for row in binary]
        x = np.arange(len(names))
        width = 0.8 / 2
        for class_index, class_name in enumerate(BINARY_NAMES):
            values = [row[1]["f1_per_class"][class_index] * 100 for row in binary]
            ax.bar(x + (class_index - 0.5) * width, values, width, label=class_name)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 105)
        ax.set_title("Per-class F1 (binary)")
        ax.legend(
            frameon=False, ncol=2, fontsize=8,
            loc="upper center", bbox_to_anchor=(0.5, -0.12),
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_f1_per_class", output_dir)


def _confusion_chart(rows, output_dir):
    rows = [row for row in rows if row[2] is not None]
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.4 * len(rows), 4.0), squeeze=False)
    for ax, (name, metrics, matrix) in zip(axes[0], rows):
        array = np.array(matrix)
        names = BINARY_NAMES if array.shape == (2, 2) else CLASS_NAMES
        ax.imshow(array, cmap="Blues")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}\nmacro-F1 {metrics['macro_f1'] * 100:.1f}%", fontsize=9)
        threshold = array.max() / 2
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                ax.text(
                    column,
                    row,
                    str(array[row, column]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if array[row, column] > threshold else "#222222",
                )
    fig.tight_layout()
    _save(fig, "fig_confusion_matrices", output_dir)


def _coverage_rows():
    base = _load(ROOT / "data" / "lexicons" / "composite_12d" / "v0.2.0" / "composite_lexicon.json")
    rows = []
    if base:
        for split, key in (("train", "train"), ("dev", "dev")):
            coverage = base["stats"]["coverage"][key]
            rows.append(
                {
                    "version": "v0.2.0",
                    "split": split,
                    "polarity": coverage["per_dimension"]["polarity"]["word_token_coverage_pct"],
                    "overall": coverage["word_token_coverage"]["token_coverage_pct"],
                    "eligible": None,
                }
            )
    for version in ("v0.3.2", "v0.3.3"):
        payload = _load(
            ROOT / "data" / "lexicons" / "composite_12d" / version / "coverage_expansion.json"
        )
        if not payload:
            continue
        for split in ("train", "dev"):
            rows.append(
                {
                    "version": version,
                    "split": split,
                    "polarity": payload[split]["expanded"]["per_dimension"]["polarity"]["word_token_coverage_pct"],
                    "overall": payload[split]["expanded"]["word_token_coverage"]["token_coverage_pct"],
                    "eligible": payload[split]["sentiment_eligible"]["expanded"]["token_coverage_pct"],
                }
            )
    return rows


def _coverage_chart(rows, output_dir):
    if not rows:
        return
    versions = sorted({row["version"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, metric, title in zip(
        axes, ("polarity", "overall"), ("Polarity coverage", "Any-dimension coverage")
    ):
        width = 0.8 / len(versions)
        for index, version in enumerate(versions):
            values = []
            for split in ("train", "dev"):
                match = [row for row in rows if row["version"] == version and row["split"] == split]
                values.append(match[0][metric] if match else 0.0)
            x = np.arange(2) + (index - (len(versions) - 1) / 2) * width
            bars = ax.bar(x, values, width, label=version)
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.6,
                    f"{bar.get_height():.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        ax.set_xticks(range(2))
        ax.set_xticklabels(("train", "dev"))
        ax.set_ylabel("% word-token coverage")
        ax.set_title(title)
        ax.set_ylim(0, max(100, max(row[metric] for row in rows) + 8))
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_lexicon_coverage", output_dir)


def _training_curve(output_dir):
    if not LOG.is_file():
        return
    epochs, losses, macro = [], [], []
    pattern = re.compile(
        r"epoch (\d+): train_loss=([0-9.]+) dev_macro_f1=([0-9.]+)"
    )
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            macro.append(float(match.group(3)) * 100)
    if not epochs:
        return
    fig, ax1 = plt.subplots(figsize=(8.2, 4.3))
    ax1.plot(epochs, losses, marker="o", color=COLORS[0], label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color=COLORS[0])
    ax2 = ax1.twinx()
    ax2.plot(epochs, macro, marker="s", color=COLORS[1], label="Dev macro-F1")
    ax2.set_ylabel("Dev macro-F1 (%)", color=COLORS[1])
    best_epoch = epochs[int(np.argmax(macro))]
    ax1.axvline(best_epoch, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    ax1.annotate(
        f"best epoch {best_epoch}",
        xy=(best_epoch, max(losses)),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
        color="#444444",
    )
    ax1.set_xticks(epochs)
    ax1.set_title("AIVIVN binary training curve")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_training_curve", output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(FIGURES))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    rows = _model_rows()
    if rows:
        _macro_f1_chart(rows, output_dir)
        _per_class_chart(rows, output_dir)
        _confusion_chart(rows, output_dir)
    _coverage_chart(_coverage_rows(), output_dir)
    _training_curve(output_dir)
    print(f"figures written to: {output_dir}")
    for path in sorted(output_dir.glob("*")):
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
