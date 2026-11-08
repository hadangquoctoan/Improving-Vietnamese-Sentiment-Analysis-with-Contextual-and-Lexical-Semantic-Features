"""Polarity seeds, audited conflicts, and the unconditional stoplist."""

from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Sequence

from .canonical import Canonicalize
from .config import HARD_STOPWORDS, PRONOUN_STOPWORDS, VAD_MARGIN, VAD_NEUTRAL


def load_seeds(
    entries: dict[str, dict[str, list]],
    min_abs: float = 0.5,
    *,
    require_vad_consistency: bool = True,
) -> dict[str, float]:
    seeds: dict[str, float] = {}
    for key, entry in entries.items():
        if entry["mask"][0] != 1:
            continue
        value = float(entry["values"][0])
        if abs(value) < min_abs:
            continue
        if require_vad_consistency and entry["mask"][1] == 1:
            valence = float(entry["values"][1])
            if abs(valence - VAD_NEUTRAL) >= VAD_MARGIN and (value > 0) != (
                valence > VAD_NEUTRAL
            ):
                continue
        seeds[key] = value
    return seeds


def load_audited_conflicts(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return {row["key"] for row in reader if row.get("key")}


def load_stoplist(
    wordnet_root: Path,
    canonicalize: Canonicalize,
    files: Sequence[str] = ("conj", "prep", "pron"),
) -> set[str]:
    stoplist: set[str] = set()
    for word in (*HARD_STOPWORDS, *PRONOUN_STOPWORDS):
        key = canonicalize(word)
        if key is not None:
            stoplist.add(key)
    for name in files:
        path = wordnet_root / f"{name}.csv"
        for line in path.read_text(encoding="utf-8").splitlines():
            for raw_term in line.split(","):
                if not raw_term.strip():
                    continue
                key = canonicalize(raw_term)
                if key is not None:
                    stoplist.add(key)
    return stoplist
