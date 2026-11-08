"""Cross-source sign consistency between polarity and VAD valence."""

from __future__ import annotations

from .config import VAD_MARGIN, VAD_NEUTRAL


def vad_conflict(entry: dict[str, list] | None, polarity: float) -> bool:
    if entry is None or entry["mask"][1] != 1:
        return False
    valence = float(entry["values"][1])
    if abs(valence - VAD_NEUTRAL) < VAD_MARGIN:
        return False
    return (polarity > 0) != (valence > VAD_NEUTRAL)
