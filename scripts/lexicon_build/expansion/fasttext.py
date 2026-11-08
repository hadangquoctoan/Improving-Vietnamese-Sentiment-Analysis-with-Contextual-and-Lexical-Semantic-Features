"""Stream fastText .vec files and propagate polarity from nearest seeds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .synonym import ExpandedValue
from .vad import vad_conflict


def syllables(key: str) -> tuple[str, ...]:
    return tuple(key.split("_")) if "_" in key else (key,)


class VectorStore:
    def __init__(self, vectors: dict[str, np.ndarray], dim: int):
        self._vectors = vectors
        self._dim = dim
        self._cache: dict[str, np.ndarray | None] = {}

    @classmethod
    def stream(cls, path: Path, needed: set[str]) -> "VectorStore":
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().split()
            if len(header) != 2:
                raise ValueError(f"invalid fastText header in {path}")
            dim = int(header[1])
            vectors: dict[str, np.ndarray] = {}
            for line in handle:
                key, _, payload = line.rstrip("\n").partition(" ")
                if key not in needed or not payload:
                    continue
                vector = np.fromstring(payload, sep=" ", dtype=np.float32)
                if vector.size != dim:
                    continue
                vectors[key] = vector
        return cls(vectors, dim)

    def __len__(self) -> int:
        return len(self._vectors)

    def vector(self, key: str) -> np.ndarray | None:
        if key in self._cache:
            return self._cache[key]
        if "_" not in key:
            vector = self._vectors.get(key)
        else:
            parts = key.split("_")
            vectors = [self._vectors.get(part) for part in parts]
            vector = (
                None
                if any(item is None for item in vectors)
                else np.mean(np.stack(vectors), axis=0)
            )
        if vector is not None:
            norm = float(np.linalg.norm(vector))
            vector = None if norm == 0.0 else vector / norm
        self._cache[key] = vector
        return vector


def propagate_fasttext(
    store: VectorStore,
    targets: list[str],
    seeds: dict[str, float],
    *,
    base_entries: dict[str, dict[str, list]] | None = None,
    top_k: int = 10,
    min_cosine: float = 0.55,
    min_sign_agreement: float = 0.8,
    min_neighbors: int = 2,
    min_mean_sim: float = 0.65,
    shrink: float = 0.5,
    min_abs_score: float = 0.15,
) -> tuple[list[ExpandedValue], list[dict[str, Any]]]:
    seed_keys: list[str] = []
    seed_values: list[float] = []
    rows: list[np.ndarray] = []
    for key in sorted(seeds):
        vector = store.vector(key)
        if vector is None:
            continue
        seed_keys.append(key)
        seed_values.append(float(seeds[key]))
        rows.append(vector)
    if not rows:
        return [], []

    matrix = np.stack(rows)
    values = np.array(seed_values, dtype=np.float32)
    expanded: list[ExpandedValue] = []
    logs: list[dict[str, Any]] = []

    def log(key: str, reason: str, detail: str) -> None:
        logs.append(
            {"key": key, "dim": 0, "reason": reason, "source": "fasttext_nn", "detail": detail}
        )

    for target in sorted(targets):
        vector = store.vector(target)
        if vector is None:
            log(target, "no_vector", "missing syllable vector")
            continue
        sims = matrix @ vector
        k = min(top_k, len(sims))
        if k == len(sims):
            order = np.argsort(-sims)
        else:
            order = np.argpartition(-sims, k - 1)[:k]
            order = order[np.argsort(-sims[order])]
        keep = sims[order] >= min_cosine
        order = order[keep]
        sims_top = sims[order]
        if len(order) == 0:
            log(target, "low_similarity", f"top_k={k};min_cosine={min_cosine}")
            continue
        if len(order) < min_neighbors:
            log(target, "low_neighbors", f"neighbors={len(order)}")
            continue
        mean_sim = float(np.mean(sims_top))
        if mean_sim < min_mean_sim:
            log(target, "low_mean_sim", f"mean_sim={mean_sim:.3f}")
            continue
        score = float(np.dot(sims_top, values[order]) / np.sum(sims_top))
        if score == 0.0:
            log(target, "low_score", "weighted score is zero")
            continue
        sign = 1 if score > 0 else -1
        agreement = float(np.mean(np.sign(values[order]) == sign))
        if agreement < min_sign_agreement:
            log(target, "low_sign_agreement", f"agreement={agreement:.3f}")
            continue
        score = max(-1.0, min(1.0, score * shrink))
        if abs(score) < min_abs_score:
            log(target, "low_score", f"shrunk={score:.4f}")
            continue
        if vad_conflict(base_entries.get(target) if base_entries else None, score):
            log(target, "vad_conflict", f"value={score:.4f}")
            continue
        expanded.append(
            ExpandedValue(
                key=target,
                dim=0,
                value=round(score, 6),
                source="fasttext_nn",
                confidence=mean_sim,
                seeds=tuple(seed_keys[index] for index in order),
                seed_values=tuple(float(values[index]) for index in order),
                detail=f"top_k={len(order)};mean_sim={mean_sim:.3f}",
            )
        )
    return expanded, logs
