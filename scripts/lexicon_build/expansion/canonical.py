"""Canonicalize WordNet terms with a persistent per-term cache."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lexicon_build.keys import canonicalize_key

Canonicalize = Callable[[str], str | None]


class CachedCanonicalizer:
    def __init__(self, segmenter: Any, cache_path: Path | None = None):
        self._segmenter = segmenter
        self._cache_path = cache_path
        self._cache: dict[str, str | None] = {}
        if cache_path is not None and cache_path.is_file():
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            self._cache = {str(key): value for key, value in loaded.items()}

    def __call__(self, raw: str) -> str | None:
        term = str(raw).strip()
        if not term:
            return None
        if term in self._cache:
            return self._cache[term]
        key = canonicalize_key(term, self._segmenter)
        result = None if not key or " " in key else key
        self._cache[term] = result
        return result

    def flush(self) -> None:
        if self._cache_path is None:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self._cache, handle, ensure_ascii=False, sort_keys=True, indent=0)
            handle.write("\n")


def build_canonicalizer(
    segmenter: Any, cache_path: Path | None = None
) -> CachedCanonicalizer:
    return CachedCanonicalizer(segmenter, cache_path)
