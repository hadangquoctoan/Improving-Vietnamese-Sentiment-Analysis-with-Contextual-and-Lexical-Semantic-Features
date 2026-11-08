"""Load Vietnamese WordNet synonym groups as canonical keys."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import Canonicalize


@dataclass(frozen=True)
class SynonymGroup:
    source_file: str
    terms: tuple[str, ...]


def load_groups(
    root: Path,
    files: Sequence[str],
    canonicalize: Canonicalize,
) -> tuple[list[SynonymGroup], list[dict[str, Any]]]:
    groups: list[SynonymGroup] = []
    dropped: list[dict[str, Any]] = []
    for name in files:
        path = root / f"{name}.csv"
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            raw_terms = [term.strip() for term in line.split(",") if term.strip()]
            if not raw_terms:
                continue
            terms: list[str] = []
            for raw_term in raw_terms:
                key = canonicalize(raw_term)
                if key is None:
                    dropped.append(
                        {
                            "source": name,
                            "line": line_no,
                            "raw_term": raw_term,
                            "canonical_key": "",
                            "detail": "empty or multi-token after canonicalization",
                        }
                    )
                    continue
                if key not in terms:
                    terms.append(key)
            if len(terms) < 2:
                continue
            groups.append(SynonymGroup(name, tuple(terms)))
    return groups, dropped


def build_key_file_map(
    root: Path,
    files: Sequence[str],
    canonicalize: Canonicalize,
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, set[str]] = {}
    for name in files:
        path = root / f"{name}.csv"
        for line in path.read_text(encoding="utf-8").splitlines():
            for raw_term in line.split(","):
                if not raw_term.strip():
                    continue
                key = canonicalize(raw_term)
                if key is None:
                    continue
                mapping.setdefault(key, set()).add(name)
    return {key: tuple(sorted(names)) for key, names in mapping.items()}
