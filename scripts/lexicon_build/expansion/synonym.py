"""Propagate polarity through Vietnamese WordNet synonym groups."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from .vad import vad_conflict
from .wordnet import SynonymGroup


@dataclass(frozen=True)
class ExpandedValue:
    key: str
    dim: int
    value: float
    source: str
    confidence: float
    seeds: tuple[str, ...]
    seed_values: tuple[float, ...]
    detail: str


def propagate_synonyms(
    groups: list[SynonymGroup],
    seeds: dict[str, float],
    base_entries: dict[str, dict[str, list]],
    audited: set[str],
    stoplist: set[str],
    *,
    fill_conflicts: bool = False,
) -> tuple[list[ExpandedValue], list[dict[str, Any]]]:
    candidates: dict[str, list[tuple[float, str, dict[str, float]]]] = {}
    logs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def log(key: str, reason: str, detail: str) -> None:
        token = (key, reason)
        if token in seen:
            return
        seen.add(token)
        logs.append(
            {"key": key, "dim": 0, "reason": reason, "source": "wordnet_syn", "detail": detail}
        )

    for group in groups:
        hit = {term: seeds[term] for term in group.terms if term in seeds}
        if not hit:
            continue
        signs = {1 if value > 0 else -1 for value in hit.values()}
        if len(signs) != 1:
            log(
                group.terms[0],
                "sign_conflict",
                f"file={group.source_file};terms={','.join(group.terms)};seeds={hit}",
            )
            continue
        value = float(median(hit.values()))
        for term in group.terms:
            if term in base_entries and base_entries[term]["mask"][0] == 1:
                log(term, "already_active", f"file={group.source_file}")
                continue
            if term in audited and not fill_conflicts:
                log(term, "audited_conflict", f"file={group.source_file}")
                continue
            if term in stoplist:
                log(term, "stoplist", f"file={group.source_file}")
                continue
            if vad_conflict(base_entries.get(term), value):
                log(term, "vad_conflict", f"file={group.source_file}")
                continue
            candidates.setdefault(term, []).append((value, group.source_file, hit))

    expanded: list[ExpandedValue] = []
    for key in sorted(candidates):
        items = candidates[key]
        values = [item[0] for item in items]
        if len({1 if value > 0 else -1 for value in values}) != 1:
            log(key, "multi_group_disagreement", f"values={values}")
            continue
        seed_union = tuple(sorted({seed for _, _, hit in items for seed in hit}))
        expanded.append(
            ExpandedValue(
                key=key,
                dim=0,
                value=round(float(median(values)), 6),
                source="wordnet_syn",
                confidence=1.0,
                seeds=seed_union,
                seed_values=tuple(seeds[seed] for seed in seed_union),
                detail=(
                    f"groups={len(items)};spread={max(values) - min(values):.3f};"
                    f"files={','.join(sorted({source for _, source, _ in items}))}"
                ),
            )
        )
    return expanded, logs
