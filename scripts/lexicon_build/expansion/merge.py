"""Non-destructive merge of expanded values into the frozen composite."""

from __future__ import annotations

from typing import Any

from .config import SOURCE_PRIORITY
from .synonym import ExpandedValue


def merge_entries(
    base_entries: dict[str, dict[str, list]],
    expanded: list[ExpandedValue],
    audited: set[str],
    *,
    fill_conflicts: bool = False,
) -> tuple[dict[str, dict[str, list]], list[dict[str, Any]]]:
    merged = {
        key: {"values": list(entry["values"]), "mask": list(entry["mask"])}
        for key, entry in base_entries.items()
    }
    logs: list[dict[str, Any]] = []
    accepted: set[tuple[str, int]] = set()

    ordered = sorted(
        expanded,
        key=lambda item: (SOURCE_PRIORITY.get(item.source, 99), item.key, item.dim),
    )
    for item in ordered:
        entry = merged.get(item.key)
        if entry is not None and entry["mask"][item.dim] == 1:
            reason = (
                "source_precedence"
                if (item.key, item.dim) in accepted
                else "already_active"
            )
            logs.append(_row(item, "rejected", reason))
            continue
        if item.key in audited and not fill_conflicts:
            logs.append(_row(item, "rejected", "audited_conflict"))
            continue
        if entry is None:
            entry = {"values": [0.0] * 12, "mask": [0] * 12}
            merged[item.key] = entry
        entry["values"][item.dim] = item.value
        entry["mask"][item.dim] = 1
        accepted.add((item.key, item.dim))
        reason = "audited_override" if item.key in audited else "accepted"
        logs.append(_row(item, "accepted", reason))
    return merged, logs


def _row(item: ExpandedValue, decision: str, reason: str) -> dict[str, Any]:
    return {
        "key": item.key,
        "dim": item.dim,
        "value": item.value,
        "source": item.source,
        "confidence": item.confidence,
        "seeds": item.seeds,
        "seed_values": item.seed_values,
        "detail": item.detail,
        "decision": decision,
        "reason": reason,
    }
