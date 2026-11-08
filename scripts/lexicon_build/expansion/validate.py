"""Invariants that every expanded composite must satisfy."""

from __future__ import annotations

from typing import Any


def assert_non_destructive(
    base_entries: dict[str, dict[str, list]],
    merged: dict[str, dict[str, list]],
) -> None:
    for key, entry in base_entries.items():
        for dim, flag in enumerate(entry["mask"]):
            if flag != 1:
                continue
            assert key in merged, f"active key disappeared: {key!r}"
            assert merged[key]["mask"][dim] == 1, f"active mask cleared: {key!r} dim {dim}"
            assert merged[key]["values"][dim] == entry["values"][dim], (
                f"active value changed: {key!r} dim {dim}"
            )


def assert_provenance_complete(
    base_entries: dict[str, dict[str, list]],
    merged: dict[str, dict[str, list]],
    expansions: list[dict[str, Any]],
) -> None:
    proven = {
        (row["key"], row["dim"])
        for row in expansions
        if row.get("decision", "accepted") == "accepted"
    }
    for key, entry in merged.items():
        base = base_entries.get(key)
        for dim, flag in enumerate(entry["mask"]):
            if flag != 1:
                continue
            base_flag = 0 if base is None else base["mask"][dim]
            if base_flag == 1:
                continue
            assert (key, dim) in proven, f"missing provenance: {key!r} dim {dim}"


def assert_no_empty_entries(merged: dict[str, dict[str, list]]) -> None:
    for key, entry in merged.items():
        assert any(flag == 1 for flag in entry["mask"]), f"all-zero entry: {key!r}"
