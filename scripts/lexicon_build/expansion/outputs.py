"""Write the expanded composite, provenance tables, coverage, and README."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lexicon_build.tsv import write_tsv

EXPANSION_FIELDS = [
    "key",
    "dim",
    "value",
    "source",
    "confidence",
    "seeds",
    "seed_values",
    "detail",
]
CONFLICT_FIELDS = ["key", "dim", "reason", "source", "detail"]
DROPPED_FIELDS = ["source", "line", "raw_term", "canonical_key", "detail"]


def ensure_output_absent(output_dir: Path) -> None:
    path = output_dir / "composite_lexicon.json"
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    expansions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    coverage: dict[str, Any],
    dropped: list[dict[str, Any]],
    readme_lines: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "composite_lexicon.json", payload)
    _write_json(output_dir / "coverage_expansion.json", coverage)
    write_tsv(
        output_dir / "expansions.tsv",
        EXPANSION_FIELDS,
        [_expansion_row(row) for row in expansions],
    )
    write_tsv(output_dir / "conflicts_expansion.tsv", CONFLICT_FIELDS, conflicts)
    write_tsv(output_dir / "dropped_multitoken.tsv", DROPPED_FIELDS, dropped)
    (output_dir / "README.source.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8", newline="\n"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _expansion_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": row["key"],
        "dim": row["dim"],
        "value": row["value"],
        "source": row["source"],
        "confidence": f"{float(row['confidence']):.6f}",
        "seeds": ",".join(row["seeds"]),
        "seed_values": ",".join(f"{float(value):.6f}" for value in row["seed_values"]),
        "detail": row.get("detail", ""),
    }
