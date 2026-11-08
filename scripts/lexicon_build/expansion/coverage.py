"""Coverage restricted to a sentiment-eligible key subset."""

from __future__ import annotations

from collections import Counter
from typing import Any


def subset_coverage(
    word_counts: Counter[str] | dict[str, int],
    entries: dict[str, dict[str, list]],
    subset_keys: set[str],
) -> dict[str, Any]:
    counts = {key: count for key, count in word_counts.items() if key in subset_keys}
    total_types = len(counts)
    total_tokens = sum(counts.values())
    covered_types = sum(1 for key in counts if key in entries)
    covered_tokens = sum(count for key, count in counts.items() if key in entries)
    denominator_types = total_types or 1
    denominator_tokens = total_tokens or 1
    return {
        "types_covered": covered_types,
        "types_total": total_types,
        "type_coverage_pct": round(covered_types / denominator_types * 100, 2),
        "tokens_covered": covered_tokens,
        "tokens_total": total_tokens,
        "token_coverage_pct": round(covered_tokens / denominator_tokens * 100, 2),
    }
