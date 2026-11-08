#!/usr/bin/env python3
"""Build composite v0.3.x by expanding v0.2.0 with WordNet and fastText."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for candidate in (ROOT / "src", SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lexicon_build.corpus import load_corpus_token_counts  # noqa: E402
from lexicon_build.coverage import compute_split_coverage  # noqa: E402
from lexicon_build.expansion.canonical import build_canonicalizer  # noqa: E402
from lexicon_build.expansion.config import (  # noqa: E402
    ALLOWED_FILES,
    BASE_CONFLICTS_PATH,
    BASE_PATH,
    COVERAGE_FILES,
    DEFAULT_CACHE_PATH,
    DEFAULT_OUTPUT_DIR,
    DEV_PATH,
    FASTTEXT_DEFAULTS,
    FASTTEXT_PATH,
    FEATURES,
    HARD_STOPWORDS,
    PARENT_VERSION,
    SEED_MIN_ABS,
    TRAIN_PATH,
    VERSION,
    WORDNET_ROOT,
)
from lexicon_build.expansion.coverage import subset_coverage  # noqa: E402
from lexicon_build.expansion.fasttext import (  # noqa: E402
    VectorStore,
    propagate_fasttext,
    syllables,
)
from lexicon_build.expansion.merge import merge_entries  # noqa: E402
from lexicon_build.expansion.outputs import ensure_output_absent, write_outputs  # noqa: E402
from lexicon_build.expansion.seeds import (  # noqa: E402
    load_audited_conflicts,
    load_seeds,
    load_stoplist,
)
from lexicon_build.expansion.synonym import propagate_synonyms  # noqa: E402
from lexicon_build.expansion.validate import (  # noqa: E402
    assert_no_empty_entries,
    assert_non_destructive,
    assert_provenance_complete,
)
from lexicon_build.expansion.wordnet import build_key_file_map, load_groups  # noqa: E402
from lexicon_build.validation import validate_composite  # noqa: E402
from vnsentiment.preprocessing.segmenter import VnCoreNLPSegmenter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    output_dir = Path(args.output_dir)
    ensure_output_absent(output_dir)

    base_payload = json.loads(Path(args.base).read_text(encoding="utf-8"))
    base_entries = base_payload["entries"]
    canonicalize = build_canonicalizer(
        VnCoreNLPSegmenter(), Path(args.cache) if args.cache else None
    )

    print("loading WordNet groups...", flush=True)
    wordnet_root = Path(args.wordnet_root)
    groups, dropped = load_groups(wordnet_root, ALLOWED_FILES, canonicalize)
    allowed_map = build_key_file_map(wordnet_root, ALLOWED_FILES, canonicalize)
    coverage_map = build_key_file_map(wordnet_root, COVERAGE_FILES, canonicalize)
    seeds = load_seeds(base_entries, SEED_MIN_ABS)
    audited = load_audited_conflicts(BASE_CONFLICTS_PATH)
    stoplist = load_stoplist(wordnet_root, canonicalize)
    print(
        f"groups={len(groups)} dropped_terms={len(dropped)} seeds={len(seeds)} "
        f"audited={len(audited)} stoplist={len(stoplist)}",
        flush=True,
    )

    expanded, logs = propagate_synonyms(
        groups,
        seeds,
        base_entries,
        audited,
        stoplist,
        fill_conflicts=args.fill_conflicts,
    )
    fasttext_stats: dict[str, object] = {"enabled": False}
    if args.with_fasttext:
        targets = _fasttext_targets(
            allowed_map, base_entries, audited, stoplist, args.fill_conflicts
        )
        needed = _needed_vectors(seeds, targets)
        print(
            f"streaming fastText: targets={len(targets)} needed_vectors={len(needed)}",
            flush=True,
        )
        store = VectorStore.stream(Path(args.fasttext_path), needed)
        nn_expanded, nn_logs = propagate_fasttext(
            store, targets, seeds, base_entries=base_entries, **FASTTEXT_DEFAULTS
        )
        expanded += nn_expanded
        logs += nn_logs
        fasttext_stats = {
            "enabled": True,
            "targets": len(targets),
            "vectors_kept": len(store),
            "expanded": len(nn_expanded),
        }
        print(f"fastText expanded={len(nn_expanded)}", flush=True)

    merged, merge_logs = merge_entries(
        base_entries, expanded, audited, fill_conflicts=args.fill_conflicts
    )
    accepted = [row for row in merge_logs if row["decision"] == "accepted"]
    rejected = [row for row in merge_logs if row["decision"] == "rejected"]
    assert_non_destructive(base_entries, merged)
    assert_provenance_complete(base_entries, merged, accepted)
    assert_no_empty_entries(merged)

    train_raw, train_word = load_corpus_token_counts(TRAIN_PATH)
    dev_raw, dev_word = load_corpus_token_counts(DEV_PATH)
    coverage = _coverage(
        train_raw, train_word, dev_raw, dev_word, base_entries, merged, coverage_map
    )

    payload = _payload(
        args, base_payload, merged, seeds, accepted, logs + rejected, coverage,
        fasttext_stats, len(groups), len(dropped),
    )
    validate_composite(payload)
    write_outputs(
        output_dir,
        payload,
        expansions=sorted(accepted, key=lambda row: row["key"]),
        conflicts=_sorted_conflicts(logs + rejected),
        coverage=coverage,
        dropped=dropped,
        readme_lines=_readme(args, payload, coverage),
    )
    canonicalize.flush()
    _print_summary(output_dir, payload, coverage)
    return 0


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(BASE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--wordnet-root", default=str(WORDNET_ROOT))
    parser.add_argument("--fasttext-path", default=str(FASTTEXT_PATH))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--with-fasttext", action="store_true")
    parser.add_argument("--fill-conflicts", action="store_true")
    return parser.parse_args(argv)


def _fasttext_targets(
    allowed_map: dict[str, tuple[str, ...]],
    base_entries: dict[str, dict[str, list]],
    audited: set[str],
    stoplist: set[str],
    fill_conflicts: bool,
) -> list[str]:
    targets: list[str] = []
    for key in sorted(allowed_map):
        entry = base_entries.get(key)
        if entry is not None and entry["mask"][0] == 1:
            continue
        if key in stoplist:
            continue
        if key in audited and not fill_conflicts:
            continue
        targets.append(key)
    return targets


def _needed_vectors(seeds: dict[str, float], targets: list[str]) -> set[str]:
    base = set(seeds) | set(targets)
    needed = set(base)
    for key in base:
        needed.update(syllables(key))
    return needed


def _coverage(train_raw, train_word, dev_raw, dev_word, before, after, coverage_map):
    result = {}
    for split, raw, word in (
        ("train", train_raw, train_word),
        ("dev", dev_raw, dev_word),
    ):
        result[split] = {
            "parent": compute_split_coverage(raw, word, before),
            "expanded": compute_split_coverage(raw, word, after),
            "sentiment_eligible": {
                "parent": subset_coverage(word, before, set(coverage_map)),
                "expanded": subset_coverage(word, after, set(coverage_map)),
            },
        }
    return result


def _payload(
    args, base_payload, merged, seeds, accepted, conflicts, coverage,
    fasttext_stats, group_count, dropped_count,
):
    return {
        "source": {
            "name": "vnsentiment_composite_12d",
            "version": args.version,
            "parent_version": PARENT_VERSION,
            "scale_contract": base_payload["source"].get("scale_contract", {}),
            "canonicalization": base_payload["source"].get("canonicalization", ""),
            "expansion": {
                "sources": [
                    "vietnamese-wordnet@6ef96b386955c96c885657a84505f164babc9c22",
                    "fasttext-cc.vi.300" if fasttext_stats["enabled"] else "wordnet-only",
                ],
                "policy": {
                    "synonym_files": list(ALLOWED_FILES),
                    "coverage_files": list(COVERAGE_FILES),
                    "seed_min_abs_polarity": SEED_MIN_ABS,
                    "require_sign_consistency": True,
                    "fill_audited_conflicts": bool(args.fill_conflicts),
                    "hard_stopwords": list(HARD_STOPWORDS),
                    "fasttext": dict(FASTTEXT_DEFAULTS),
                },
                "seed_count": len(seeds),
                "wordnet_groups": group_count,
                "dropped_terms": dropped_count,
                "fasttext": fasttext_stats,
            },
        },
        "features": list(FEATURES),
        "stats": {
            "total_entries": len(merged),
            "parent_entries": len(base_payload["entries"]),
            "active_entries_per_dimension": _active_per_dimension(merged),
            "seed_count": len(seeds),
            "accepted_expansions": len(accepted),
            "rejected_candidates": len(conflicts),
            "coverage": coverage,
        },
        "entries": merged,
    }


def _active_per_dimension(entries):
    return {
        feature: sum(1 for item in entries.values() if item["mask"][index] == 1)
        for index, feature in enumerate(FEATURES)
    }


def _sorted_conflicts(rows):
    return sorted(
        rows,
        key=lambda row: (row.get("key", ""), row.get("dim", 0), row.get("reason", ""), row.get("source", "")),
    )


def _readme(args, payload, coverage):
    train_before = coverage["train"]["parent"]["word_token_coverage"]
    train_after = coverage["train"]["expanded"]["word_token_coverage"]
    dev_before = coverage["dev"]["parent"]["word_token_coverage"]
    dev_after = coverage["dev"]["expanded"]["word_token_coverage"]
    eligible_before = coverage["train"]["sentiment_eligible"]["parent"]
    eligible_after = coverage["train"]["sentiment_eligible"]["expanded"]
    expansion = payload["source"]["expansion"]
    return [
        f"# Composite 12D Vietnamese Lexicon — `v{args.version}`",
        "",
        "## 1. Nguồn",
        f"- Parent: `v{PARENT_VERSION}` (`composite_lexicon.json`).",
        "- WordNet: `data/lexicons/vietnamese-wordnet/extracted` (community snapshot).",
        f"- fastText: {'`cc.vi.300.vec`' if args.with_fasttext else 'không dùng'}.",
        "",
        "## 2. Chính sách mở rộng",
        f"- Files synonym: {', '.join(expansion['policy']['synonym_files'])}.",
        f"- Seed `|p| >= {SEED_MIN_ABS}`, bắt buộc đồng dấu trong nhóm.",
        f"- `fill_audited_conflicts = {expansion['policy']['fill_audited_conflicts']}`.",
        f"- Seed count: {expansion['seed_count']}; WordNet groups: {expansion['wordnet_groups']}.",
        f"- fastText: {expansion['fasttext']}.",
        "",
        "## 3. Coverage (word-token)",
        f"- train: {train_before['token_coverage_pct']:.2f}% -> {train_after['token_coverage_pct']:.2f}%.",
        f"- dev: {dev_before['token_coverage_pct']:.2f}% -> {dev_after['token_coverage_pct']:.2f}%.",
        f"- sentiment-eligible train: {eligible_before['token_coverage_pct']:.2f}% -> {eligible_after['token_coverage_pct']:.2f}%.",
        "",
        "## 4. Cảnh báo",
        "- Đây là mở rộng tự động; precision chưa được audit đầy đủ.",
        "- Không dùng `data/processed/test.jsonl` trong build này.",
    ]


def _print_summary(output_dir, payload, coverage):
    stats = payload["stats"]
    print(f"Generated composite v{payload['source']['version']} at: {output_dir}")
    print(f"Entries: {stats['parent_entries']} -> {stats['total_entries']}")
    print(f"Accepted expansions: {stats['accepted_expansions']}")
    for split in ("train", "dev"):
        before = coverage[split]["parent"]["word_token_coverage"]
        after = coverage[split]["expanded"]["word_token_coverage"]
        print(
            f"{split} word-token coverage: {before['token_coverage_pct']:.2f}% -> "
            f"{after['token_coverage_pct']:.2f}%"
        )


if __name__ == "__main__":
    raise SystemExit(main())
