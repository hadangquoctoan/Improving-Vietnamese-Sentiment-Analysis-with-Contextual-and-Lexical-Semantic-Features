"""Deterministic stratified train/dev split for VLSP2016."""

import random


def stratified_split(
    samples, dev_ratio: float = 0.1, seed: int = 42
) -> tuple[list, list]:
    if not samples:
        raise ValueError("samples must not be empty")
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between 0 and 1")

    groups: dict[int, list] = {}
    for sample in samples:
        groups.setdefault(sample["label"], []).append(sample)

    rng = random.Random(seed)
    fit, dev = [], []
    for label in sorted(groups):
        rows = list(groups[label])
        rng.shuffle(rows)
        dev_count = max(1, round(len(rows) * dev_ratio))
        dev.extend(rows[:dev_count])
        fit.extend(rows[dev_count:])

    rng.shuffle(fit)
    rng.shuffle(dev)
    return fit, dev
