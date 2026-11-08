"""Load the user-provided AIVIVN 2019 splits with manifest SHA verification."""

import hashlib
import json
from pathlib import Path

_SPLITS = ("train", "dev", "test")
_DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "aivivn2019"
)


def load_aivivn2019(
    split: str, allow_test: bool = False
) -> list[dict[str, object]]:
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
    if split == "test" and not allow_test:
        raise ValueError(
            "AIVIVN test split is locked; pass allow_test=True only "
            "for the one-shot benchmark"
        )

    manifest = json.loads(
        (_DATA_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    expected = manifest["splits"][split]["output_sha256"]
    path = _DATA_DIR / f"{split}.jsonl"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{path.name} SHA-256 mismatch: expected {expected}, got {actual}"
        )

    with path.open(encoding="utf-8") as handle:
        return [
            _parse_row(line, line_number)
            for line_number, line in enumerate(handle, start=1)
        ]


def _parse_row(line: str, line_number: int) -> dict[str, object]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON on line {line_number}") from error
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number} must be a JSON object")

    text = row.get("text")
    label = row.get("label")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"line {line_number} has an empty text")
    if (
        isinstance(label, bool)
        or not isinstance(label, int)
        or label not in (0, 1)
    ):
        raise ValueError(f"line {line_number} has an invalid label: {label!r}")
    return {"text": text, "label": label}
