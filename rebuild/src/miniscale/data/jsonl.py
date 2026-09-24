"""Common JSONL readers used by training data pipelines."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import json


def iter_jsonl(path: str | Path) -> Iterator[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            yield value


def load_jsonl_rows(path: str | Path, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in iter_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows
