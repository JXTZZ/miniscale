from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..integrity import atomic_output_path, atomic_write_json, path_identity
from .sft import conversation_digest


def prepare_sft_jsonl(
    source_path: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    deduplicate_exact: bool = True,
    exclude_patterns: Sequence[str] = (),
) -> dict[str, object]:
    """Create an immutable, auditable derived SFT JSONL without touching raw data."""

    source = Path(source_path)
    output = Path(output_path)
    manifest = Path(manifest_path) if manifest_path is not None else output.with_suffix(
        output.suffix + ".manifest.json"
    )
    if not source.is_file():
        raise FileNotFoundError(f"SFT source data does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("prepared SFT output must not overwrite its source")
    if output.resolve() == manifest.resolve():
        raise ValueError("prepared SFT output and manifest must use different paths")
    if any(not pattern.strip() for pattern in exclude_patterns):
        raise ValueError("exclude_patterns must not contain empty strings")
    for target in (output, manifest):
        if target.exists():
            raise FileExistsError(f"prepared SFT artifact already exists: {target}")

    seen: set[bytes] = set()
    excluded: Counter[str] = Counter()
    input_digest = hashlib.sha256()
    input_bytes = input_rows = written_rows = invalid_rows = duplicate_rows = excluded_rows = 0
    with atomic_output_path(output) as temporary:
        with source.open("rb") as input_file, temporary.open("wb") as output_file:
            for line_number, line in enumerate(input_file, 1):
                input_digest.update(line)
                input_bytes += len(line)
                if not line.strip():
                    continue
                input_rows += 1
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError(f"invalid JSON at {source}:{line_number}") from error
                messages = row.get("conversations") if isinstance(row, dict) else None
                if not isinstance(messages, list) or not messages or not all(
                    isinstance(message, dict) for message in messages
                ):
                    invalid_rows += 1
                    continue
                digest = conversation_digest(messages)
                if digest in seen:
                    duplicate_rows += 1
                    if deduplicate_exact:
                        continue
                else:
                    seen.add(digest)
                searchable = json.dumps(messages, ensure_ascii=False).casefold()
                matched = [pattern for pattern in exclude_patterns if pattern.casefold() in searchable]
                if matched:
                    excluded_rows += 1
                    excluded.update(matched)
                    continue
                output_file.write(line if line.endswith(b"\n") else line + b"\n")
                written_rows += 1

    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "prepared_sft_data",
        "source": {
            "path": str(source.resolve()),
            "identity": {
                "kind": "file",
                "sha256": input_digest.hexdigest(),
                "size_bytes": input_bytes,
            },
        },
        "output": {"path": str(output.resolve()), "identity": path_identity(output)},
        "policy": {
            "deduplicate_exact": deduplicate_exact,
            "exclude_patterns": list(exclude_patterns),
        },
        "counts": {
            "input_rows": input_rows,
            "written_rows": written_rows,
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "excluded_rows": excluded_rows,
            "excluded_by_pattern": dict(excluded),
        },
        "manifest": str(manifest.resolve()),
    }
    atomic_write_json(manifest, report)
    return report
