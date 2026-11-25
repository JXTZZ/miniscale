"""Shared input scan and provenance for SFT data preparation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from typing import Sequence

from dataset.file_utils import path_identity
from dataset.sft_dataset import conversation_digest


def replace_string_values(
    value: object,
    replacements: Sequence[tuple[re.Pattern[str], str, str]],
    counts: Counter[str],
) -> object:
    """Recursively replace text values while preserving JSON structure and keys."""

    if isinstance(value, str):
        replaced = value
        for pattern, replacement, source in replacements:
            replaced, occurrences = pattern.subn(lambda _match: replacement, replaced)
            counts[source] += occurrences
        return replaced
    if isinstance(value, list):
        return [replace_string_values(item, replacements, counts) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_string_values(item, replacements, counts)
            for key, item in value.items()
        }
    return value


class SFTSource:
    """Scan, rewrite, and filter either form of derived SFT data."""

    def __init__(self, path, deduplicate_exact, exclude_patterns, replacements):
        self.path = path
        self.deduplicate_exact = deduplicate_exact
        self.exclude_patterns = exclude_patterns
        self.replacements = replacements
        self.digest = hashlib.sha256()
        self.bytes = self.input_rows = self.invalid_rows = 0
        self.duplicate_rows = self.excluded_rows = self.replaced_rows = 0
        self.excluded = Counter()
        self.replacement_occurrences = Counter()

    def rows(self):
        seen = set()
        with self.path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                self.digest.update(line)
                self.bytes += len(line)
                if not line.strip():
                    continue
                self.input_rows += 1
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError(f"invalid JSON at {self.path}:{number}") from error
                messages = row.get("conversations") if isinstance(row, dict) else None
                if not isinstance(messages, list) or not messages or not all(isinstance(msg, dict) for msg in messages):
                    self.invalid_rows += 1
                    continue
                counts = Counter()
                messages = replace_string_values(messages, self.replacements, counts)
                row["conversations"] = messages
                replaced = bool(counts.total())
                if replaced:
                    self.replaced_rows += 1
                    self.replacement_occurrences.update(counts)
                digest = conversation_digest(messages)
                if digest in seen:
                    self.duplicate_rows += 1
                    if self.deduplicate_exact:
                        continue
                else:
                    seen.add(digest)
                searchable = json.dumps(messages, ensure_ascii=False).casefold()
                matched = [pattern for pattern in self.exclude_patterns if pattern.casefold() in searchable]
                if matched:
                    self.excluded_rows += 1
                    self.excluded.update(matched)
                    continue
                yield number, line, row, digest, replaced

    def report(self, output, manifest, version, kind, written_rows):
        return {
            "schema_version": version, "kind": kind,
            "source": {
                "path": str(self.path.resolve()),
                "identity": {"kind": "file", "sha256": self.digest.hexdigest(), "size_bytes": self.bytes},
            },
            "output": {"path": str(output.resolve()), "identity": path_identity(output)},
            "policy": {
                "deduplicate_exact": self.deduplicate_exact,
                "exclude_patterns": list(self.exclude_patterns),
                "replace_patterns": [
                    {"source": source, "replacement": replacement}
                    for _, replacement, source in self.replacements
                ],
            },
            "counts": {
                "input_rows": self.input_rows, "written_rows": written_rows, "invalid_rows": self.invalid_rows,
                "duplicate_rows": self.duplicate_rows, "excluded_rows": self.excluded_rows,
                "excluded_by_pattern": dict(self.excluded), "replaced_rows": self.replaced_rows,
                "replacement_occurrences": {
                    source: self.replacement_occurrences[source] for _, _, source in self.replacements
                },
                "total_replacement_occurrences": self.replacement_occurrences.total(),
            },
            "manifest": str(manifest.resolve()),
        }
