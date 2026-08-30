from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import heapq
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

from ..integrity import atomic_output_path, atomic_write_json, path_identity
from ..tokenizer import Tokenizer
from .sft import conversation_digest
from .sft_quality import (
    SFTQualityPolicy,
    normalize_sft_text,
    response_has_severe_repetition,
    stable_text_digest,
    target_category,
)


def _replace_string_values(
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
        return [_replace_string_values(item, replacements, counts) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_string_values(item, replacements, counts)
            for key, item in value.items()
        }
    return value


def prepare_sft_jsonl(
    source_path: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    deduplicate_exact: bool = True,
    exclude_patterns: Sequence[str] = (),
    replace_patterns: Sequence[tuple[str, str]] = (),
    quality_policy: str | Path | dict[str, object] | None = None,
    tokenizer: Tokenizer | None = None,
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
    if any(not source.strip() for source, _replacement in replace_patterns):
        raise ValueError("replacement source patterns must not be empty")
    normalized_sources = [source.casefold() for source, _replacement in replace_patterns]
    if len(set(normalized_sources)) != len(normalized_sources):
        raise ValueError("replacement source patterns must be unique ignoring case")
    for target in (output, manifest):
        if target.exists():
            raise FileExistsError(f"prepared SFT artifact already exists: {target}")

    replacements = [
        (re.compile(re.escape(source), flags=re.IGNORECASE), replacement, source)
        for source, replacement in replace_patterns
    ]

    if quality_policy is not None:
        if tokenizer is None:
            raise ValueError("quality-policy SFT preparation requires a tokenizer")
        policy = SFTQualityPolicy.load(quality_policy)
        return _prepare_quality_sft_jsonl(
            source,
            output,
            manifest,
            policy=policy,
            tokenizer=tokenizer,
            deduplicate_exact=deduplicate_exact,
            exclude_patterns=exclude_patterns,
            replacements=replacements,
            replace_patterns=replace_patterns,
        )

    seen: set[bytes] = set()
    excluded: Counter[str] = Counter()
    replacement_occurrences: Counter[str] = Counter()
    input_digest = hashlib.sha256()
    input_bytes = input_rows = written_rows = invalid_rows = duplicate_rows = excluded_rows = 0
    replaced_rows = 0
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
                row_replacements: Counter[str] = Counter()
                prepared_messages = _replace_string_values(messages, replacements, row_replacements)
                assert isinstance(prepared_messages, list)
                if row_replacements.total():
                    replaced_rows += 1
                    replacement_occurrences.update(row_replacements)
                    row["conversations"] = prepared_messages
                messages = prepared_messages
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
                if row_replacements.total():
                    serialized = json.dumps(
                        row, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    output_file.write(serialized + b"\n")
                else:
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
            "replace_patterns": [
                {"source": source, "replacement": replacement}
                for source, replacement in replace_patterns
            ],
        },
        "counts": {
            "input_rows": input_rows,
            "written_rows": written_rows,
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "excluded_rows": excluded_rows,
            "excluded_by_pattern": dict(excluded),
            "replaced_rows": replaced_rows,
            "replacement_occurrences": {
                source: replacement_occurrences[source]
                for source, _replacement in replace_patterns
            },
            "total_replacement_occurrences": replacement_occurrences.total(),
        },
        "manifest": str(manifest.resolve()),
    }
    atomic_write_json(manifest, report)
    return report


def _prepare_quality_sft_jsonl(
    source: Path,
    output: Path,
    manifest: Path,
    *,
    policy: SFTQualityPolicy,
    tokenizer: Tokenizer,
    deduplicate_exact: bool,
    exclude_patterns: Sequence[str],
    replacements: Sequence[tuple[re.Pattern[str], str, str]],
    replace_patterns: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Select deterministic high-quality assistant targets without rewriting raw data."""

    seen_conversations: set[bytes] = set()
    prompt_counts: Counter[int] = Counter()
    response_counts: Counter[int] = Counter()
    category_candidate_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    replacement_occurrences: Counter[str] = Counter()
    input_digest = hashlib.sha256()
    input_bytes = input_rows = invalid_rows = duplicate_rows = excluded_rows = replaced_rows = 0
    candidate_targets = 0
    # Negative priority makes heap[0] the worst retained candidate, so the
    # globally smallest deterministic priorities survive regardless of file order.
    retained: list[tuple[int, int, int, str]] = []

    with source.open("rb") as input_file:
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
            row_replacements: Counter[str] = Counter()
            prepared = _replace_string_values(messages, replacements, row_replacements)
            assert isinstance(prepared, list)
            messages = prepared
            if row_replacements.total():
                replaced_rows += 1
                replacement_occurrences.update(row_replacements)
            digest = conversation_digest(messages)
            if digest in seen_conversations:
                duplicate_rows += 1
                if deduplicate_exact:
                    continue
            else:
                seen_conversations.add(digest)
            searchable = json.dumps(messages, ensure_ascii=False).casefold()
            matched = [pattern for pattern in exclude_patterns if pattern.casefold() in searchable]
            if matched:
                excluded_rows += 1
                excluded.update(matched)
                continue

            row_candidates = 0
            for position, message in enumerate(messages):
                if message.get("role") != "assistant":
                    continue
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                if not (isinstance(content, str) and content.strip()) and not tool_calls:
                    rejected["empty_target"] += 1
                    continue
                if policy.reject_severe_repetition and response_has_severe_repetition(message):
                    rejected["severe_repetition"] += 1
                    continue
                response_text = str(content or "")
                if tool_calls:
                    response_text += json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
                target_tokens = len(tokenizer.encode(response_text)) + 1
                if target_tokens > policy.max_target_tokens:
                    rejected["target_too_long"] += 1
                    continue
                latest_user = next(
                    (
                        str(previous.get("content") or "")
                        for previous in reversed(messages[:position])
                        if previous.get("role") == "user"
                    ),
                    "",
                )
                prompt_digest = stable_text_digest(latest_user)
                category = target_category(messages, position)
                category_limit = policy.category_max_targets.get(category, policy.max_targets)
                if category_candidate_counts[category] >= category_limit:
                    rejected["category_cap"] += 1
                    continue
                prompt_limit = (
                    policy.max_tool_prompt_occurrences
                    if category == "tool"
                    else policy.max_prompt_occurrences
                )
                if prompt_counts[prompt_digest] >= prompt_limit:
                    rejected["prompt_cap"] += 1
                    continue
                normalized_response = normalize_sft_text(response_text)
                response_digest = stable_text_digest(response_text, code=category == "code")
                if category == "identity":
                    response_limit = policy.max_identity_response_occurrences
                elif len(normalized_response) < policy.short_response_characters:
                    response_limit = policy.max_short_response_occurrences
                else:
                    response_limit = policy.max_response_occurrences
                if response_counts[response_digest] >= response_limit:
                    rejected["response_cap"] += 1
                    continue
                if row_candidates >= policy.max_targets_per_conversation:
                    rejected["conversation_cap"] += 1
                    continue

                prompt_counts[prompt_digest] += 1
                response_counts[response_digest] += 1
                category_candidate_counts[category] += 1
                row_candidates += 1
                candidate_targets += 1
                priority_bytes = hashlib.blake2b(
                    policy.seed.to_bytes(8, "big", signed=True)
                    + digest
                    + position.to_bytes(4, "big"),
                    digest_size=8,
                ).digest()
                priority = int.from_bytes(priority_bytes, "big")
                candidate = (-priority, line_number, position, category)
                if len(retained) < policy.max_targets:
                    heapq.heappush(retained, candidate)
                elif priority < -retained[0][0]:
                    heapq.heapreplace(retained, candidate)

    selected_by_line: dict[int, list[tuple[int, str]]] = {}
    selected_categories: Counter[str] = Counter()
    for _negative_priority, line_number, position, category in retained:
        selected_by_line.setdefault(line_number, []).append((position, category))
        selected_categories[category] += 1

    written_rows = 0
    with atomic_output_path(output) as temporary:
        with source.open("rb") as input_file, temporary.open("wb") as output_file:
            for line_number, line in enumerate(input_file, 1):
                selected = selected_by_line.get(line_number)
                if not selected:
                    continue
                row = json.loads(line)
                messages = row["conversations"]
                prepared = _replace_string_values(messages, replacements, Counter())
                row["conversations"] = prepared
                selected.sort()
                row["sft_selection"] = {
                    "version": "target_positions_v1",
                    "target_positions": [position for position, _category in selected],
                    "categories": {
                        str(position): category for position, category in selected
                    },
                }
                output_file.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    + b"\n"
                )
                written_rows += 1

    report: dict[str, object] = {
        "schema_version": 2,
        "kind": "prepared_sft_quality_data",
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
            "quality": {
                name: value for name, value in asdict(policy).items()
            },
            "deduplicate_exact": deduplicate_exact,
            "exclude_patterns": list(exclude_patterns),
            "replace_patterns": [
                {"source": source_text, "replacement": replacement}
                for source_text, replacement in replace_patterns
            ],
        },
        "counts": {
            "input_rows": input_rows,
            "written_rows": written_rows,
            "invalid_rows": invalid_rows,
            "duplicate_rows": duplicate_rows,
            "excluded_rows": excluded_rows,
            "excluded_by_pattern": dict(excluded),
            "candidate_targets": candidate_targets,
            "selected_targets": len(retained),
            "selected_categories": dict(selected_categories),
            "rejected_targets": dict(rejected),
            "replaced_rows": replaced_rows,
            "replacement_occurrences": {
                source_text: replacement_occurrences[source_text]
                for source_text, _replacement in replace_patterns
            },
            "total_replacement_occurrences": replacement_occurrences.total(),
        },
        "manifest": str(manifest.resolve()),
    }
    atomic_write_json(manifest, report)
    return report
