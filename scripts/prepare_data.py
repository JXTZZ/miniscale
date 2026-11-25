"""Prepare and filter supervised fine-tuning data."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
import re
import sys
from typing import Sequence

# Support running this file directly as well as with python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import atomic_output_path, atomic_write_json
from dataset.sft_dataset import DEFAULT_SFT_DATA
from dataset.sft_quality import (
    normalize_sft_text,
    response_has_severe_repetition,
    SFTQualityPolicy,
    stable_text_digest,
    target_category,
)
from model.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer, Tokenizer
from scripts.sft_source import replace_string_values, SFTSource


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

    corpus = SFTSource(source, deduplicate_exact, exclude_patterns, replacements)
    if quality_policy is not None:
        if tokenizer is None:
            raise ValueError("quality-policy SFT preparation requires a tokenizer")
        policy = SFTQualityPolicy.load(quality_policy)
        return _prepare_quality_sft_jsonl(corpus, output, manifest, policy, tokenizer)

    written_rows = 0
    with atomic_output_path(output) as temporary, temporary.open("wb") as stream:
        for _, line, row, _, replaced in corpus.rows():
            if replaced:
                line = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            stream.write(line if line.endswith(b"\n") else line + b"\n")
            written_rows += 1

    report = corpus.report(output, manifest, 1, "prepared_sft_data", written_rows)
    atomic_write_json(manifest, report)
    return report


def _prepare_quality_sft_jsonl(
    corpus: SFTSource, output: Path, manifest: Path, policy: SFTQualityPolicy, tokenizer: Tokenizer,
) -> dict[str, object]:
    """Select deterministic high-quality assistant targets without rewriting raw data."""

    prompt_counts: Counter[int] = Counter()
    response_counts: Counter[int] = Counter()
    category_candidate_counts: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    candidate_targets = 0
    # Negate the priority so the heap exposes the worst retained candidate.
    retained: list[tuple[int, int, int, str]] = []

    for line_number, _, row, digest, _ in corpus.rows():
        messages = row["conversations"]
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
        with corpus.path.open("rb") as input_file, temporary.open("wb") as output_file:
            for line_number, line in enumerate(input_file, 1):
                selected = selected_by_line.get(line_number)
                if not selected:
                    continue
                row = json.loads(line)
                messages = row["conversations"]
                prepared = replace_string_values(messages, corpus.replacements, Counter())
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

    report = corpus.report(output, manifest, 2, "prepared_sft_quality_data", written_rows)
    report["policy"]["quality"] = asdict(policy)
    report["counts"].update({
        "candidate_targets": candidate_targets, "selected_targets": len(retained),
        "selected_categories": dict(selected_categories), "rejected_targets": dict(rejected),
    })
    atomic_write_json(manifest, report)
    return report


def _text_replacement(value: str) -> tuple[str, str]:
    source, separator, replacement = value.partition("=")
    if not separator or not source.strip():
        raise argparse.ArgumentTypeError("replacement must use non-empty OLD=NEW syntax")
    return source, replacement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='write a deduplicated and optionally filtered derived SFT JSONL')
    parser.add_argument("--data", type=Path, default=DEFAULT_SFT_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--deduplicate-exact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="case-insensitive identity/brand substring to exclude; repeat for multiple patterns",
    )
    parser.add_argument(
        "--replace-pattern",
        action="append",
        default=[],
        type=_text_replacement,
        metavar="OLD=NEW",
        help="case-insensitive conversation text replacement; repeat for multiple rules",
    )
    parser.add_argument(
        "--quality-policy",
        type=Path,
        help="JSON policy for deterministic target filtering, caps, and global selection",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    quality_tokenizer = (
        load_tokenizer(arguments.tokenizer) if arguments.quality_policy is not None else None
    )
    result = prepare_sft_jsonl(
        arguments.data,
        arguments.output,
        manifest_path=arguments.manifest,
        deduplicate_exact=arguments.deduplicate_exact,
        exclude_patterns=arguments.exclude_pattern,
        replace_patterns=arguments.replace_pattern,
        quality_policy=arguments.quality_policy,
        tokenizer=quality_tokenizer,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
