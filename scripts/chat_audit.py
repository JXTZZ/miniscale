"""Corpus statistics for assistant-turn SFT and paired preferences."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict
import json
from pathlib import Path
import random

from dataset.preference_dataset import encode_preference_pair, parse_preference_pair, PreferenceCorpusIndex
from dataset.sft_dataset import SFTCorpusIndex, truncate_sft_example
from dataset.sft_quality import (
    repeated_character_ngram_fraction,
    repeated_sentence_coverage,
    response_has_severe_repetition,
    stable_text_digest,
    target_category,
)
from model.tokenizer import Tokenizer


def audit_sft_jsonl(
    path: str | Path,
    tokenizer: Tokenizer,
    *,
    max_length: int,
    min_context_tokens: int = 32,
    target_mode: str = "reasoning_and_response",
    validation_fraction: float = 0.005,
    sample_size: int = 5000,
    seed: int = 42,
    identity_patterns: Sequence[str] = (),
) -> dict[str, object]:
    """Fully scan SFT structure and tokenize a deterministic global sample."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if any(not pattern.strip() for pattern in identity_patterns):
        raise ValueError("identity_patterns must not contain empty strings")
    index = SFTCorpusIndex.build(
        path,
        validation_fraction=validation_fraction,
        target_mode=target_mode,
        destination="split",
        deduplicate_exact=False,
    )
    rng = random.Random(seed)
    sample: list[list[dict[str, object]]] = []
    target_examples_seen = 0
    role_counts: Counter[str] = Counter()
    identity_counts: Counter[str] = Counter()
    prompt_digests: Counter[int] = Counter()
    response_digests: Counter[int] = Counter()
    category_counts: Counter[str] = Counter()
    reasoning_messages = empty_content_messages = tool_call_messages = 0
    severe_repetition_messages = 0
    repeated_ngram_values: list[float] = []
    repeated_sentence_values: list[float] = []

    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("conversations") if isinstance(row, dict) else None
            if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
                continue
            selection = row.get("sft_selection") if isinstance(row, dict) else None
            selected_positions = (
                set(selection.get("target_positions", [])) if isinstance(selection, dict) else None
            )
            searchable = json.dumps(messages, ensure_ascii=False).casefold()
            for pattern in identity_patterns:
                if pattern.casefold() in searchable:
                    identity_counts[pattern] += 1
            for position, message in enumerate(messages):
                role = message.get("role")
                if isinstance(role, str):
                    role_counts[role] += 1
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    empty_content_messages += 1
                if message.get("tool_calls"):
                    tool_call_messages += 1
                reasoning = message.get("reasoning_content")
                if role == "assistant" and isinstance(reasoning, str) and reasoning.strip():
                    reasoning_messages += 1
                if role != "assistant":
                    continue
                if selected_positions is not None and position not in selected_positions:
                    continue
                has_response = (isinstance(content, str) and bool(content.strip())) or bool(message.get("tool_calls"))
                has_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())
                if not has_response and not (target_mode == "reasoning_and_response" and has_reasoning):
                    continue
                target_examples_seen += 1
                latest_user = next(
                    (
                        str(previous.get("content") or "")
                        for previous in reversed(messages[:position])
                        if previous.get("role") == "user"
                    ),
                    "",
                )
                prompt_digests[stable_text_digest(latest_user)] += 1
                response_text = str(content or "")
                if response_text.strip():
                    response_digests[stable_text_digest(response_text)] += 1
                    repeated_ngram_values.append(repeated_character_ngram_fraction(response_text))
                    repeated_sentence_values.append(repeated_sentence_coverage(response_text))
                    severe_repetition_messages += response_has_severe_repetition(message)
                category_counts[target_category(messages, position)] += 1
                candidate = [dict(item) for item in messages[: position + 1]]
                if len(sample) < sample_size:
                    sample.append(candidate)
                else:
                    replacement = rng.randrange(target_examples_seen)
                    if replacement < sample_size:
                        sample[replacement] = candidate

    raw_lengths: list[int] = []
    supervised_lengths: list[int] = []
    retained_supervised_lengths: list[int] = []
    truncated_examples = target_truncated_examples = 0
    dropped_context_tokens = dropped_target_tokens = 0
    for messages in sample:
        input_ids, labels = tokenizer.encode_sft(
            messages,
            target_mode=target_mode,
            target_assistant_index=-1,
        )
        raw_lengths.append(len(input_ids))
        supervised_lengths.append(sum(label != -100 for label in labels[1:]))
        encoded = truncate_sft_example(
            input_ids,
            labels,
            max_length=max_length,
            min_context_tokens=min_context_tokens,
        )
        retained_supervised_lengths.append(encoded.supervised_tokens)
        if encoded.original_tokens > max_length:
            truncated_examples += 1
        if encoded.dropped_target_tokens:
            target_truncated_examples += 1
        dropped_context_tokens += encoded.dropped_context_tokens
        dropped_target_tokens += encoded.dropped_target_tokens

    retained = sum(retained_supervised_lengths)
    original = sum(supervised_lengths)

    def duplicate_summary(counts: Counter[int]) -> dict[str, object]:
        total = counts.total()
        duplicate_instances = sum(count for count in counts.values() if count > 1)
        return {
            "unique": len(counts),
            "duplicate_groups": sum(count > 1 for count in counts.values()),
            "instances_in_duplicate_groups": duplicate_instances,
            "duplicate_group_fraction": duplicate_instances / total if total else 0.0,
            "excess_duplicates": sum(count - 1 for count in counts.values() if count > 1),
            "largest_group": max(counts.values(), default=0),
        }
    return {
        "schema_version": 1,
        "kind": "sft_data_audit",
        "data": {"path": str(Path(path).resolve()), "identity": index.identity},
        "configuration": {
            "max_length": max_length,
            "min_context_tokens": min_context_tokens,
            "target_mode": target_mode,
            "validation_fraction": validation_fraction,
            "sample_size": sample_size,
            "seed": seed,
            "identity_patterns": list(identity_patterns),
        },
        "structure": {
            **asdict(index.stats),
            "role_counts": dict(role_counts),
            "reasoning_messages": reasoning_messages,
            "empty_content_messages": empty_content_messages,
            "tool_call_messages": tool_call_messages,
            "identity_pattern_conversations": dict(identity_counts),
        },
        "token_sample": {
            "examples": len(sample),
            "raw_tokens": integer_percentiles(raw_lengths),
            "supervised_tokens": integer_percentiles(supervised_lengths),
            "retained_supervised_tokens": integer_percentiles(retained_supervised_lengths),
            "truncated_examples": truncated_examples,
            "truncated_fraction": truncated_examples / len(sample) if sample else 0.0,
            "target_truncated_examples": target_truncated_examples,
            "target_truncated_fraction": target_truncated_examples / len(sample) if sample else 0.0,
            "supervised_token_retention": retained / original if original else 0.0,
            "dropped_context_tokens": dropped_context_tokens,
            "dropped_target_tokens": dropped_target_tokens,
        },
        "quality": {
            "normalization": "unicode_nfkc_casefold_alphanumeric_v1",
            "normalized_prompt_duplicates": duplicate_summary(prompt_digests),
            "normalized_response_duplicates": duplicate_summary(response_digests),
            "category_counts": dict(category_counts),
            "severe_repetition_messages": severe_repetition_messages,
            "severe_repetition_fraction": (
                severe_repetition_messages / target_examples_seen if target_examples_seen else 0.0
            ),
            "character_4gram_repetition": integer_percentiles(
                [round(value * 1_000_000) for value in repeated_ngram_values]
            ),
            "sentence_repetition_coverage": integer_percentiles(
                [round(value * 1_000_000) for value in repeated_sentence_values]
            ),
            "fraction_scale": 1_000_000,
            "near_duplicate_check": False,
        },
    }


INVALID_REASON_MARKERS = {
    "not_an_object": "expected a JSON object",
    "invalid_conversation": "must be a conversation",
    "invalid_role": "contains invalid roles",
    "not_assistant_terminated": "must end with an assistant",
    "empty_target": "has an empty target",
    "prompt_mismatch": "must share an identical prompt",
    "identical_responses": "responses are identical",
}


def _invalid_reason(message: str) -> str:
    for reason, marker in INVALID_REASON_MARKERS.items():
        if marker in message:
            return reason
    return "other"


def audit_dpo_jsonl(
    path: str | Path,
    tokenizer: Tokenizer,
    *,
    max_length: int,
    min_context_tokens: int = 32,
    target_mode: str = "reasoning_and_response",
    validation_fraction: float = 0.05,
    sample_size: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Fully scan preference structure and tokenize a deterministic global sample."""

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    index = PreferenceCorpusIndex.build(
        path,
        validation_fraction=validation_fraction,
        target_mode=target_mode,
        destination="split",
        deduplicate_exact=False,
    )
    invalid_reasons: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    reasoning_targets = 0
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                pair = parse_preference_pair(
                    row,
                    target_mode=target_mode,
                    location=f"{source_path}:{line_number}",
                )
            except ValueError as error:
                invalid_reasons[_invalid_reason(str(error))] += 1
                continue
            for message in pair.prompt:
                role = message.get("role")
                if isinstance(role, str):
                    role_counts[role] += 1
            for response in (pair.chosen[-1], pair.rejected[-1]):
                reasoning = response.get("reasoning_content")
                reasoning_targets += int(isinstance(reasoning, str) and bool(reasoning.strip()))

    offsets = list(index.train_offsets) + list(index.validation_offsets)
    selected = random.Random(seed).sample(offsets, min(sample_size, len(offsets)))
    context_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    truncated_pairs = chosen_truncated = rejected_truncated = 0
    dropped_context = dropped_chosen = dropped_rejected = 0
    with source_path.open("rb") as source:
        for offset in selected:
            source.seek(offset)
            row = json.loads(source.readline())
            pair = parse_preference_pair(row, target_mode=target_mode, location=f"{source_path}@{offset}")
            encoded = encode_preference_pair(
                pair,
                tokenizer,
                max_length=max_length,
                min_context_tokens=min_context_tokens,
                target_mode=target_mode,
            )
            context_lengths.append(encoded.context_tokens)
            chosen_lengths.append(encoded.chosen_target_tokens)
            rejected_lengths.append(encoded.rejected_target_tokens)
            was_truncated = bool(
                encoded.dropped_context_tokens
                or encoded.dropped_chosen_tokens
                or encoded.dropped_rejected_tokens
            )
            truncated_pairs += int(was_truncated)
            chosen_truncated += int(bool(encoded.dropped_chosen_tokens))
            rejected_truncated += int(bool(encoded.dropped_rejected_tokens))
            dropped_context += encoded.dropped_context_tokens
            dropped_chosen += encoded.dropped_chosen_tokens
            dropped_rejected += encoded.dropped_rejected_tokens

    sample_count = len(selected)
    return {
        "schema_version": 1,
        "kind": "dpo_data_audit",
        "data": {"path": str(source_path.resolve()), "identity": index.identity},
        "configuration": {
            "max_length": max_length,
            "min_context_tokens": min_context_tokens,
            "target_mode": target_mode,
            "validation_fraction": validation_fraction,
            "sample_size": sample_size,
            "seed": seed,
        },
        "structure": {
            **asdict(index.stats),
            "invalid_reasons": dict(invalid_reasons),
            "prompt_role_counts": dict(role_counts),
            "reasoning_targets": reasoning_targets,
        },
        "token_sample": {
            "pairs": sample_count,
            "context_tokens": integer_percentiles(context_lengths),
            "chosen_target_tokens": integer_percentiles(chosen_lengths),
            "rejected_target_tokens": integer_percentiles(rejected_lengths),
            "truncated_pairs": truncated_pairs,
            "truncated_fraction": truncated_pairs / sample_count if sample_count else 0.0,
            "chosen_truncated_pairs": chosen_truncated,
            "rejected_truncated_pairs": rejected_truncated,
            "dropped_context_tokens": dropped_context,
            "dropped_chosen_tokens": dropped_chosen,
            "dropped_rejected_tokens": dropped_rejected,
        },
    }


def integer_percentiles(
    values: Iterable[int],
    percentiles: Sequence[int] = (50, 75, 90, 95, 99),
) -> dict[str, int]:
    """Return nearest-index percentiles without changing historical rounding.

    NumPy and ``statistics.quantiles`` use interpolation conventions that can
    produce different answers for small audit samples. This project needs
    stable integer results for reproducible reports, so the established
    nearest-index rule remains explicit here.
    """

    ordered = sorted(values)
    if not ordered:
        return {}
    result = {
        f"p{percentile}": ordered[round((len(ordered) - 1) * percentile / 100)]
        for percentile in percentiles
    }
    result["max"] = ordered[-1]
    return result
