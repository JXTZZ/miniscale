from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random

from .integrity import atomic_write_json
from .preference_data import (
    PreferenceCorpusIndex,
    encode_preference_pair,
    parse_preference_pair,
)
from .tokenizer import Tokenizer


INVALID_REASON_MARKERS = {
    "not_an_object": "expected a JSON object",
    "invalid_conversation": "must be a conversation",
    "invalid_role": "contains invalid roles",
    "not_assistant_terminated": "must end with an assistant",
    "empty_target": "has an empty target",
    "prompt_mismatch": "must share an identical prompt",
    "identical_responses": "responses are identical",
}


def _percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    result = {
        f"p{percentile}": ordered[round((len(ordered) - 1) * percentile / 100)]
        for percentile in (50, 75, 90, 95, 99)
    }
    result["max"] = ordered[-1]
    return result


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
            "context_tokens": _percentiles(context_lengths),
            "chosen_target_tokens": _percentiles(chosen_lengths),
            "rejected_target_tokens": _percentiles(rejected_lengths),
            "truncated_pairs": truncated_pairs,
            "truncated_fraction": truncated_pairs / sample_count if sample_count else 0.0,
            "chosen_truncated_pairs": chosen_truncated,
            "rejected_truncated_pairs": rejected_truncated,
            "dropped_context_tokens": dropped_context,
            "dropped_chosen_tokens": dropped_chosen,
            "dropped_rejected_tokens": dropped_rejected,
        },
    }


def save_dpo_data_audit(report: dict[str, object], path: str | Path) -> Path:
    return atomic_write_json(path, report)
