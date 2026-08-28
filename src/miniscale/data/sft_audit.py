from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Sequence

from ..integrity import atomic_write_json
from ..tokenizer import Tokenizer
from .metrics import integer_percentiles
from .sft import SFTCorpusIndex, truncate_sft_example


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
    reasoning_messages = empty_content_messages = tool_call_messages = 0

    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("conversations") if isinstance(row, dict) else None
            if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
                continue
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
                has_response = (isinstance(content, str) and bool(content.strip())) or bool(message.get("tool_calls"))
                has_reasoning = isinstance(reasoning, str) and bool(reasoning.strip())
                if not has_response and not (target_mode == "reasoning_and_response" and has_reasoning):
                    continue
                target_examples_seen += 1
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
    }


def save_sft_data_audit(report: dict[str, object], path: str | Path) -> Path:
    return atomic_write_json(path, report)
