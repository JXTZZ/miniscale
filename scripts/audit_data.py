"""Audit the JSONL inputs for the five training stages."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

# Direct file execution starts with this subdirectory on sys.path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import atomic_write_json, path_identity, tokenizer_identity
from dataset.lm_dataset import (
    DEFAULT_PRETRAIN_DATA,
    DEFAULT_SFT_DATA,
    SFTCorpusIndex,
    is_validation_text,
    truncate_sft_example,
)
from dataset.preference_dataset import (
    DEFAULT_DPO_DATA,
    PreferenceCorpusIndex,
    encode_preference_pair,
    parse_preference_pair,
)
from dataset.rl_dataset import DEFAULT_RL_DATA, build_agent_corpus, build_rl_corpus
from dataset.sft_quality import (
    repeated_character_ngram_fraction,
    repeated_sentence_coverage,
    response_has_severe_repetition,
    stable_text_digest,
    target_category,
)
from model.tokenizer import DEFAULT_TOKENIZER_PATH, Tokenizer, load_tokenizer
from trainer.rl_utils import AgentRLOptions, GRPOOptions
from trainer.train_dpo import DPOOptions
from trainer.train_full_sft import SFTOptions
from trainer.train_pretrain import PretrainOptions
from trainer.trainer_utils import option_default


def audit_rl_jsonl(
    path: str | Path,
    *,
    validation_fraction: float = 0.05,
    sample_size: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    corpus = build_rl_corpus(path, validation_fraction=validation_fraction, seed=seed)
    all_tasks = list(corpus.train) + list(corpus.validation)
    selected = random.Random(seed + 1).sample(all_tasks, min(sample_size, len(all_tasks)))
    prompt_lengths = [len(task.prompt) for task in selected]
    prompt_percentiles = integer_percentiles(prompt_lengths, (50, 90, 99))

    return {
        "schema_version": 1,
        "kind": "grpo_data_audit",
        "data": {"path": str(Path(path).resolve()), "identity": corpus.identity},
        "configuration": {
            "validation_fraction": validation_fraction,
            "sample_size": sample_size,
            "seed": seed,
        },
        "structure": {**asdict(corpus.stats), "invalid_reasons": corpus.invalid_reasons},
        "sample": {
            "tasks": len(selected),
            "prompt_characters": {
                "p50": prompt_percentiles.get("p50", 0),
                "p90": prompt_percentiles.get("p90", 0),
                "p99": prompt_percentiles.get("p99", 0),
                "max": prompt_percentiles.get("max", 0),
            },
        },
    }


def audit_agent_jsonl(
    path: str | Path,
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    corpus = build_agent_corpus(path, validation_fraction=validation_fraction, seed=seed)
    return {
        "schema_version": 1,
        "kind": "agent_rl_data_audit",
        "data": {"path": str(Path(path).resolve()), "identity": corpus.identity},
        "configuration": {"validation_fraction": validation_fraction, "seed": seed},
        "structure": {
            **asdict(corpus.stats),
            "invalid_reasons": corpus.invalid_reasons,
            "unsupported_tools": corpus.unsupported_tools,
        },
    }


def _packed_statistics(stream_tokens: int, sequence_length: int) -> dict[str, int]:
    blocks = (stream_tokens - 1) // (sequence_length - 1) if stream_tokens >= 2 else 0
    return {
        "stream_tokens": stream_tokens,
        "packed_blocks": blocks,
        "loader_input_tokens": blocks * sequence_length,
        "next_token_targets": blocks * (sequence_length - 1),
        "discarded_tail_tokens": max(stream_tokens - (blocks * (sequence_length - 1) + 1), 0),
    }


def audit_pretrain_jsonl(
    path: str | Path,
    tokenizer: Tokenizer,
    *,
    validation_fraction: float = 0.005,
    sequence_length: int = 768,
    tokenizer_batch_size: int = 4096,
) -> dict[str, Any]:
    """Fully scan pretraining JSONL and return a reproducible quality report."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if sequence_length < 2 or tokenizer_batch_size < 1:
        raise ValueError("sequence_length must be at least 2 and tokenizer_batch_size must be positive")

    source_path = Path(path)
    counts = {
        "rows": 0,
        "blank_lines": 0,
        "invalid_json_rows": 0,
        "non_object_rows": 0,
        "missing_or_empty_text_rows": 0,
        "valid_documents": 0,
        "whitespace_only_documents": 0,
        "exact_duplicate_rows": 0,
        "train_documents": 0,
        "validation_documents": 0,
    }
    total_characters = 0
    min_characters: int | None = None
    max_characters = 0
    train_tokens = 0
    validation_tokens = 0
    min_tokens: int | None = None
    max_tokens = 0
    seen_texts: set[bytes] = set()
    pending: list[tuple[str, bool]] = []

    def flush() -> None:
        nonlocal train_tokens, validation_tokens, min_tokens, max_tokens
        if not pending:
            return
        encoded = tokenizer.encode_batch([text for text, _ in pending], bos=True, eos=True)
        if len(encoded) != len(pending):
            raise RuntimeError("tokenizer returned the wrong number of batch encodings")
        for (_, is_validation), token_ids in zip(pending, encoded, strict=True):
            token_count = len(token_ids)
            min_tokens = token_count if min_tokens is None else min(min_tokens, token_count)
            max_tokens = max(max_tokens, token_count)
            if is_validation:
                validation_tokens += token_count
            else:
                train_tokens += token_count
        pending.clear()

    with source_path.open(encoding="utf-8") as source:
        for line in source:
            counts["rows"] += 1
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["invalid_json_rows"] += 1
                continue
            if not isinstance(row, dict):
                counts["non_object_rows"] += 1
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text:
                counts["missing_or_empty_text_rows"] += 1
                continue

            counts["valid_documents"] += 1
            counts["whitespace_only_documents"] += int(not text.strip())
            characters = len(text)
            total_characters += characters
            min_characters = characters if min_characters is None else min(min_characters, characters)
            max_characters = max(max_characters, characters)
            text_hash = hashlib.blake2b(text.encode(), digest_size=16).digest()
            if text_hash in seen_texts:
                counts["exact_duplicate_rows"] += 1
            else:
                seen_texts.add(text_hash)

            validation = is_validation_text(text, validation_fraction)
            counts["validation_documents" if validation else "train_documents"] += 1
            pending.append((text, validation))
            if len(pending) >= tokenizer_batch_size:
                flush()
    flush()

    valid_documents = counts["valid_documents"]
    total_tokens = train_tokens + validation_tokens
    return {
        "schema_version": 1,
        "data": {"path": str(source_path.resolve()), "identity": path_identity(source_path)},
        "tokenizer": tokenizer_identity(tokenizer),
        "split": {"method": "blake2b_content_hash", "validation_fraction": validation_fraction},
        "sequence_length": sequence_length,
        "counts": counts,
        "characters": {
            "total": total_characters,
            "minimum_per_document": min_characters,
            "maximum_per_document": max_characters,
            "mean_per_document": total_characters / valid_documents if valid_documents else None,
        },
        "tokens": {
            "total_stream_tokens": total_tokens,
            "minimum_per_document": min_tokens,
            "maximum_per_document": max_tokens,
            "mean_per_document": total_tokens / valid_documents if valid_documents else None,
            "train": _packed_statistics(train_tokens, sequence_length),
            "validation": _packed_statistics(validation_tokens, sequence_length),
        },
        "quality": {
            "training_compatible": (
                counts["invalid_json_rows"] == 0
                and counts["non_object_rows"] == 0
                and valid_documents > 0
            ),
            "exact_duplicate_fraction": (
                counts["exact_duplicate_rows"] / valid_documents if valid_documents else None
            ),
            "exact_duplicates_removed": False,
            "near_duplicate_check": False,
            "benchmark_contamination_check": False,
            "source_mixture_available": False,
        },
    }


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Audit training JSONL inputs.')
    commands = parser.add_subparsers(dest="command", required=True)
    audit_data = commands.add_parser('pretrain', help='scan pretraining JSONL and emit a reproducible data report')
    audit_data.add_argument("--data", type=Path, default=DEFAULT_PRETRAIN_DATA)
    audit_data.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    audit_data.add_argument("--output", type=Path)
    audit_data.add_argument(
        "--validation-fraction", type=float, default=option_default(PretrainOptions, "validation_fraction")
    )
    audit_data.add_argument(
        "--sequence-length", type=int, default=option_default(PretrainOptions, "sequence_length")
    )
    audit_data.add_argument("--tokenizer-batch-size", type=int, default=4096)
    audit_sft = commands.add_parser('sft', help='scan SFT JSONL and report data and target quality')
    audit_sft.add_argument("--data", type=Path, default=DEFAULT_SFT_DATA)
    audit_sft.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    audit_sft.add_argument("--output", type=Path)
    audit_sft.add_argument("--max-length", type=int, default=512)
    audit_sft.add_argument("--min-context-tokens", type=int, default=option_default(SFTOptions, "min_context_tokens"))
    audit_sft.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default=option_default(SFTOptions, "target_mode"),
    )
    audit_sft.add_argument("--validation-fraction", type=float, default=option_default(SFTOptions, "validation_fraction"))
    audit_sft.add_argument("--sample-size", type=int, default=5000)
    audit_sft.add_argument("--seed", type=int, default=option_default(SFTOptions, "seed"))
    audit_sft.add_argument(
        "--identity-pattern",
        action="append",
        default=[],
        help="case-insensitive identity/brand substring to count; repeat for multiple patterns",
    )
    audit_dpo = commands.add_parser('dpo', help='scan preference JSONL and report pair validity, splits, and truncation')
    audit_dpo.add_argument("--data", type=Path, default=DEFAULT_DPO_DATA)
    audit_dpo.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    audit_dpo.add_argument("--output", type=Path)
    audit_dpo.add_argument("--max-length", type=int, default=512)
    audit_dpo.add_argument(
        "--min-context-tokens", type=int, default=option_default(DPOOptions, "min_context_tokens")
    )
    audit_dpo.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default="reasoning_and_response",
    )
    audit_dpo.add_argument(
        "--validation-fraction", type=float, default=option_default(DPOOptions, "validation_fraction")
    )
    audit_dpo.add_argument("--sample-size", type=int, default=2000)
    audit_dpo.add_argument("--seed", type=int, default=option_default(DPOOptions, "seed"))
    audit_grpo = commands.add_parser('grpo', help='scan verifiable RL data, stable splits, duplicates, and invalid rows')
    audit_grpo.add_argument("--data", type=Path, default=DEFAULT_RL_DATA)
    audit_grpo.add_argument("--output", type=Path)
    audit_grpo.add_argument(
        "--validation-fraction", type=float, default=option_default(GRPOOptions, "validation_fraction")
    )
    audit_grpo.add_argument("--sample-size", type=int, default=2000)
    audit_grpo.add_argument("--seed", type=int, default=option_default(GRPOOptions, "seed"))
    audit_agent = commands.add_parser('agent', help='scan Agent-RL rows and verify executable tool capabilities')
    audit_agent.add_argument("--data", type=Path, default=DEFAULT_RL_DATA)
    audit_agent.add_argument("--output", type=Path)
    audit_agent.add_argument(
        "--validation-fraction", type=float, default=option_default(AgentRLOptions, "validation_fraction")
    )
    audit_agent.add_argument("--seed", type=int, default=option_default(AgentRLOptions, "seed"))
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "pretrain":
        tokenizer = load_tokenizer(arguments.tokenizer)
        result = audit_pretrain_jsonl(
            arguments.data,
            tokenizer,
            validation_fraction=arguments.validation_fraction,
            sequence_length=arguments.sequence_length,
            tokenizer_batch_size=arguments.tokenizer_batch_size,
        )
    elif arguments.command == "sft":
        tokenizer = load_tokenizer(arguments.tokenizer)
        result = audit_sft_jsonl(
            arguments.data,
            tokenizer,
            max_length=arguments.max_length,
            min_context_tokens=arguments.min_context_tokens,
            target_mode=arguments.target_mode,
            validation_fraction=arguments.validation_fraction,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
            identity_patterns=arguments.identity_pattern,
        )
    elif arguments.command == "dpo":
        tokenizer = load_tokenizer(arguments.tokenizer)
        result = audit_dpo_jsonl(
            arguments.data,
            tokenizer,
            max_length=arguments.max_length,
            min_context_tokens=arguments.min_context_tokens,
            target_mode=arguments.target_mode,
            validation_fraction=arguments.validation_fraction,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
        )
    elif arguments.command == "grpo":
        result = audit_rl_jsonl(
            arguments.data,
            validation_fraction=arguments.validation_fraction,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
        )
    elif arguments.command == "agent":
        result = audit_agent_jsonl(
            arguments.data,
            validation_fraction=arguments.validation_fraction,
            seed=arguments.seed,
        )
    if arguments.output is not None:
        atomic_write_json(arguments.output, result)
        result["report"] = str(arguments.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
