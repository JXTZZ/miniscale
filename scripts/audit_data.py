"""Audit the JSONL inputs for the five training stages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

# Support running this file directly as well as with python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import atomic_write_json, path_identity, tokenizer_identity
from dataset.lm_dataset import DEFAULT_PRETRAIN_DATA, is_validation_text
from dataset.preference_dataset import DEFAULT_DPO_DATA
from dataset.rl_dataset import build_agent_corpus, build_rl_corpus, DEFAULT_RL_DATA
from dataset.sft_dataset import DEFAULT_SFT_DATA
from model.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer, Tokenizer
from scripts.chat_audit import audit_dpo_jsonl, audit_sft_jsonl, integer_percentiles
from trainer.cli import option_default
from trainer.rl.options import AgentRLOptions, GRPOOptions
from trainer.train_dpo import DPOOptions
from trainer.train_full_sft import SFTOptions
from trainer.train_pretrain import PretrainOptions


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit training JSONL inputs.")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text, data, options, *, tokenizer=False, seed=True):
        subparser = commands.add_parser(name, help=help_text)
        subparser.add_argument("--data", type=Path, default=data)
        subparser.add_argument("--output", type=Path)
        subparser.add_argument("--validation-fraction", type=float,
                               default=option_default(options, "validation_fraction"))
        if tokenizer:
            subparser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
        if seed:
            subparser.add_argument("--seed", type=int, default=option_default(options, "seed"))
        return subparser

    pretrain = command("pretrain", "scan text and packing", DEFAULT_PRETRAIN_DATA, PretrainOptions,
                       tokenizer=True, seed=False)
    pretrain.add_argument("--sequence-length", type=int, default=option_default(PretrainOptions, "sequence_length"))
    pretrain.add_argument("--tokenizer-batch-size", type=int, default=4096)
    for name, help_text, data, options, sample_size in (
        ("sft", "scan assistant targets and data quality", DEFAULT_SFT_DATA, SFTOptions, 5000),
        ("dpo", "scan preference pairs, splits, and truncation", DEFAULT_DPO_DATA, DPOOptions, 2000),
    ):
        subparser = command(name, help_text, data, options, tokenizer=True)
        subparser.add_argument("--max-length", type=int, default=512)
        subparser.add_argument("--min-context-tokens", type=int, default=option_default(options, "min_context_tokens"))
        subparser.add_argument("--target-mode", choices=("reasoning_and_response", "response_only"),
                               default=option_default(SFTOptions, "target_mode"))
        subparser.add_argument("--sample-size", type=int, default=sample_size)
        if name == "sft":
            subparser.add_argument("--identity-pattern", action="append", default=[],
                                   help="case-insensitive identity/brand substring to count; repeat for multiple patterns")
    grpo = command("grpo", "scan verifiable RL tasks and splits", DEFAULT_RL_DATA, GRPOOptions)
    grpo.add_argument("--sample-size", type=int, default=2000)
    command("agent", "scan Agent-RL tasks and available tools", DEFAULT_RL_DATA, AgentRLOptions)
    return parser


def main(argv: list[str] | None = None) -> None:
    options = vars(build_parser().parse_args(argv))
    command = options.pop("command")
    data, output = options.pop("data"), options.pop("output")
    audit = {
        "pretrain": audit_pretrain_jsonl, "sft": audit_sft_jsonl, "dpo": audit_dpo_jsonl,
        "grpo": audit_rl_jsonl, "agent": audit_agent_jsonl,
    }[command]
    if "tokenizer" in options:
        options["tokenizer"] = load_tokenizer(options["tokenizer"])
    if "identity_pattern" in options:
        options["identity_patterns"] = options.pop("identity_pattern")
    result = audit(data, **options)
    if output is not None:
        atomic_write_json(output, result)
        result["report"] = str(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
