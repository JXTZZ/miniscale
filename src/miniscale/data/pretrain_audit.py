from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..integrity import atomic_write_json, path_identity, tokenizer_identity
from ..tokenizer import Tokenizer
from . import is_validation_text


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


def save_data_audit(report: dict[str, Any], path: str | Path) -> Path:
    return atomic_write_json(path, report)
