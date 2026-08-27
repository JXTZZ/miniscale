from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import BinaryIO, Literal, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import collate_lm_batch
from .tokenizer import Tokenizer


DPO_DATA_ORDER_VERSION = "indexed_global_permutation_v1"
DPO_PAIR_FORMAT_VERSION = "shared_prompt_last_assistant_v1"
DPO_TRUNCATION_VERSION = "shared_recent_context_completion_prefix_v1"

VALID_ROLES = {"system", "user", "assistant", "tool"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def preference_pair_digest(
    chosen: Sequence[dict[str, object]], rejected: Sequence[dict[str, object]]
) -> bytes:
    return hashlib.blake2b(_canonical([list(chosen), list(rejected)]), digest_size=16).digest()


def preference_prompt_digest(messages: Sequence[dict[str, object]]) -> bytes:
    return hashlib.blake2b(_canonical(list(messages)), digest_size=16).digest()


def _is_validation_prompt(digest: bytes, validation_fraction: float) -> bool:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    return int.from_bytes(digest[:8], "big") / 2**64 < validation_fraction


def _assistant_has_target(message: dict[str, object], target_mode: str) -> bool:
    content = message.get("content")
    if (isinstance(content, str) and bool(content.strip())) or bool(message.get("tool_calls")):
        return True
    reasoning = message.get("reasoning_content")
    return target_mode == "reasoning_and_response" and isinstance(reasoning, str) and bool(reasoning.strip())


@dataclass(frozen=True, slots=True)
class PreferencePair:
    prompt: list[dict[str, object]]
    chosen: list[dict[str, object]]
    rejected: list[dict[str, object]]


def parse_preference_pair(
    row: object,
    *,
    target_mode: str,
    location: str,
) -> PreferencePair:
    """Validate one preference row and return its shared-prompt representation."""

    if target_mode not in {"reasoning_and_response", "response_only"}:
        raise ValueError("target_mode must be 'reasoning_and_response' or 'response_only'")
    if not isinstance(row, dict):
        raise ValueError(f"expected a JSON object at {location}")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    for name, messages in (("chosen", chosen), ("rejected", rejected)):
        if not isinstance(messages, list) or len(messages) < 2 or not all(
            isinstance(message, dict) for message in messages
        ):
            raise ValueError(f"{name} must be a conversation with prompt and response at {location}")
        invalid_roles = [message.get("role") for message in messages if message.get("role") not in VALID_ROLES]
        if invalid_roles:
            raise ValueError(f"{name} contains invalid roles {invalid_roles!r} at {location}")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"{name} must end with an assistant response at {location}")
        if not _assistant_has_target(messages[-1], target_mode):
            raise ValueError(f"{name} has an empty target response at {location}")
    assert isinstance(chosen, list) and isinstance(rejected, list)
    if chosen[:-1] != rejected[:-1]:
        raise ValueError(f"chosen and rejected must share an identical prompt at {location}")
    if chosen[-1] == rejected[-1]:
        raise ValueError(f"chosen and rejected responses are identical at {location}")
    prompt = [dict(message) for message in chosen[:-1]]
    return PreferencePair(
        prompt=prompt,
        chosen=[dict(message) for message in chosen],
        rejected=[dict(message) for message in rejected],
    )


@dataclass(frozen=True, slots=True)
class PreferenceIndexStats:
    rows: int
    valid_pairs: int
    invalid_pairs: int
    duplicate_pairs: int
    contradictory_pairs: int
    repeated_prompt_pairs: int
    train_pairs: int
    validation_pairs: int


@dataclass(slots=True)
class PreferenceCorpusIndex:
    path: Path
    train_offsets: array[int]
    validation_offsets: array[int]
    identity: dict[str, object]
    stats: PreferenceIndexStats

    @classmethod
    def build(
        cls,
        path: str | Path,
        *,
        validation_fraction: float,
        target_mode: str,
        destination: Literal["split", "train", "validation"] = "split",
        deduplicate_exact: bool = True,
    ) -> PreferenceCorpusIndex:
        if destination not in {"split", "train", "validation"}:
            raise ValueError("destination must be 'split', 'train', or 'validation'")
        if destination == "split" and not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"DPO data does not exist: {source_path}")

        train_offsets = array("Q")
        validation_offsets = array("Q")
        seen_pairs: set[bytes] = set()
        seen_prompts: set[bytes] = set()
        raw_digest = hashlib.sha256()
        size_bytes = 0
        rows = valid = invalid = duplicates = contradictions = repeated_prompts = 0

        with source_path.open("rb") as source:
            line_number = 0
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                line_number += 1
                raw_digest.update(line)
                size_bytes += len(line)
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise ValueError(f"invalid JSON at {source_path}:{line_number}") from error
                try:
                    pair = parse_preference_pair(
                        row,
                        target_mode=target_mode,
                        location=f"{source_path}:{line_number}",
                    )
                except ValueError:
                    invalid += 1
                    continue
                valid += 1
                digest = preference_pair_digest(pair.chosen, pair.rejected)
                reverse_digest = preference_pair_digest(pair.rejected, pair.chosen)
                if digest in seen_pairs:
                    duplicates += 1
                    if deduplicate_exact:
                        continue
                elif reverse_digest in seen_pairs:
                    contradictions += 1
                    if deduplicate_exact:
                        continue
                else:
                    seen_pairs.add(digest)

                prompt_digest = preference_prompt_digest(pair.prompt)
                if prompt_digest in seen_prompts:
                    repeated_prompts += 1
                else:
                    seen_prompts.add(prompt_digest)
                if destination == "train":
                    validation = False
                elif destination == "validation":
                    validation = True
                else:
                    validation = _is_validation_prompt(prompt_digest, validation_fraction)
                (validation_offsets if validation else train_offsets).append(offset)

        return cls(
            path=source_path,
            train_offsets=train_offsets,
            validation_offsets=validation_offsets,
            identity={"kind": "file", "sha256": raw_digest.hexdigest(), "size_bytes": size_bytes},
            stats=PreferenceIndexStats(
                rows=rows,
                valid_pairs=valid,
                invalid_pairs=invalid,
                duplicate_pairs=duplicates,
                contradictory_pairs=contradictions,
                repeated_prompt_pairs=repeated_prompts,
                train_pairs=len(train_offsets),
                validation_pairs=len(validation_offsets),
            ),
        )


@dataclass(frozen=True, slots=True)
class EncodedPreferencePair:
    chosen_input_ids: list[int]
    chosen_labels: list[int]
    rejected_input_ids: list[int]
    rejected_labels: list[int]
    context_tokens: int
    chosen_target_tokens: int
    rejected_target_tokens: int
    dropped_context_tokens: int
    dropped_chosen_tokens: int
    dropped_rejected_tokens: int


def _target_start(labels: Sequence[int], *, name: str) -> int:
    for index, label in enumerate(labels):
        if label != -100:
            if index < 1:
                raise ValueError(f"{name} has no prompt context before its target")
            return index
    raise ValueError(f"{name} contains no supervised assistant tokens")


def encode_preference_pair(
    pair: PreferencePair,
    tokenizer: Tokenizer,
    *,
    max_length: int,
    min_context_tokens: int,
    target_mode: str,
) -> EncodedPreferencePair:
    """Encode chosen/rejected with one identical, recent prompt context."""

    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    if not 1 <= min_context_tokens < max_length:
        raise ValueError("min_context_tokens must be in [1, max_length)")
    chosen_ids, chosen_labels = tokenizer.encode_sft(
        pair.chosen, target_mode=target_mode, target_assistant_index=-1
    )
    rejected_ids, rejected_labels = tokenizer.encode_sft(
        pair.rejected, target_mode=target_mode, target_assistant_index=-1
    )
    chosen_start = _target_start(chosen_labels, name="chosen")
    rejected_start = _target_start(rejected_labels, name="rejected")
    chosen_context = chosen_ids[:chosen_start]
    rejected_context = rejected_ids[:rejected_start]
    if chosen_context != rejected_context:
        raise ValueError("chosen and rejected tokenization produced different prompt contexts")
    if not chosen_context:
        raise ValueError("preference pair contains no prompt context")

    chosen_targets = chosen_ids[chosen_start:]
    rejected_targets = rejected_ids[rejected_start:]
    chosen_target_labels = chosen_labels[chosen_start:]
    rejected_target_labels = rejected_labels[rejected_start:]
    minimum_context = min(len(chosen_context), min_context_tokens)
    full_target_context = max_length - max(len(chosen_targets), len(rejected_targets))
    context_keep = min(
        len(chosen_context),
        max(minimum_context, min(max_length - 1, full_target_context)),
    )
    target_budget = max_length - context_keep
    if target_budget < 1:
        raise ValueError("max_length leaves no room for preference targets")

    context = chosen_context[-context_keep:]
    kept_chosen_ids = chosen_targets[:target_budget]
    kept_rejected_ids = rejected_targets[:target_budget]
    kept_chosen_labels = chosen_target_labels[:target_budget]
    kept_rejected_labels = rejected_target_labels[:target_budget]
    chosen_result_labels = [-100] * context_keep + kept_chosen_labels
    rejected_result_labels = [-100] * context_keep + kept_rejected_labels
    if not any(label != -100 for label in chosen_result_labels[1:]):
        raise ValueError("preference truncation removed every chosen target")
    if not any(label != -100 for label in rejected_result_labels[1:]):
        raise ValueError("preference truncation removed every rejected target")
    return EncodedPreferencePair(
        chosen_input_ids=context + kept_chosen_ids,
        chosen_labels=chosen_result_labels,
        rejected_input_ids=context + kept_rejected_ids,
        rejected_labels=rejected_result_labels,
        context_tokens=context_keep,
        chosen_target_tokens=sum(label != -100 for label in chosen_result_labels[1:]),
        rejected_target_tokens=sum(label != -100 for label in rejected_result_labels[1:]),
        dropped_context_tokens=len(chosen_context) - context_keep,
        dropped_chosen_tokens=len(chosen_targets) - len(kept_chosen_ids),
        dropped_rejected_tokens=len(rejected_targets) - len(kept_rejected_ids),
    )


class IndexedPreferenceDataset(Dataset[dict[str, dict[str, Tensor]]]):
    """Random-access preference pairs backed by JSONL byte offsets."""

    def __init__(
        self,
        index: PreferenceCorpusIndex,
        tokenizer: Tokenizer,
        *,
        split: Literal["train", "validation"],
        max_length: int,
        min_context_tokens: int,
        target_mode: str,
    ) -> None:
        self.path = index.path
        self.offsets = index.train_offsets if split == "train" else index.validation_offsets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_context_tokens = min_context_tokens
        self.target_mode = target_mode
        self._source: BinaryIO | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_source"] = None
        return state

    def close(self) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

    def __del__(self) -> None:
        self.close()

    def __getitem__(self, index: int) -> dict[str, dict[str, Tensor]]:
        if self._source is None:
            self._source = self.path.open("rb")
        self._source.seek(self.offsets[index])
        row = json.loads(self._source.readline())
        pair = parse_preference_pair(row, target_mode=self.target_mode, location=f"{self.path}@{self.offsets[index]}")
        encoded = encode_preference_pair(
            pair,
            self.tokenizer,
            max_length=self.max_length,
            min_context_tokens=self.min_context_tokens,
            target_mode=self.target_mode,
        )
        return {
            "chosen": {
                "input_ids": torch.tensor(encoded.chosen_input_ids, dtype=torch.long),
                "labels": torch.tensor(encoded.chosen_labels, dtype=torch.long),
            },
            "rejected": {
                "input_ids": torch.tensor(encoded.rejected_input_ids, dtype=torch.long),
                "labels": torch.tensor(encoded.rejected_labels, dtype=torch.long),
            },
        }


def collate_preference_batch(
    examples: Sequence[dict[str, dict[str, Tensor]]], pad_token_id: int
) -> dict[str, dict[str, Tensor]]:
    if not examples:
        raise ValueError("cannot collate an empty preference batch")
    count = len(examples)
    combined = collate_lm_batch(
        [example[side] for side in ("chosen", "rejected") for example in examples],
        pad_token_id,
    )
    return {
        "chosen": {name: value[:count] for name, value in combined.items()},
        "rejected": {name: value[count:] for name, value in combined.items()},
    }


def fixed_preference_validation_batches(
    dataset: Dataset[dict[str, dict[str, Tensor]]],
    *,
    batch_size: int,
    batches: int,
    pad_token_id: int,
    seed: int,
) -> list[dict[str, dict[str, Tensor]]]:
    if batch_size < 1 or batches < 1:
        raise ValueError("batch_size and batches must be positive")
    if not len(dataset):
        raise ValueError("DPO validation split contains no pairs")
    capacity = min(len(dataset), batch_size * batches)
    selected = random.Random(seed).sample(range(len(dataset)), capacity)
    return [
        collate_preference_batch(
            [dataset[index] for index in selected[start : start + batch_size]],
            pad_token_id,
        )
        for start in range(0, capacity, batch_size)
    ]
