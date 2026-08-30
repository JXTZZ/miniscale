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

from ..tokenizer import Tokenizer
from . import collate_lm_batch


SFT_DATA_ORDER_VERSION = "indexed_global_permutation_v1"
SFT_EXAMPLE_FORMAT_VERSION = "assistant_turn_selection_v2"
SFT_TRUNCATION_VERSION = "preserve_recent_context_and_target_prefix_v1"
SFT_SELECTION_VERSION = "target_positions_v1"


def canonical_conversation(messages: Sequence[dict[str, object]]) -> bytes:
    """Return the semantic, path-independent identity input for a conversation."""

    return json.dumps(
        list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def conversation_digest(messages: Sequence[dict[str, object]]) -> bytes:
    return hashlib.blake2b(canonical_conversation(messages), digest_size=16).digest()


def is_validation_conversation(digest: bytes, validation_fraction: float) -> bool:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < validation_fraction


def _assistant_has_target(message: dict[str, object], target_mode: str) -> bool:
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    if (isinstance(content, str) and bool(content.strip())) or bool(tool_calls):
        return True
    reasoning = message.get("reasoning_content")
    return target_mode == "reasoning_and_response" and isinstance(reasoning, str) and bool(reasoning.strip())


@dataclass(frozen=True, slots=True)
class SFTIndexStats:
    rows: int
    conversations: int
    invalid_conversations: int
    duplicate_conversations: int
    assistant_messages: int
    selected_assistant_messages: int
    unselected_assistant_messages: int
    empty_target_messages: int
    train_examples: int
    validation_examples: int


@dataclass(slots=True)
class SFTCorpusIndex:
    path: Path
    train_offsets: array[int]
    train_message_positions: array[int]
    validation_offsets: array[int]
    validation_message_positions: array[int]
    identity: dict[str, object]
    stats: SFTIndexStats

    @classmethod
    def build(
        cls,
        path: str | Path,
        *,
        validation_fraction: float,
        target_mode: str,
        destination: Literal["split", "train", "validation"] = "split",
        deduplicate_exact: bool = True,
    ) -> SFTCorpusIndex:
        if target_mode not in {"reasoning_and_response", "response_only"}:
            raise ValueError("target_mode must be 'reasoning_and_response' or 'response_only'")
        if destination not in {"split", "train", "validation"}:
            raise ValueError("destination must be 'split', 'train', or 'validation'")
        if destination == "split" and not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")

        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(f"SFT data does not exist: {source_path}")
        train_offsets = array("Q")
        train_positions = array("I")
        validation_offsets = array("Q")
        validation_positions = array("I")
        seen: set[bytes] = set()
        raw_digest = hashlib.sha256()
        size_bytes = 0
        rows = conversations = invalid = duplicates = assistant_messages = selected_messages = 0
        unselected_messages = empty_targets = 0

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
                if not isinstance(row, dict):
                    raise ValueError(f"expected a JSON object at {source_path}:{line_number}")
                messages = row.get("conversations")
                if not isinstance(messages, list) or not messages or not all(isinstance(item, dict) for item in messages):
                    invalid += 1
                    continue
                if any(message.get("role") not in {"system", "user", "assistant", "tool"} for message in messages):
                    invalid += 1
                    continue
                conversations += 1
                selected_positions: set[int] | None = None
                selection = row.get("sft_selection")
                if selection is not None:
                    raw_positions = selection.get("target_positions") if isinstance(selection, dict) else None
                    if not isinstance(raw_positions, list) or not all(
                        isinstance(position, int) and 0 <= position < len(messages)
                        for position in raw_positions
                    ):
                        invalid += 1
                        conversations -= 1
                        continue
                    selected_positions = set(raw_positions)
                semantic_digest = conversation_digest(messages)
                duplicate = semantic_digest in seen
                if duplicate:
                    duplicates += 1
                    if deduplicate_exact:
                        continue
                else:
                    seen.add(semantic_digest)

                if destination == "train":
                    validation = False
                elif destination == "validation":
                    validation = True
                else:
                    validation = is_validation_conversation(semantic_digest, validation_fraction)
                offsets = validation_offsets if validation else train_offsets
                positions = validation_positions if validation else train_positions
                for message_position, message in enumerate(messages):
                    if message.get("role") != "assistant":
                        continue
                    assistant_messages += 1
                    if selected_positions is not None and message_position not in selected_positions:
                        unselected_messages += 1
                        continue
                    selected_messages += 1
                    if not _assistant_has_target(message, target_mode):
                        empty_targets += 1
                        continue
                    offsets.append(offset)
                    positions.append(message_position)

        stats = SFTIndexStats(
            rows=rows,
            conversations=conversations,
            invalid_conversations=invalid,
            duplicate_conversations=duplicates,
            assistant_messages=assistant_messages,
            selected_assistant_messages=selected_messages,
            unselected_assistant_messages=unselected_messages,
            empty_target_messages=empty_targets,
            train_examples=len(train_offsets),
            validation_examples=len(validation_offsets),
        )
        return cls(
            path=source_path,
            train_offsets=train_offsets,
            train_message_positions=train_positions,
            validation_offsets=validation_offsets,
            validation_message_positions=validation_positions,
            identity={"kind": "file", "sha256": raw_digest.hexdigest(), "size_bytes": size_bytes},
            stats=stats,
        )


@dataclass(frozen=True, slots=True)
class EncodedSFTExample:
    input_ids: list[int]
    labels: list[int]
    original_tokens: int
    dropped_context_tokens: int
    dropped_target_tokens: int

    @property
    def supervised_tokens(self) -> int:
        return sum(label != -100 for label in self.labels[1:])


def truncate_sft_example(
    input_ids: Sequence[int],
    labels: Sequence[int],
    *,
    max_length: int,
    min_context_tokens: int,
) -> EncodedSFTExample:
    """Fit one assistant-turn example without ever dropping all prompt context."""

    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels must have equal length")
    if max_length < 3:
        raise ValueError("max_length must be at least 3")
    if not 1 <= min_context_tokens < max_length:
        raise ValueError("min_context_tokens must be in [1, max_length)")
    supervised = [index for index, label in enumerate(labels) if label != -100]
    if not supervised:
        raise ValueError("SFT example contains no supervised assistant tokens")
    first_target = supervised[0]
    if first_target < 1:
        raise ValueError("SFT example has no prompt context before its first target")

    original_tokens = len(input_ids)
    if original_tokens <= max_length:
        result_ids = list(input_ids)
        result_labels = list(labels)
        dropped_context = dropped_target = 0
    else:
        context_ids = list(input_ids[:first_target])
        target_ids = list(input_ids[first_target:])
        target_labels = list(labels[first_target:])
        context_keep = min(len(context_ids), min_context_tokens)
        if len(target_ids) <= max_length - context_keep:
            context_keep = min(len(context_ids), max_length - len(target_ids))
        target_keep = max_length - context_keep
        if target_keep < 1:
            raise ValueError("max_length leaves no room for supervised tokens")
        result_ids = context_ids[-context_keep:] + target_ids[:target_keep]
        result_labels = [-100] * context_keep + target_labels[:target_keep]
        dropped_context = len(context_ids) - context_keep
        dropped_target = len(target_ids) - target_keep

    if not any(label != -100 for label in result_labels[1:]):
        raise ValueError("SFT truncation removed every next-token target")
    if result_labels[0] != -100:
        raise AssertionError("SFT truncation must retain prompt context at position zero")
    return EncodedSFTExample(
        input_ids=result_ids,
        labels=result_labels,
        original_tokens=original_tokens,
        dropped_context_tokens=dropped_context,
        dropped_target_tokens=dropped_target,
    )


class IndexedJsonlSFTDataset(Dataset[dict[str, Tensor]]):
    """Random-access assistant-turn examples backed by byte offsets in JSONL."""

    def __init__(
        self,
        index: SFTCorpusIndex,
        tokenizer: Tokenizer,
        *,
        split: Literal["train", "validation"],
        max_length: int,
        min_context_tokens: int,
        target_mode: str,
    ) -> None:
        self.path = index.path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_context_tokens = min_context_tokens
        self.target_mode = target_mode
        self.offsets = index.train_offsets if split == "train" else index.validation_offsets
        self.message_positions = (
            index.train_message_positions if split == "train" else index.validation_message_positions
        )
        self._source: BinaryIO | None = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_source"] = None
        return state

    def close(self) -> None:
        source = getattr(self, "_source", None)
        if source is not None:
            source.close()
            self._source = None

    def __del__(self) -> None:
        self.close()

    def _read_messages(self, index: int) -> list[dict[str, object]]:
        if self._source is None:
            self._source = self.path.open("rb")
        self._source.seek(self.offsets[index])
        row = json.loads(self._source.readline())
        messages = row.get("conversations")
        if not isinstance(messages, list):
            raise RuntimeError("indexed SFT row no longer contains conversations; input changed during training")
        position = self.message_positions[index]
        return [dict(message) for message in messages[: position + 1]]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        messages = self._read_messages(index)
        input_ids, labels = self.tokenizer.encode_sft(
            messages,
            target_mode=self.target_mode,
            target_assistant_index=-1,
        )
        example = truncate_sft_example(
            input_ids,
            labels,
            max_length=self.max_length,
            min_context_tokens=self.min_context_tokens,
        )
        return {
            "input_ids": torch.tensor(example.input_ids, dtype=torch.long),
            "labels": torch.tensor(example.labels, dtype=torch.long),
        }


def fixed_validation_batches(
    dataset: Dataset[dict[str, Tensor]],
    *,
    batch_size: int,
    batches: int,
    pad_token_id: int,
    seed: int,
) -> list[dict[str, Tensor]]:
    if batch_size < 1 or batches < 1:
        raise ValueError("batch_size and batches must be positive")
    if not len(dataset):
        raise ValueError("SFT validation split contains no examples")
    capacity = min(len(dataset), batch_size * batches)
    selected = random.Random(seed).sample(range(len(dataset)), capacity)
    return [
        collate_lm_batch(
            [dataset[index] for index in selected[start : start + batch_size]],
            pad_token_id,
        )
        for start in range(0, capacity, batch_size)
    ]
