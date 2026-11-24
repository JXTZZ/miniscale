"""Streaming pretraining and assistant-turn SFT datasets."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import BinaryIO, Literal

from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from model.tokenizer import ByteTokenizer, Tokenizer


DEFAULT_PRETRAIN_DATA = Path("dataset/pretrain_t2t_mini.jsonl")
DEFAULT_SFT_DATA = Path("dataset/sft_t2t_mini.jsonl")


def is_validation_text(text: str, validation_fraction: float) -> bool:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    bucket = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big") / 2**64
    return bucket < validation_fraction


class PretrainDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, texts: Sequence[str], tokenizer: ByteTokenizer, sequence_length: int) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        stream: list[int] = []
        for text in texts:
            stream.extend(tokenizer.encode(text, bos=True, eos=True))
        self.examples = [
            stream[start : start + sequence_length]
            for start in range(0, max(len(stream) - 1, 0), sequence_length - 1)
            if len(stream[start : start + sequence_length]) >= 2
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        ids = torch.tensor(self.examples[index], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class SFTDataset(Dataset[dict[str, Tensor]]):
    """Small eager ByteTokenizer dataset used only by the smoke pipeline."""

    def __init__(
        self,
        conversations: Sequence[Sequence[dict[str, str]]],
        tokenizer: ByteTokenizer,
        max_length: int | None = None,
    ) -> None:
        self.examples = []
        for messages in conversations:
            input_ids, labels = tokenizer.encode_sft(messages)
            if max_length is not None:
                # Keep the newest turns so the supervised assistant answer is not
                # silently discarded when a byte-level example exceeds context.
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]
            self.examples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        input_ids, labels = self.examples[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            yield value


def _worker_rows(path: str | Path) -> Iterator[dict[str, object]]:
    worker = get_worker_info()
    for index, row in enumerate(iter_jsonl(path)):
        if worker is None or index % worker.num_workers == worker.id:
            yield row


class JsonlPretrainDataset(IterableDataset[dict[str, Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Tokenizer,
        sequence_length: int,
        *,
        split: str = "all",
        validation_fraction: float = 0.005,
        shuffle_buffer_size: int = 0,
        seed: int = 42,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.split = split
        self.validation_fraction = validation_fraction
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self._iteration = 0
        if split not in {"all", "train", "validation"}:
            raise ValueError("split must be 'all', 'train', or 'validation'")
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative")

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        iteration = self._iteration
        self._iteration += 1
        examples = self._iter_packed_examples()
        if self.shuffle_buffer_size <= 1:
            yield from examples
            return

        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = random.Random(self.seed + iteration * 1_000_003 + worker_id)
        buffer: list[dict[str, Tensor]] = []
        for example in examples:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(example)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = example
        while buffer:
            index = rng.randrange(len(buffer))
            selected = buffer[index]
            buffer[index] = buffer[-1]
            buffer.pop()
            yield selected

    def _iter_packed_examples(self) -> Iterator[dict[str, Tensor]]:
        buffer: list[int] = []
        target_length = self.sequence_length
        for row in _worker_rows(self.path):
            text = row.get("text")
            if not isinstance(text, str) or not text:
                continue
            if self.split != "all":
                is_validation = is_validation_text(text, self.validation_fraction)
                if (self.split == "validation") != is_validation:
                    continue
            buffer.extend(self.tokenizer.encode(text, bos=True, eos=True))
            while len(buffer) >= target_length:
                ids = torch.tensor(buffer[:target_length], dtype=torch.long)
                # Retain the final token as the first context token of the next
                # block, so the boundary next-token target is not discarded.
                del buffer[: self.sequence_length - 1]
                yield {"input_ids": ids, "labels": ids.clone()}


def load_jsonl_rows(path: str | Path, limit: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in iter_jsonl(path):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def collate_lm_batch(examples: Sequence[dict[str, Tensor]], pad_token_id: int = 0) -> dict[str, Tensor]:
    input_rows = [example["input_ids"] for example in examples]
    label_rows = [example["labels"] for example in examples]
    masks = [torch.ones_like(example["input_ids"]) for example in examples]
    return {
        "input_ids": pad_sequence(input_rows, batch_first=True, padding_value=pad_token_id),
        "labels": pad_sequence(label_rows, batch_first=True, padding_value=-100),
        "attention_mask": pad_sequence(masks, batch_first=True, padding_value=0),
    }


def reservoir_sample_lm_batches(
    examples: Iterable[dict[str, Tensor]],
    *,
    batch_size: int,
    batches: int,
    pad_token_id: int,
    seed: int,
) -> list[dict[str, Tensor]]:
    """Build a deterministic fixed validation set sampled across a full stream."""

    if batch_size < 1 or batches < 1:
        raise ValueError("batch_size and batches must be positive")
    capacity = batch_size * batches
    rng = random.Random(seed)
    reservoir: list[dict[str, Tensor]] = []
    for index, example in enumerate(examples):
        if index < capacity:
            reservoir.append(example)
            continue
        replacement = rng.randrange(index + 1)
        if replacement < capacity:
            reservoir[replacement] = example
    if not reservoir:
        raise ValueError("validation corpus produced no packed examples")
    rng.shuffle(reservoir)
    return [
        collate_lm_batch(reservoir[start : start + batch_size], pad_token_id)
        for start in range(0, len(reservoir), batch_size)
    ]


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


def assistant_has_target(message: dict[str, object], target_mode: str) -> bool:
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
                    if not assistant_has_target(message, target_mode):
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
