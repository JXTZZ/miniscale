from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
import json
import hashlib
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .tokenizer import ByteTokenizer, Tokenizer


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
    max_length = max(example["input_ids"].numel() for example in examples)
    input_rows, label_rows, masks = [], [], []
    for example in examples:
        pad_length = max_length - example["input_ids"].numel()
        input_rows.append(F_pad(example["input_ids"], pad_length, pad_token_id))
        label_rows.append(F_pad(example["labels"], pad_length, -100))
        masks.append(F_pad(torch.ones_like(example["input_ids"]), pad_length, 0))
    return {
        "input_ids": torch.stack(input_rows),
        "labels": torch.stack(label_rows),
        "attention_mask": torch.stack(masks),
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


def F_pad(tensor: Tensor, right: int, value: int) -> Tensor:
    if right == 0:
        return tensor
    return torch.cat((tensor, torch.full((right,), value, dtype=tensor.dtype)))
