from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import json
import hashlib

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .tokenizer import ByteTokenizer, Tokenizer


class PretrainDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, texts: Sequence[str], tokenizer: ByteTokenizer, sequence_length: int) -> None:
        stream: list[int] = []
        for text in texts:
            stream.extend(tokenizer.encode(text, bos=True, eos=True))
        self.examples = [
            stream[start : start + sequence_length + 1]
            for start in range(0, max(len(stream) - 1, 0), sequence_length)
            if len(stream[start : start + sequence_length + 1]) >= 2
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        ids = torch.tensor(self.examples[index], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class SFTDataset(Dataset[dict[str, Tensor]]):
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
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.split = split
        self.validation_fraction = validation_fraction

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        buffer: list[int] = []
        target_length = self.sequence_length + 1
        for row in _worker_rows(self.path):
            text = row.get("text")
            if not isinstance(text, str) or not text:
                continue
            if self.split != "all":
                bucket = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big") / 2**64
                is_validation = bucket < self.validation_fraction
                if (self.split == "validation") != is_validation:
                    continue
            buffer.extend(self.tokenizer.encode(text, bos=True, eos=True))
            while len(buffer) >= target_length:
                ids = torch.tensor(buffer[:target_length], dtype=torch.long)
                del buffer[: self.sequence_length]
                yield {"input_ids": ids, "labels": ids.clone()}


class JsonlSFTDataset(IterableDataset[dict[str, Tensor]]):
    def __init__(self, path: str | Path, tokenizer: Tokenizer, max_length: int) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        for row in _worker_rows(self.path):
            messages = row.get("conversations")
            if not isinstance(messages, list) or not messages:
                continue
            input_ids, labels = self.tokenizer.encode_sft(messages)
            input_ids, labels = input_ids[-self.max_length :], labels[-self.max_length :]
            if any(label != -100 for label in labels):
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }


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


def F_pad(tensor: Tensor, right: int, value: int) -> Tensor:
    if right == 0:
        return tensor
    return torch.cat((tensor, torch.full((right,), value, dtype=tensor.dtype)))
