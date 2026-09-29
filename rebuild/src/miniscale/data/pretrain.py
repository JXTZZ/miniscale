from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import hashlib
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ..tokenizer import ByteTokenizer, Tokenizer
from .jsonl import iter_jsonl


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
        # Non-persistent workers receive a fresh dataset copy each epoch, so
        # their local _iteration resets. DataLoader's seeded worker seed advances.
        epoch_seed = iteration * 1_000_003 + (worker.seed if worker is not None else 0)
        rng = random.Random(self.seed + epoch_seed + worker_id)
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
            offset = 0
            while len(buffer) - offset >= target_length:
                ids = torch.tensor(buffer[offset : offset + target_length], dtype=torch.long)
                # Adjacent blocks overlap by one token, preserving every
                # next-token target across both block and document boundaries.
                offset += target_length - 1
                yield {"input_ids": ids, "labels": ids.clone()}
            # Remove consumed tokens once per document. Repeatedly deleting
            # from the front becomes quadratic for a long document.
            if offset:
                del buffer[:offset]
