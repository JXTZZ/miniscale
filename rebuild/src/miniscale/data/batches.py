"""Language-model batch padding and fixed validation sampling."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import random

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


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
