"""Small in-memory SFT dataset for smoke tests and examples."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..tokenizer import ByteTokenizer


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
