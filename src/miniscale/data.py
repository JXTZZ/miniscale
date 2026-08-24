from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


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
