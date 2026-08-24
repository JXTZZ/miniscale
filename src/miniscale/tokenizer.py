from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
import json
from typing import Protocol


class Tokenizer(Protocol):
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    unk_token_id: int
    vocab_size: int

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]: ...
    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str: ...
    def format_messages(self, messages: Sequence[dict[str, object]], *, generation_prompt: bool = False) -> str: ...
    def encode_sft(self, messages: Sequence[dict[str, object]]) -> tuple[list[int], list[int]]: ...


def _message_content(message: dict[str, object]) -> str:
    content = message.get("content") or ""
    if message.get("tools"):
        content = f"{content}\n{message['tools']}".strip()
    if message.get("tool_calls"):
        content = f"{content}<tool_call>{message['tool_calls']}</tool_call>"
    return str(content)


class ChatTemplateMixin:
    def format_messages(self, messages: Sequence[dict[str, object]], *, generation_prompt: bool = False) -> str:
        text = "".join(f"<|{message['role']}|>\n{_message_content(message)}<|end|>\n" for message in messages)
        if generation_prompt:
            text += "<|assistant|>\n"
        return text

    def encode_sft(self, messages: Sequence[dict[str, object]]) -> tuple[list[int], list[int]]:
        input_ids = [self.bos_token_id]
        labels = [-100]
        for message in messages:
            prefix = self.encode(f"<|{message['role']}|>\n")
            body = self.encode(f"{_message_content(message)}<|end|>\n")
            input_ids.extend(prefix)
            input_ids.extend(body)
            labels.extend([-100] * len(prefix))
            labels.extend(body if message["role"] == "assistant" else [-100] * len(body))
        input_ids.append(self.eos_token_id)
        labels.append(self.eos_token_id if messages and messages[-1]["role"] == "assistant" else -100)
        return input_ids, labels


class ByteTokenizer(ChatTemplateMixin):
    """A deterministic UTF-8 tokenizer that keeps the project self-contained."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 3
    vocab_size = 260

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = [byte + 4 for byte in text.encode("utf-8")]
        if bos:
            ids.insert(0, self.bos_token_id)
        if eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        payload = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if 4 <= token_id < self.vocab_size:
                payload.append(token_id - 4)
            elif not skip_special_tokens:
                payload.extend(f"<|{token_id}|>".encode())
        return payload.decode("utf-8", errors="replace")



class SentencePieceTokenizer(ChatTemplateMixin):
    def __init__(self, model_path: str | Path) -> None:
        import sentencepiece as spm

        self.model_path = Path(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.pad_token_id = self.processor.pad_id()
        self.bos_token_id = self.processor.bos_id()
        self.eos_token_id = self.processor.eos_id()
        self.unk_token_id = self.processor.unk_id()
        self.vocab_size = self.processor.vocab_size()
        if min(self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id) < 0:
            raise ValueError("SentencePiece model must define pad/bos/eos/unk ids")

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = list(self.processor.encode(text, out_type=int))
        if bos:
            ids.insert(0, self.bos_token_id)
        if eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        values = [int(token_id) for token_id in ids]
        if skip_special_tokens:
            special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            values = [token_id for token_id in values if token_id not in special]
        return self.processor.decode(values)


def iter_tokenizer_texts(jsonl_path: str | Path, limit: int | None = None) -> Iterator[str]:
    with Path(jsonl_path).open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if limit is not None and index >= limit:
                break
            text = json.loads(line).get("text")
            if isinstance(text, str) and text.strip():
                yield text


def train_sentencepiece(
    jsonl_path: str | Path,
    output_prefix: str | Path,
    *,
    vocab_size: int = 8192,
    input_sentence_size: int = 1_000_000,
) -> Path:
    import sentencepiece as spm

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_tokenizer_texts(jsonl_path, input_sentence_size),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        normalization_rule_name="identity",
        character_coverage=0.9995,
        byte_fallback=True,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|end|>"],
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return prefix.with_suffix(".model")
