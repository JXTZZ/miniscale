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
    def encode_batch(self, texts: Sequence[str], *, bos: bool = False, eos: bool = False) -> list[list[int]]: ...
    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str: ...
    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]: ...
    def format_messages(self, messages: Sequence[dict[str, object]], *, generation_prompt: bool = False) -> str: ...
    def encode_sft(
        self,
        messages: Sequence[dict[str, object]],
        *,
        target_mode: str = "reasoning_and_response",
        target_assistant_index: int | None = None,
    ) -> tuple[list[int], list[int]]: ...
    def format_tool_observation(self, observation: str, *, assistant_closed: bool = False) -> str: ...


def _message_content(message: dict[str, object], *, include_reasoning: bool = True) -> str:
    content = message.get("content") or ""
    if (
        not include_reasoning
        and message.get("role") == "assistant"
        and isinstance(content, str)
        and "</think>" in content
    ):
        content = content.split("</think>", 1)[1].lstrip("\n")
    reasoning = message.get("reasoning_content") or ""
    if include_reasoning and message.get("role") == "assistant" and reasoning:
        content = f"<think>\n{str(reasoning).strip()}\n</think>\n\n{content}"
    if message.get("tools"):
        content = f"{content}\n{message['tools']}".strip()
    if message.get("tool_calls"):
        content = f"{content}<tool_call>{message['tool_calls']}</tool_call>"
    return str(content)


def _assistant_target_indices(assistant_count: int, requested: int | None) -> set[int]:
    """Resolve one assistant turn, or all turns, into zero-based indices."""

    if requested is None:
        return set(range(assistant_count))
    index = requested if requested >= 0 else assistant_count + requested
    if not 0 <= index < assistant_count:
        raise ValueError(
            f"target_assistant_index {requested} is invalid for {assistant_count} assistant messages"
        )
    return {index}


class ChatTemplateMixin:
    def format_messages(self, messages: Sequence[dict[str, object]], *, generation_prompt: bool = False) -> str:
        text = "".join(f"<|{message['role']}|>\n{_message_content(message)}<|end|>\n" for message in messages)
        if generation_prompt:
            text += "<|assistant|>\n"
        return text

    def encode_sft(
        self,
        messages: Sequence[dict[str, object]],
        *,
        target_mode: str = "reasoning_and_response",
        target_assistant_index: int | None = None,
    ) -> tuple[list[int], list[int]]:
        if target_mode not in {"reasoning_and_response", "response_only"}:
            raise ValueError("target_mode must be 'reasoning_and_response' or 'response_only'")
        assistant_count = sum(message.get("role") == "assistant" for message in messages)
        targets = _assistant_target_indices(assistant_count, target_assistant_index)
        input_ids = [self.bos_token_id]
        labels = [-100]
        assistant_index = 0
        for message in messages:
            prefix = self.encode(f"<|{message['role']}|>\n")
            is_assistant = message.get("role") == "assistant"
            is_target = is_assistant and assistant_index in targets
            body = self.encode(
                f"{_message_content(message, include_reasoning=target_mode != 'response_only' or not is_target)}"
                "<|end|>\n"
            )
            input_ids.extend(prefix)
            input_ids.extend(body)
            labels.extend([-100] * len(prefix))
            labels.extend(body if is_target else [-100] * len(body))
            if is_assistant:
                assistant_index += 1
        input_ids.append(self.eos_token_id)
        last_is_target = bool(
            messages
            and messages[-1].get("role") == "assistant"
            and assistant_count - 1 in targets
        )
        labels.append(self.eos_token_id if last_is_target else -100)
        return input_ids, labels

    def format_tool_observation(self, observation: str, *, assistant_closed: bool = False) -> str:
        return f"\n<|tool|>\n{observation}<|end|>\n<|assistant|>\n"


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

    def encode_batch(self, texts: Sequence[str], *, bos: bool = False, eos: bool = False) -> list[list[int]]:
        return [self.encode(text, bos=bos, eos=eos) for text in texts]

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        payload = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if 4 <= token_id < self.vocab_size:
                payload.append(token_id - 4)
            elif not skip_special_tokens:
                payload.extend(f"<|{token_id}|>".encode())
        return payload.decode("utf-8", errors="replace")

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        special = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        return [special.get(int(token_id), f"<0x{int(token_id) - 4:02X}>") for token_id in ids]



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

    def encode_batch(self, texts: Sequence[str], *, bos: bool = False, eos: bool = False) -> list[list[int]]:
        return [self.encode(text, bos=bos, eos=eos) for text in texts]

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        values = [int(token_id) for token_id in ids]
        if skip_special_tokens:
            special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            values = [token_id for token_id in values if token_id not in special]
        return self.processor.decode(values)

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        return [self.processor.id_to_piece(int(token_id)) for token_id in ids]


class HuggingFaceTokenizer:
    """Adapter for a local Hugging Face fast-tokenizer directory."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        from transformers import AutoTokenizer

        path = Path(tokenizer_path)
        self.model_path = path.parent if path.is_file() else path
        self.processor = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.pad_token_id = self._required_id("pad_token_id")
        self.bos_token_id = self._required_id("bos_token_id")
        self.eos_token_id = self._required_id("eos_token_id")
        self.unk_token_id = self._required_id("unk_token_id")
        self.vocab_size = len(self.processor)

    def _required_id(self, name: str) -> int:
        value = getattr(self.processor, name)
        if value is None:
            raise ValueError(f"Hugging Face tokenizer must define {name}")
        return int(value)

    @staticmethod
    def _parse_json_value(value: object) -> object:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        ids = list(self.processor.encode(text, add_special_tokens=False))
        if bos and (not ids or ids[0] != self.bos_token_id):
            ids.insert(0, self.bos_token_id)
        if eos and (not ids or ids[-1] != self.eos_token_id):
            ids.append(self.eos_token_id)
        return ids

    def encode_batch(self, texts: Sequence[str], *, bos: bool = False, eos: bool = False) -> list[list[int]]:
        encoded = self.processor(
            list(texts),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        rows: list[list[int]] = []
        for raw_ids in encoded:
            ids = [int(token_id) for token_id in raw_ids]
            if bos and (not ids or ids[0] != self.bos_token_id):
                ids.insert(0, self.bos_token_id)
            if eos and (not ids or ids[-1] != self.eos_token_id):
                ids.append(self.eos_token_id)
            rows.append(ids)
        return rows

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self.processor.decode(
            [int(token_id) for token_id in ids],
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> list[str]:
        tokens = self.processor.convert_ids_to_tokens([int(token_id) for token_id in ids])
        return [str(token) for token in tokens]

    def format_messages(self, messages: Sequence[dict[str, object]], *, generation_prompt: bool = False) -> str:
        prepared: list[dict[str, object]] = []
        tools: object | None = None
        for raw_message in messages:
            message = dict(raw_message)
            raw_tools = message.pop("tools", None)
            if message.get("role") == "system" and raw_tools:
                tools = self._parse_json_value(raw_tools)
            if message.get("tool_calls"):
                message["tool_calls"] = self._parse_json_value(message["tool_calls"])
            message["content"] = message.get("content") or ""
            prepared.append(message)
        kwargs: dict[str, object] = {
            "tokenize": False,
            "add_generation_prompt": generation_prompt,
            "open_thinking": False,
        }
        if tools is not None:
            kwargs["tools"] = tools
        rendered = self.processor.apply_chat_template(prepared, **kwargs)
        if not isinstance(rendered, str):
            raise TypeError("chat template must render text when tokenize=False")
        return rendered

    def encode_sft(
        self,
        messages: Sequence[dict[str, object]],
        *,
        target_mode: str = "reasoning_and_response",
        target_assistant_index: int | None = None,
    ) -> tuple[list[int], list[int]]:
        if target_mode not in {"reasoning_and_response", "response_only"}:
            raise ValueError("target_mode must be 'reasoning_and_response' or 'response_only'")
        assistant_count = sum(message.get("role") == "assistant" for message in messages)
        targets = _assistant_target_indices(assistant_count, target_assistant_index)
        prepared_messages = [dict(message) for message in messages]
        if target_mode == "response_only":
            assistant_index = 0
            for message in prepared_messages:
                if message.get("role") != "assistant":
                    continue
                if assistant_index in targets:
                    message["reasoning_content"] = ""
                    content = message.get("content")
                    if isinstance(content, str) and "</think>" in content:
                        message["content"] = content.split("</think>", 1)[1].lstrip("\n")
                assistant_index += 1

        rendered = self.format_messages(prepared_messages)
        encoded = self.processor(
            rendered,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        input_ids = [int(token_id) for token_id in encoded["input_ids"]]
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        labels = [-100] * len(input_ids)
        assistant_prefix = f"{self.processor.bos_token}assistant\n"
        assistant_end = f"{self.processor.eos_token}\n"
        empty_thinking = "<think>\n\n</think>\n\n"
        character_index = 0
        assistant_index = 0
        while True:
            prefix_start = rendered.find(assistant_prefix, character_index)
            if prefix_start < 0:
                break
            start = prefix_start + len(assistant_prefix)
            end = rendered.find(assistant_end, start)
            if end < 0:
                raise ValueError("chat template produced an unterminated assistant span")
            supervised_start = start
            supervised_end = end + len(assistant_end)
            if assistant_index in targets:
                if target_mode == "response_only":
                    if not rendered.startswith(empty_thinking, start):
                        raise ValueError("chat template response-only thinking prefix changed unexpectedly")
                    supervised_start += len(empty_thinking)
                for token_index, (token_start, token_end) in enumerate(offsets):
                    if token_end > supervised_start and token_start < supervised_end:
                        labels[token_index] = input_ids[token_index]
            assistant_index += 1
            character_index = supervised_end
        if assistant_index != assistant_count:
            raise ValueError(
                "chat template assistant span count does not match input messages: "
                f"rendered={assistant_index}, messages={assistant_count}"
            )
        return input_ids, labels

    def format_tool_observation(self, observation: str, *, assistant_closed: bool = False) -> str:
        close_assistant = "" if assistant_closed else f"{self.processor.eos_token}\n"
        return (
            f"{close_assistant}<|im_start|>user\n<tool_response>\n{observation}\n"
            "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )


def load_tokenizer(tokenizer_path: str | Path) -> Tokenizer:
    """Load either a SentencePiece model or a Hugging Face tokenizer directory."""

    path = Path(tokenizer_path)
    if path.is_dir() or path.name == "tokenizer.json":
        return HuggingFaceTokenizer(path)
    if path.suffix == ".model":
        return SentencePieceTokenizer(path)
    raise ValueError(f"unsupported tokenizer path: {path}")


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
