from __future__ import annotations

from collections.abc import Iterable, Sequence


class ByteTokenizer:
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

    def format_messages(self, messages: Sequence[dict[str, str]], *, generation_prompt: bool = False) -> str:
        text = "".join(
            f"<|{message['role']}|>\n{message['content']}<|end|>\n"
            for message in messages
        )
        if generation_prompt:
            text += "<|assistant|>\n"
        return text

    def encode_sft(self, messages: Sequence[dict[str, str]]) -> tuple[list[int], list[int]]:
        input_ids = [self.bos_token_id]
        labels = [-100]
        for message in messages:
            prefix = self.encode(f"<|{message['role']}|>\n")
            body = self.encode(f"{message['content']}<|end|>\n")
            input_ids.extend(prefix)
            input_ids.extend(body)
            labels.extend([-100] * len(prefix))
            labels.extend(body if message["role"] == "assistant" else [-100] * len(body))
        input_ids.append(self.eos_token_id)
        labels.append(self.eos_token_id if messages and messages[-1]["role"] == "assistant" else -100)
        return input_ids, labels
