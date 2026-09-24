from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import unicodedata


SFT_QUALITY_POLICY_VERSION = "sft_quality_policy_v1"
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


@dataclass(frozen=True, slots=True)
class SFTQualityPolicy:
    version: str = SFT_QUALITY_POLICY_VERSION
    seed: int = 42
    max_targets: int = 250_000
    max_targets_per_conversation: int = 3
    max_prompt_occurrences: int = 3
    max_tool_prompt_occurrences: int = 20
    max_response_occurrences: int = 5
    max_short_response_occurrences: int = 20
    max_identity_response_occurrences: int = 20
    short_response_characters: int = 16
    max_target_tokens: int = 736
    reject_severe_repetition: bool = True
    category_max_targets: dict[str, int] = field(default_factory=lambda: {
        "knowledge_chat": 125_000,
        "writing_translation": 37_500,
        "math_reasoning": 37_500,
        "code": 30_000,
        "tool": 12_500,
        "safety": 5_000,
        "identity": 2_500,
    })

    def validate(self) -> None:
        if self.version != SFT_QUALITY_POLICY_VERSION:
            raise ValueError(f"unsupported SFT quality policy version: {self.version}")
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name not in {"version", "seed", "reject_severe_repetition", "category_max_targets"}
        }
        if any(not isinstance(value, int) or value < 1 for value in positive.values()):
            raise ValueError("SFT quality policy numeric limits must be positive integers")
        if any(not name or not isinstance(limit, int) or limit < 1 for name, limit in self.category_max_targets.items()):
            raise ValueError("SFT quality category limits must have names and positive integer limits")

    @classmethod
    def load(cls, value: str | Path | dict[str, object]) -> SFTQualityPolicy:
        if isinstance(value, dict):
            payload = value
        else:
            payload = json.loads(Path(value).read_text(encoding="utf-8"))
        policy = cls(**payload)
        policy.validate()
        return policy


def normalize_sft_text(text: str, *, code: bool = False) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    if code:
        return "\n".join(line.rstrip() for line in normalized.strip().splitlines())
    normalized = normalized.casefold()
    return "".join(character for character in normalized if character.isalnum())


def stable_text_digest(text: str, *, code: bool = False) -> int:
    normalized = normalize_sft_text(text, code=code)
    return int.from_bytes(hashlib.blake2b(normalized.encode(), digest_size=8).digest(), "big")


def repeated_character_ngram_fraction(text: str, n: int = 4) -> float:
    normalized = normalize_sft_text(text)
    total = len(normalized) - n + 1
    if total <= 0:
        return 0.0
    counts = Counter(normalized[index : index + n] for index in range(total))
    return sum(count - 1 for count in counts.values()) / total


def repeated_sentence_coverage(text: str) -> float:
    sentences = [
        normalize_sft_text(sentence)
        for sentence in re.split(r"[。！？!?；;\n]+", text)
        if len(normalize_sft_text(sentence)) >= 4
    ]
    if not sentences:
        return 0.0
    counts = Counter(sentences)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(sentences)


def has_consecutive_sentence_loop(text: str, repeats: int = 3) -> bool:
    sentences = [
        normalize_sft_text(sentence)
        for sentence in re.split(r"[。！？!?；;\n]+", text)
        if len(normalize_sft_text(sentence)) >= 4
    ]
    return any(
        len(set(sentences[index : index + repeats])) == 1
        for index in range(len(sentences) - repeats + 1)
    )


def looks_like_code(message: dict[str, object]) -> bool:
    content = message.get("content")
    if not isinstance(content, str):
        return False
    return "```" in content or bool(re.search(r"(^|\n)\s*(def |class |function |import )", content))


def response_has_severe_repetition(message: dict[str, object]) -> bool:
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    if looks_like_code(message):
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        counts = Counter(lines)
        repeated_lines = sum(count for line, count in counts.items() if count >= 4 and len(line) >= 4)
        return bool(lines) and repeated_lines / len(lines) > 0.40
    if has_consecutive_sentence_loop(content):
        return True
    if repeated_sentence_coverage(content) > 0.25:
        return True
    normalized_length = len(normalize_sft_text(content))
    repeated_4gram_fraction = repeated_character_ngram_fraction(content)
    # Short phrase loops often contain no sentence separator, so sentence-loop
    # detection alone misses replies such as "A、A、A、A、A".  Keep a stricter
    # threshold for short text and the broader threshold for substantive text.
    return (
        normalized_length >= 16 and repeated_4gram_fraction > 0.45
    ) or (
        normalized_length >= 64 and repeated_4gram_fraction > 0.20
    )


def target_category(messages: list[dict[str, object]], position: int) -> str:
    message = messages[position]
    if message.get("tool_calls"):
        return "tool"
    if looks_like_code(message):
        return "code"
    latest_user = next(
        (
            str(previous.get("content") or "")
            for previous in reversed(messages[:position])
            if previous.get("role") == "user"
        ),
        "",
    ).casefold()
    if any(term in latest_user for term in ("你是谁", "介绍一下你", "who are you", "introduce yourself")):
        return "identity"
    if any(term in latest_user for term in ("翻译", "translate", "改写", "rewrite", "文章", "write")):
        return "writing_translation"
    if any(term in latest_user for term in ("计算", "证明", "数学", "calculate", "solve", "equation")):
        return "math_reasoning"
    if any(term in latest_user for term in ("拒绝", "危险", "违法", "password", "窃取")):
        return "safety"
    return "knowledge_chat"
