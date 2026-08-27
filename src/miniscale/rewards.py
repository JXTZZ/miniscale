from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re


_NUMBER = re.compile(
    r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?(?![\w.])"
)
_TOOL_BLOCK = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class MathReward:
    total: float
    correctness: float
    format_bonus: float
    extra_answer_penalty: float
    length_penalty: float
    exact: bool
    predicted: tuple[float | str, ...]
    expected: tuple[float | str, ...]

    def metrics(self) -> dict[str, float]:
        values = asdict(self)
        return {
            "reward": float(values["total"]),
            "correctness": float(values["correctness"]),
            "format_bonus": float(values["format_bonus"]),
            "extra_answer_penalty": float(values["extra_answer_penalty"]),
            "length_penalty": float(values["length_penalty"]),
            "exact": float(values["exact"]),
        }


def normalize_answer(value: str) -> float | str:
    candidate = value.strip().replace(",", "")
    percent = candidate.endswith("%")
    if percent:
        candidate = candidate[:-1]
    try:
        number = float(candidate)
    except ValueError:
        return candidate.lstrip("+")
    if percent:
        number /= 100.0
    if not math.isfinite(number):
        return candidate
    return round(number, 10)


def extract_numeric_answers(text: str) -> tuple[float | str, ...]:
    """Extract answer candidates while excluding tool-call arguments."""

    visible = _TOOL_BLOCK.sub("", text)
    return tuple(normalize_answer(value) for value in _NUMBER.findall(visible))


def score_math_answer(
    completion: str,
    answer: str | tuple[str, ...],
    *,
    max_length_penalty: float = 0.05,
) -> MathReward:
    """Score numeric answers without rewarding an arbitrary superset of guesses."""

    expected_text = (answer,) if isinstance(answer, str) else answer
    expected = tuple(normalize_answer(value) for value in expected_text if str(value).strip())
    predicted = extract_numeric_answers(completion)
    expected_counts = Counter(expected)
    predicted_counts = Counter(predicted)
    matched = sum(min(count, predicted_counts[value]) for value, count in expected_counts.items())
    correctness = matched / len(expected) if expected else 0.0
    extras = max(len(predicted) - matched, 0)
    extra_penalty = min(0.25, extras * 0.1)
    exact = bool(expected) and predicted_counts == expected_counts
    format_bonus = 0.05 if exact else 0.0
    length_penalty = min(max_length_penalty, max(len(completion) - 160, 0) * 0.00025)
    total = max(-0.5, correctness + format_bonus - extra_penalty - length_penalty)
    return MathReward(
        total=total,
        correctness=correctness,
        format_bonus=format_bonus,
        extra_answer_penalty=extra_penalty,
        length_penalty=length_penalty,
        exact=exact,
        predicted=predicted,
        expected=expected,
    )
