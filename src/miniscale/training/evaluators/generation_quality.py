from __future__ import annotations

from collections import Counter
import ast
import json
from pathlib import Path
import re
from typing import Sequence


SFT_GENERATION_METRICS_VERSION = "sft_generation_quality_v1"


def repeated_ngram_fraction(token_ids: Sequence[int], n: int = 4) -> float:
    if n < 1:
        raise ValueError("n must be positive")
    total = len(token_ids) - n + 1
    if total <= 0:
        return 0.0
    counts = Counter(tuple(token_ids[index : index + n]) for index in range(total))
    return sum(count - 1 for count in counts.values()) / total


def distinct_ngram_fraction(token_ids: Sequence[int], n: int) -> float:
    total = len(token_ids) - n + 1
    if total <= 0:
        return 1.0
    return len({tuple(token_ids[index : index + n]) for index in range(total)}) / total


def _normalized_text(text: str) -> str:
    return "".join(text.casefold().split())


def _has_repeated_sentence(text: str) -> bool:
    sentences = [
        _normalized_text(sentence)
        for sentence in re.split(r"[。！？!?；;\n]+", text)
        if len(_normalized_text(sentence)) >= 4
    ]
    return any(count >= 2 for count in Counter(sentences).values())


def _python_source(response: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else response.strip()


def task_rule_passes(probe: dict[str, object], response: str, generated_tokens: int) -> bool | None:
    has_rule = any(
        key in probe for key in ("exact", "required", "forbidden", "regex", "max_tokens", "python_ast")
    )
    if not has_rule:
        return None
    normalized = _normalized_text(response)
    exact = probe.get("exact")
    if isinstance(exact, str) and normalized != _normalized_text(exact):
        return False
    required = probe.get("required", [])
    if isinstance(required, list) and any(
        isinstance(value, str) and _normalized_text(value) not in normalized for value in required
    ):
        return False
    forbidden = probe.get("forbidden", [])
    if isinstance(forbidden, list) and any(
        isinstance(value, str) and _normalized_text(value) in normalized for value in forbidden
    ):
        return False
    pattern = probe.get("regex")
    if isinstance(pattern, str) and re.search(pattern, response, flags=re.IGNORECASE | re.DOTALL) is None:
        return False
    max_tokens = probe.get("max_tokens")
    if isinstance(max_tokens, int) and generated_tokens > max_tokens:
        return False
    if probe.get("python_ast") is True:
        try:
            ast.parse(_python_source(response))
        except (SyntaxError, ValueError):
            return False
    return True


def score_generation(
    probe: dict[str, object],
    response: str,
    token_ids: Sequence[int],
    *,
    finish_reason: str,
) -> dict[str, object]:
    repeated = repeated_ngram_fraction(token_ids, 4)
    prompt = str(probe.get("prompt", ""))
    prompt_normalized = _normalized_text(prompt)
    response_normalized = _normalized_text(response)
    special_leak = any(marker in response for marker in ("<|im_start|>", "<|im_end|>", "<|end|>"))
    think_leak = "<think>" in response or "</think>" in response
    return {
        "finish_reason": finish_reason,
        "eos": finish_reason == "eos",
        "max_length": finish_reason == "max_tokens",
        "repeated_4gram_fraction": repeated,
        "loop": repeated > 0.20 or _has_repeated_sentence(response),
        "distinct_1": distinct_ngram_fraction(token_ids, 1),
        "distinct_2": distinct_ngram_fraction(token_ids, 2),
        "distinct_3": distinct_ngram_fraction(token_ids, 3),
        "prompt_echo": bool(
            len(prompt_normalized) >= 8 and prompt_normalized in response_normalized
        ),
        "special_token_leak": special_leak,
        "think_leak": think_leak,
        "task_pass": task_rule_passes(probe, response, len(token_ids)),
    }


def summarize_generations(samples: Sequence[dict[str, object]]) -> dict[str, float]:
    if not samples:
        raise ValueError("generation evaluation requires at least one sample")
    rule_samples = [sample for sample in samples if sample.get("task_pass") is not None]

    def rate(name: str, rows: Sequence[dict[str, object]] = samples) -> float:
        return sum(bool(row.get(name)) for row in rows) / len(rows) if rows else 0.0

    return {
        "generation_eos_rate": rate("eos"),
        "generation_max_length_rate": rate("max_length"),
        "generation_loop_rate": rate("loop"),
        "generation_repeated_4gram_fraction": sum(
            float(sample["repeated_4gram_fraction"]) for sample in samples
        ) / len(samples),
        "generation_task_pass_rate": rate("task_pass", rule_samples),
        "generation_prompt_echo_rate": rate("prompt_echo"),
        "generation_special_token_leak_rate": rate("special_token_leak"),
        "generation_think_leak_rate": rate("think_leak"),
        "generation_average_tokens": sum(int(sample["generated_tokens"]) for sample in samples)
        / len(samples),
    }


def generation_quality_score(summary: dict[str, float]) -> float:
    return (
        3.0 * summary["generation_task_pass_rate"]
        + summary["generation_eos_rate"]
        - 2.0 * summary["generation_loop_rate"]
        - summary["generation_max_length_rate"]
        - 2.0 * summary["generation_special_token_leak_rate"]
        - 2.0 * summary["generation_think_leak_rate"]
    )


def load_generation_suite(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    probes: list[dict[str, object]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"invalid generation probe at {source}:{line_number}")
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"generation probe requires a prompt at {source}:{line_number}")
            probes.append(row)
    if not probes:
        raise ValueError(f"generation suite is empty: {source}")
    identifiers = [str(probe["id"]) for probe in probes]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"generation suite contains duplicate ids: {source}")
    return probes
