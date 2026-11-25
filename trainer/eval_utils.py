"""Shared generation probes and quality metrics."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import torch

from dataset.file_utils import atomic_write_json
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.trainer_utils import autocast_context


SFT_GENERATION_PROMPTS: tuple[dict[str, str], ...] = (
    {"name": "chinese_qa", "language": "zh", "prompt": "中国的首都是哪里？"},
    {"name": "english_qa", "language": "en", "prompt": "What is the capital of France?"},
    {"name": "code", "language": "python", "prompt": "写一个返回斐波那契数列第 n 项的 Python 函数。"},
    {"name": "identity", "language": "zh", "prompt": "请介绍一下你自己。"},
)


@torch.no_grad()
def evaluate_sft_generation_quality(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    probes: list[dict[str, object]],
    *,
    device: torch.device,
    max_new_tokens: int,
    autocast_dtype: torch.dtype | None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    was_training = model.training
    model.eval()
    samples: list[dict[str, object]] = []
    try:
        for probe in probes:
            prompt = tokenizer.format_messages(
                [{"role": "user", "content": probe["prompt"]}], generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt, bos=True)
            prompt_budget = model.config.max_position_embeddings - max_new_tokens
            if prompt_budget < 1:
                raise ValueError("generation_max_new_tokens must be smaller than the model context length")
            prompt_ids = prompt_ids[-prompt_budget:]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            probe_seed = int.from_bytes(
                hashlib.blake2b(str(probe.get("id", probe["prompt"])).encode(), digest_size=8).digest(),
                "big",
            )
            generator = torch.Generator(device=device).manual_seed(seed ^ probe_seed)
            with autocast_context(device, autocast_dtype):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=temperature > 0,
                    generator=generator,
                )
            completion = generated[0, len(prompt_ids) :].tolist()
            if tokenizer.eos_token_id in completion:
                eos_index = completion.index(tokenizer.eos_token_id)
                response_ids = completion[:eos_index]
                finish_reason = "eos"
            else:
                response_ids = completion
                finish_reason = "max_tokens"
            response = tokenizer.decode(response_ids)
            quality = score_generation(
                probe,
                response,
                response_ids,
                finish_reason=finish_reason,
            )
            samples.append({
                **probe,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(response_ids),
                "response": response,
                **quality,
            })
    finally:
        model.train(was_training)
    summary = summarize_generations(samples)
    return samples, summary


DEFAULT_SFT_GENERATION_SUITE = Path(__file__).resolve().parents[1] / "dataset/eval/sft_generation_v1.jsonl"
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


GENERATION_EVAL_PROMPTS: tuple[dict[str, str], ...] = (
    {"name": "chinese", "language": "zh", "prompt": "人工智能的发展将会"},
    {"name": "english", "language": "en", "prompt": "The future of artificial intelligence is"},
    {
        "name": "code",
        "language": "python",
        "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    },
)


@torch.no_grad()
def evaluate_pretrain_generation(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    step: int,
    device: torch.device,
    max_new_tokens: int,
    autocast_dtype: torch.dtype | None = None,
) -> Path:
    """Generate fixed multilingual probes with deterministic greedy decoding."""

    was_training = model.training
    model.eval()
    samples: list[dict[str, object]] = []
    try:
        for probe in GENERATION_EVAL_PROMPTS:
            prompt_ids = tokenizer.encode(probe["prompt"], bos=True)
            if len(prompt_ids) >= model.config.max_position_embeddings:
                prompt_ids = prompt_ids[-(model.config.max_position_embeddings - 1) :]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with autocast_context(device, autocast_dtype):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    top_k=None,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            completion_ids = generated[0, len(prompt_ids) :].tolist()
            samples.append({
                **probe,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(completion_ids),
                "response": tokenizer.decode(completion_ids),
            })
    finally:
        model.train(was_training)

    target = Path(output_dir) / "generations" / f"step_{step:08d}.json"
    atomic_write_json(target, {
        "stage": "pretrain",
        "step": step,
        "decoding": {"do_sample": False, "strategy": "greedy", "temperature": 0.0},
        "samples": samples,
    })
    return target
