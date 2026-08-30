from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path

import torch
from torch import Tensor

from miniscale.integrity import atomic_write_json
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from ..core.runtime import autocast_context
from .generation_quality import (
    SFT_GENERATION_METRICS_VERSION,
    load_generation_suite,
    score_generation,
    summarize_generations,
)


SFT_GENERATION_PROMPTS: tuple[dict[str, str], ...] = (
    {"name": "chinese_qa", "language": "zh", "prompt": "中国的首都是哪里？"},
    {"name": "english_qa", "language": "en", "prompt": "What is the capital of France?"},
    {"name": "code", "language": "python", "prompt": "写一个返回斐波那契数列第 n 项的 Python 函数。"},
    {"name": "identity", "language": "zh", "prompt": "请介绍一下你自己。"},
)


@torch.no_grad()
def evaluate_sft(
    model: MiniScaleForCausalLM,
    batches: Iterable[dict[str, Tensor]],
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None,
) -> tuple[float, float, int]:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    targets = correct = 0
    try:
        for batch_index, batch in enumerate(batches):
            device_batch = {name: value.to(device) for name, value in batch.items()}
            with autocast_context(device, autocast_dtype):
                result = model(**device_batch)
            if result.loss is None or not bool(torch.isfinite(result.loss)):
                raise FloatingPointError(f"non-finite SFT validation loss at batch {batch_index}")
            labels = device_batch["labels"][:, 1:]
            mask = labels.ne(-100)
            target_count = int(mask.sum())
            if not target_count:
                continue
            predictions = result.logits[:, :-1].argmax(dim=-1)
            correct += int((predictions.eq(labels) & mask).sum())
            targets += target_count
            weighted_loss += float(result.loss) * target_count
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("SFT validation batches contain no supervised targets")
    return weighted_loss / targets, correct / targets, targets


@torch.no_grad()
def run_sft_generation_evaluation(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    output_dir: Path,
    *,
    step: int,
    device: torch.device,
    max_new_tokens: int,
    autocast_dtype: torch.dtype | None,
    suite_path: str | Path | None = None,
) -> tuple[Path, dict[str, float]]:
    probes = load_generation_suite(suite_path) if suite_path is not None else list(SFT_GENERATION_PROMPTS)
    samples, summary = evaluate_sft_generation_quality(
        model,
        tokenizer,
        probes,
        device=device,
        max_new_tokens=max_new_tokens,
        autocast_dtype=autocast_dtype,
    )
    target = output_dir / "generations" / f"step_{step:08d}.json"
    atomic_write_json(target, {
        "schema_version": 2,
        "metrics_version": SFT_GENERATION_METRICS_VERSION,
        "stage": "sft",
        "step": step,
        "suite": str(Path(suite_path).resolve()) if suite_path is not None else "builtin_smoke_v1",
        "decoding": {
            "profile": "raw_greedy",
            "temperature": 0.0,
            "top_k": None,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
        },
        "summary": summary,
        "samples": samples,
    })
    return target, summary


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
