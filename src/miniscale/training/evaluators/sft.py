from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from miniscale.integrity import atomic_write_json
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from ..core.runtime import autocast_context


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
) -> Path:
    was_training = model.training
    model.eval()
    samples: list[dict[str, object]] = []
    try:
        for probe in SFT_GENERATION_PROMPTS:
            prompt = tokenizer.format_messages(
                [{"role": "user", "content": probe["prompt"]}], generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt, bos=True)
            prompt_budget = model.config.max_position_embeddings - max_new_tokens
            if prompt_budget < 1:
                raise ValueError("generation_max_new_tokens must be smaller than the model context length")
            prompt_ids = prompt_ids[-prompt_budget:]
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
            completion = generated[0, len(prompt_ids) :].tolist()
            samples.append({
                **probe,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(completion),
                "response": tokenizer.decode(completion),
            })
    finally:
        model.train(was_training)
    target = output_dir / "generations" / f"step_{step:08d}.json"
    atomic_write_json(target, {"stage": "sft", "step": step, "samples": samples})
    return target
