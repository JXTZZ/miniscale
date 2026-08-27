from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor

from miniscale.integrity import atomic_write_json
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from .common import autocast_context
from .dpo_objective import concatenated_completion_log_probabilities, dpo_batch_metrics
from .sft_evaluation import SFT_GENERATION_PROMPTS


def move_preference_batch(
    batch: dict[str, dict[str, Tensor]], device: torch.device
) -> dict[str, dict[str, Tensor]]:
    return {
        side: {name: value.to(device) for name, value in values.items()}
        for side, values in batch.items()
    }


@torch.no_grad()
def evaluate_dpo(
    policy: MiniScaleForCausalLM,
    reference: MiniScaleForCausalLM,
    batches: Iterable[dict[str, dict[str, Tensor]]],
    device: torch.device,
    *,
    beta: float,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    was_training = policy.training
    policy.eval()
    totals: dict[str, float] = {}
    pairs = chosen_tokens = rejected_tokens = 0
    try:
        for batch_index, cpu_batch in enumerate(batches):
            batch = move_preference_batch(cpu_batch, device)
            pair_count = int(batch["chosen"]["input_ids"].shape[0])
            with autocast_context(device, autocast_dtype):
                policy_chosen, policy_rejected, chosen_counts, rejected_counts = (
                    concatenated_completion_log_probabilities(policy, batch)
                )
                reference_chosen, reference_rejected, _, _ = (
                    concatenated_completion_log_probabilities(reference, batch)
                )
            loss, metrics = dpo_batch_metrics(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta,
            )
            if not bool(torch.isfinite(loss)) or any(
                not bool(torch.isfinite(value)) for value in metrics.values()
            ):
                raise FloatingPointError(f"non-finite DPO validation metric at batch {batch_index}")
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + float(value) * pair_count
            pairs += pair_count
            chosen_tokens += int(chosen_counts.sum())
            rejected_tokens += int(rejected_counts.sum())
    finally:
        policy.train(was_training)
    if not pairs:
        raise ValueError("DPO validation batches contain no pairs")
    result = {f"validation_{name}": total / pairs for name, total in totals.items()}
    result.update({
        "validation_pairs": float(pairs),
        "validation_chosen_tokens": float(chosen_tokens),
        "validation_rejected_tokens": float(rejected_tokens),
    })
    return result


@torch.no_grad()
def run_dpo_generation_evaluation(
    policy: MiniScaleForCausalLM,
    reference: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    output_dir: Path,
    *,
    step: int,
    device: torch.device,
    max_new_tokens: int,
    autocast_dtype: torch.dtype | None,
) -> Path:
    policy_was_training = policy.training
    policy.eval()
    reference.eval()
    samples: list[dict[str, object]] = []
    try:
        for probe in SFT_GENERATION_PROMPTS:
            prompt = tokenizer.format_messages(
                [{"role": "user", "content": probe["prompt"]}], generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt, bos=True)
            prompt_budget = policy.config.max_position_embeddings - max_new_tokens
            if prompt_budget < 1:
                raise ValueError("generation_max_new_tokens must be smaller than the model context length")
            prompt_ids = prompt_ids[-prompt_budget:]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            responses: dict[str, object] = {}
            for name, candidate in (("policy", policy), ("reference", reference)):
                with autocast_context(device, autocast_dtype):
                    generated = candidate.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        top_k=None,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                    )
                completion = generated[0, len(prompt_ids) :].tolist()
                responses[f"{name}_generated_tokens"] = len(completion)
                responses[f"{name}_response"] = tokenizer.decode(completion)
            samples.append({**probe, "prompt_tokens": len(prompt_ids), **responses})
    finally:
        policy.train(policy_was_training)
    target = output_dir / "generations" / f"step_{step:08d}.json"
    atomic_write_json(target, {"stage": "dpo", "step": step, "samples": samples})
    return target
