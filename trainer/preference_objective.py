"""DPO completion scoring, preference loss, and validation."""

from __future__ import annotations

from collections.abc import Iterable
import math
from pathlib import Path

from torch import Tensor
import torch
import torch.nn.functional as F

from dataset.file_utils import atomic_write_json
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.eval_utils import SFT_GENERATION_PROMPTS
from trainer.trainer_utils import autocast_context


def checked_dpo_metrics(metrics: dict[str, Tensor], context: str) -> dict[str, float]:
    values = torch.stack(tuple(metrics.values())).tolist()
    if not all(map(math.isfinite, values)):
        raise FloatingPointError(f"non-finite DPO metric at {context}")
    return dict(zip(metrics, values, strict=True))


@torch.no_grad()
def evaluate_dpo(
    policy: MiniScaleForCausalLM,
    reference: MiniScaleForCausalLM,
    batches: Iterable[dict[str, Tensor]],
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
            batch = {name: value.to(device) for name, value in cpu_batch.items()}
            pair_count = cpu_batch["input_ids"].shape[0] // 2
            with autocast_context(device, autocast_dtype):
                policy_chosen, policy_rejected = completion_log_probability(policy, batch).chunk(2)
                reference_chosen, reference_rejected = completion_log_probability(reference, batch).chunk(2)
            _, metrics = dpo_batch_metrics(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta,
            )
            for name, value in checked_dpo_metrics(metrics, f"validation batch {batch_index}").items():
                totals[name] = totals.get(name, 0.0) + value * pair_count
            pairs += pair_count
            chosen, rejected = cpu_batch["labels"][:, 1:].ne(-100).chunk(2)
            chosen_tokens += int(chosen.sum())
            rejected_tokens += int(rejected.sum())
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


def completion_log_probability(
    model: MiniScaleForCausalLM,
    batch: dict[str, Tensor],
) -> Tensor:
    """Sum supervised completion log-probabilities in FP32."""

    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    if not bool(mask.any()):
        raise ValueError("DPO batch contains no supervised completion tokens")
    targets = labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits.float(), dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * mask).sum(-1)


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    if beta <= 0:
        raise ValueError("beta must be positive")
    reward_margin = beta * (
        (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
    )
    return -F.logsigmoid(reward_margin).mean(), (reward_margin > 0).float().mean()


def dpo_batch_metrics(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    loss, reward_accuracy = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta,
    )
    chosen_reward = beta * (policy_chosen - reference_chosen)
    rejected_reward = beta * (policy_rejected - reference_rejected)
    return loss, {
        "loss": loss.detach(),
        "reward_accuracy": reward_accuracy.detach(),
        "reward_margin": (chosen_reward - rejected_reward).mean().detach(),
        "chosen_reward": chosen_reward.mean().detach(),
        "rejected_reward": rejected_reward.mean().detach(),
        "policy_chosen_logp": policy_chosen.mean().detach(),
        "policy_rejected_logp": policy_rejected.mean().detach(),
        "policy_logp_margin": (policy_chosen - policy_rejected).mean().detach(),
        "reference_logp_margin": (reference_chosen - reference_rejected).mean().detach(),
        "policy_preference_accuracy": (policy_chosen > policy_rejected).float().mean().detach(),
    }
