from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from miniscale.model import MiniScaleForCausalLM


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


def concatenated_completion_log_probabilities(
    model: MiniScaleForCausalLM,
    batch: dict[str, dict[str, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Score chosen and rejected in one model forward pass."""

    chosen = batch["chosen"]
    rejected = batch["rejected"]
    if chosen["input_ids"].shape != rejected["input_ids"].shape:
        raise ValueError("chosen and rejected must share a padded batch shape")
    count = chosen["input_ids"].shape[0]
    combined = {
        name: torch.cat((chosen[name], rejected[name]), dim=0)
        for name in ("input_ids", "labels", "attention_mask")
    }
    logps = completion_log_probability(model, combined)
    target_counts = (combined["labels"][:, 1:] != -100).sum(-1)
    return logps[:count], logps[count:], target_counts[:count], target_counts[count:]


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
