from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from miniscale.model import MiniScaleForCausalLM


def sequence_token_log_probs(
    model: MiniScaleForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    targets = input_ids[:, 1:]
    return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def normalize_group_rewards(rewards: Tensor, group_size: int, eps: float = 1e-4) -> Tensor:
    if rewards.numel() % group_size:
        raise ValueError("reward count must be divisible by group_size")
    groups = rewards.float().view(-1, group_size)
    centered = groups - groups.mean(dim=1, keepdim=True)
    return (centered / (groups.std(dim=1, keepdim=True, unbiased=False) + eps)).view(-1)


def grpo_objective(
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    action_mask: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float,
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sequence-balanced GRPO objective over completion/action tokens only."""

    if policy_log_probs.shape != action_mask.shape:
        raise ValueError("log probabilities and action_mask must have identical shapes")
    ratio = torch.exp((policy_log_probs - old_log_probs).clamp(-20, 20))
    token_advantages = advantages.float()[:, None]
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
    log_ratio = (reference_log_probs - policy_log_probs).clamp(-20, 20)
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    mask = action_mask.float()
    token_counts = mask.sum(dim=1)
    valid = token_counts > 0
    if not bool(valid.any()):
        raise ValueError("GRPO batch contains no action tokens")

    def sequence_mean(values: Tensor) -> Tensor:
        per_sequence = (values * mask).sum(dim=1) / token_counts.clamp_min(1)
        return per_sequence[valid].mean()

    clip_events = ((ratio - 1.0).abs() > clip_epsilon).float()
    loss = sequence_mean(policy_loss + beta * kl)
    return loss, {
        "policy_loss": sequence_mean(policy_loss),
        "kl": sequence_mean(kl),
        "clip_fraction": sequence_mean(clip_events),
    }
