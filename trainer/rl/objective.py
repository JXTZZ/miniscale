"""Options and policy updates shared by GRPO and Agent training."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
import torch
import torch.nn.functional as F

from model.model_miniscale import MiniScaleForCausalLM
from trainer.runtime.torch_utils import autocast_context


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
    loss_type: str = "grpo",
    cispo_max_ratio: float = 5.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sequence-balanced GRPO objective over completion/action tokens only."""

    if policy_log_probs.shape != action_mask.shape:
        raise ValueError("log probabilities and action_mask must have identical shapes")
    ratio = torch.exp((policy_log_probs - old_log_probs).clamp(-20, 20))
    token_advantages = advantages.float()[:, None]
    if loss_type == "cispo":
        weight = ratio.detach().clamp(max=cispo_max_ratio)
        policy_loss = -weight * token_advantages * policy_log_probs
        clip_events = ratio > cispo_max_ratio
    else:
        clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        policy_loss = -torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
        clip_events = (ratio - 1.0).abs() > clip_epsilon
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

    loss = sequence_mean(policy_loss + beta * kl)
    return loss, {
        "policy_loss": sequence_mean(policy_loss),
        "kl": sequence_mean(kl),
        "clip_fraction": sequence_mean(clip_events.float()),
    }


@dataclass(frozen=True, slots=True)
class PolicyUpdate:
    metrics: dict[str, float]
    grad_norm: float
    optimizer_steps: int


def optimize_policy_epochs(
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    action_mask: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    advantages: Tensor,
    policy_epochs: int,
    clip_epsilon: float,
    beta: float,
    grad_clip: float,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    step: int,
    loss_type: str = "grpo",
    cispo_max_ratio: float = 5.0,
) -> PolicyUpdate:
    """Apply repeated clipped updates to one immutable on-policy rollout batch."""

    metrics: dict[str, float] = {}
    grad_norm_value = 0.0
    model.train()
    for epoch in range(1, policy_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, autocast_dtype):
            policy_log_probs = sequence_token_log_probs(model, input_ids, attention_mask)
            loss, stats = grpo_objective(
                policy_log_probs,
                old_log_probs,
                reference_log_probs,
                action_mask,
                advantages,
                clip_epsilon=clip_epsilon,
                beta=beta,
                loss_type=loss_type,
                cispo_max_ratio=cispo_max_ratio,
            )
        if not bool(torch.isfinite(loss)) or any(
            not bool(torch.isfinite(value)) for value in stats.values()
        ):
            raise FloatingPointError(f"non-finite GRPO metric at step {step}, epoch {epoch}")
        loss.backward()
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip, error_if_nonfinite=True
            )
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
            raise FloatingPointError(
                f"non-finite GRPO gradient norm at step {step}, epoch {epoch}"
            ) from error
        optimizer.step()
        scheduler.step()
        grad_norm_value = float(grad_norm)
        metrics = {
            "loss": float(loss.detach()),
            **{name: float(value.detach()) for name, value in stats.items()},
        }
    return PolicyUpdate(metrics, grad_norm_value, policy_epochs)


def compute_reference_log_probs(
    reference: MiniScaleForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    reference_device: torch.device,
    policy_device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> Tensor:
    with torch.no_grad(), autocast_context(
        reference_device, autocast_dtype if reference_device.type == "cuda" else None
    ):
        values = sequence_token_log_probs(
            reference, input_ids.to(reference_device), attention_mask.to(reference_device)
        )
    return values.to(policy_device)
