from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from miniscale.model import MiniScaleForCausalLM
from .runtime import autocast_context
from ..objectives.grpo import grpo_objective, sequence_token_log_probs


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
