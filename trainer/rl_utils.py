"""Options and policy updates shared by GRPO and Agent training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

from torch import Tensor
import torch
import torch.nn.functional as F

from model.model_miniscale import MiniScaleForCausalLM
from trainer.cli import add_optimizer_arguments, add_wandb_arguments, option_default
from trainer.trainer_utils import autocast_context


@dataclass(slots=True)
class GRPOOptions:
    """Resolved options for verifiable online GRPO."""

    steps: int = 20
    batch_size: int = 1
    group_size: int = 4
    max_new_tokens: int = 128
    learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    warmup_steps: int = 20
    policy_epochs: int = 2
    clip_epsilon: float = 0.2
    beta: float = 0.01
    temperature: float = 1.0
    top_k: int | None = 50
    grad_clip: float = 1.0
    precision: str = "fp32"
    validation_fraction: float = 0.05
    validation_every: int = 100
    validation_prompts: int = 100
    save_every: int = 200
    keep_last_checkpoints: int = 3
    seed: int = 42
    device: str = "auto"
    reference_device: str = "same"
    data_limit: int | None = None
    log_every: int = 10
    wandb_enabled: bool = False
    wandb_project: str = "MiniScale"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str = "online"
    wandb_retry_every_steps: int = 200
    resume_from: str | Path | None = None


@dataclass(slots=True)
class AgentRLOptions(GRPOOptions):
    """GRPO options plus bounded multi-turn tool execution."""

    max_turns: int = 6


def validate_rl_options(options: GRPOOptions) -> None:
    if options.steps < 1 or options.batch_size < 1 or options.group_size < 2:
        raise ValueError("steps and batch_size must be positive; group_size must be at least 2")
    if options.max_new_tokens < 1 or options.policy_epochs < 1:
        raise ValueError("max_new_tokens and policy_epochs must be positive")
    if options.learning_rate <= 0 or not 0 <= options.min_learning_rate <= options.learning_rate:
        raise ValueError("learning rates must satisfy 0 <= min_learning_rate <= learning_rate")
    if options.weight_decay < 0 or options.adam_eps <= 0 or options.grad_clip <= 0:
        raise ValueError("weight_decay must be non-negative; adam_eps and grad_clip must be positive")
    if not 0 <= options.adam_beta1 < 1 or not 0 <= options.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if not 0 < options.clip_epsilon < 1 or options.beta < 0:
        raise ValueError("clip_epsilon must be in (0, 1) and beta must be non-negative")
    if options.temperature < 0 or options.top_k is not None and options.top_k < 1:
        raise ValueError("temperature must be non-negative and top_k must be positive when set")
    if options.warmup_steps < 0 or options.log_every < 1:
        raise ValueError("warmup_steps must be non-negative and log_every must be positive")
    if not 0 <= options.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if options.validation_every < 1 or options.validation_prompts < 1:
        raise ValueError("validation settings must be positive")
    if options.save_every < 0 or options.keep_last_checkpoints < 1:
        raise ValueError("save_every must be non-negative and keep_last_checkpoints must be positive")
    if options.reference_device not in {"same", "cpu"}:
        raise ValueError("reference_device must be 'same' or 'cpu'")
    if options.data_limit is not None and options.data_limit < 1:
        raise ValueError("data_limit must be positive when set")
    if options.wandb_retry_every_steps < 1:
        raise ValueError("wandb retry interval must be positive")
    if isinstance(options, AgentRLOptions) and options.max_turns < 1:
        raise ValueError("max_turns must be positive")


def resolved_rl_options(options: GRPOOptions) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(options).items()
        if name != "resume_from"
    }


def rl_resume_options(options: GRPOOptions) -> dict[str, object]:
    """Return only options that affect the reproducible training trajectory."""

    operational = {
        "log_every",
        "save_every",
        "keep_last_checkpoints",
        "wandb_enabled",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_run_id",
        "wandb_mode",
        "wandb_retry_every_steps",
    }
    return {name: value for name, value in resolved_rl_options(options).items() if name not in operational}


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


def add_rl_arguments(command: argparse.ArgumentParser, option_type: type[GRPOOptions]) -> None:
    default = partial(option_default, option_type)
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="initialize from the previous stage")
    source.add_argument("--resume", type=Path, help="resume an exact full-stage checkpoint")
    command.add_argument("--group-size", type=int, default=default("group_size"))
    command.add_argument("--max-new-tokens", type=int, default=default("max_new_tokens"))
    command.add_argument("--policy-epochs", type=int, default=default("policy_epochs"))
    add_optimizer_arguments(command, option_type)
    command.add_argument("--clip-epsilon", type=float, default=default("clip_epsilon"))
    command.add_argument("--beta", type=float, default=default("beta"))
    command.add_argument("--temperature", type=float, default=default("temperature"))
    command.add_argument(
        "--top-k", type=int, default=default("top_k"), help="use 0 to disable top-k sampling"
    )
    command.add_argument(
        "--precision", choices=("fp32", "bf16"), default=default("precision")
    )
    command.add_argument(
        "--reference-device", choices=("same", "cpu"), default=default("reference_device"),
        help="use cpu to reduce GPU memory at the cost of slower reference scoring",
    )
    command.add_argument(
        "--validation-fraction", type=float, default=default("validation_fraction")
    )
    command.add_argument("--validation-data", type=Path)
    command.add_argument("--validation-every", type=int, default=default("validation_every"))
    command.add_argument("--validation-prompts", type=int, default=default("validation_prompts"))
    command.add_argument("--save-every", type=int, default=default("save_every"))
    command.add_argument("--keep-last", type=int, default=default("keep_last_checkpoints"))
    command.add_argument("--data-limit", type=int, default=default("data_limit"))
    command.add_argument("--seed", type=int, default=default("seed"))
    add_wandb_arguments(command, option_type)
