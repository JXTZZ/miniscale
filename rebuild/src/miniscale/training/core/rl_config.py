from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path


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


def option_default(option_type: type[GRPOOptions], name: str) -> object:
    option = next((field for field in fields(option_type) if field.name == name), None)
    if option is None:
        raise KeyError(f"unknown {option_type.__name__} option: {name}")
    if option.default is MISSING:
        raise ValueError(f"option has no default: {name}")
    return option.default


def grpo_option_default(name: str) -> object:
    return option_default(GRPOOptions, name)


def agent_rl_option_default(name: str) -> object:
    return option_default(AgentRLOptions, name)


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
