from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

from miniscale.integrity import tokenizer_identity
from miniscale.model import MiniScaleForCausalLM
from miniscale.preference_data import (
    DPO_DATA_ORDER_VERSION,
    DPO_PAIR_FORMAT_VERSION,
    DPO_TRUNCATION_VERSION,
    PreferenceCorpusIndex,
)
from miniscale.tokenizer import Tokenizer


DPO_RESUME_SIGNATURE_VERSION = 1
DPO_IMPLEMENTATION_VERSION = 2
DPO_OBJECTIVE_VERSION = "sigmoid_sum_completion_logp_v1"
DPO_OPTIMIZER_GROUPING = "matrix_weights_decay_norm_embedding_no_decay_v1"


@dataclass(slots=True)
class DPOOptions:
    """Resolved options for production direct preference optimization."""

    steps: int
    batch_size: int = 1
    max_length: int | None = None
    min_context_tokens: int = 32
    target_mode: str | None = None
    learning_rate: float = 5e-6
    min_learning_rate: float = 5e-7
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    beta: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 50
    precision: str = "fp32"
    gradient_accumulation_steps: int = 16
    validation_fraction: float = 0.05
    validation_every: int = 100
    validation_batches: int = 100
    save_every: int = 200
    keep_last_checkpoints: int = 3
    generation_every: int = 500
    generation_max_new_tokens: int = 96
    deduplicate_exact: bool = True
    log_every: int = 10
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    wandb_enabled: bool = False
    wandb_project: str = "MiniScale"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str = "online"
    wandb_retry_every_steps: int = 200
    resume_from: str | Path | None = None


def dpo_option_default(name: str) -> object:
    option = next((field for field in fields(DPOOptions) if field.name == name), None)
    if option is None:
        raise KeyError(f"unknown DPO option: {name}")
    if option.default is MISSING:
        raise ValueError(f"DPO option has no default: {name}")
    return option.default


def validate_dpo_options(
    options: DPOOptions,
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    *,
    target_mode: str,
) -> int:
    if options.steps < 1 or options.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    max_length = options.max_length or model.config.max_position_embeddings
    if max_length < 3 or max_length > model.config.max_position_embeddings:
        raise ValueError("max_length must be between 3 and the model context length")
    if not 1 <= options.min_context_tokens < max_length:
        raise ValueError("min_context_tokens must be in [1, max_length)")
    if target_mode not in {"reasoning_and_response", "response_only"}:
        raise ValueError("target_mode must be 'reasoning_and_response' or 'response_only'")
    if options.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if options.learning_rate <= 0 or not 0 <= options.min_learning_rate <= options.learning_rate:
        raise ValueError("learning rates must satisfy 0 <= min_learning_rate <= learning_rate")
    if options.weight_decay < 0 or options.adam_eps <= 0 or options.grad_clip <= 0:
        raise ValueError("weight_decay must be non-negative; adam_eps and grad_clip must be positive")
    if not 0 <= options.adam_beta1 < 1 or not 0 <= options.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if options.beta <= 0:
        raise ValueError("beta must be positive")
    if options.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0 <= options.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if options.validation_every < 1 or options.validation_batches < 1 or options.log_every < 1:
        raise ValueError("logging and validation intervals must be positive")
    if options.save_every < 0 or options.keep_last_checkpoints < 1:
        raise ValueError("save_every must be non-negative and keep_last_checkpoints must be positive")
    if options.generation_every < 0 or options.generation_max_new_tokens < 1:
        raise ValueError("generation settings must be non-negative with a positive token limit")
    if options.num_workers < 0 or options.wandb_retry_every_steps < 1:
        raise ValueError("num_workers must be non-negative and W&B retry interval must be positive")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary does not match tokenizer")
    mismatches = {
        name: (getattr(model.config, name), getattr(tokenizer, name))
        for name in ("pad_token_id", "bos_token_id", "eos_token_id")
        if getattr(model.config, name) != getattr(tokenizer, name)
    }
    if mismatches:
        raise ValueError(f"model special token ids do not match tokenizer: {mismatches}")
    return max_length


def resolved_dpo_options(options: DPOOptions, *, target_mode: str) -> dict[str, object]:
    result = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(options).items()
        if name != "resume_from"
    }
    result["target_mode"] = target_mode
    return result


def dpo_resume_signature(
    options: DPOOptions,
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    *,
    max_length: int,
    target_mode: str,
    train_index: PreferenceCorpusIndex,
    validation_index: PreferenceCorpusIndex | None,
    resolved_precision: str,
    initial_checkpoint: dict[str, object],
) -> dict[str, object]:
    validation_identity: dict[str, object]
    if validation_index is None:
        validation_identity = {
            "mode": "prompt_hash_split",
            "fraction": options.validation_fraction,
            "source": train_index.identity,
        }
    else:
        validation_identity = {"mode": "dedicated_file", "source": validation_index.identity}
    return {
        "signature_version": DPO_RESUME_SIGNATURE_VERSION,
        "implementation_version": DPO_IMPLEMENTATION_VERSION,
        "objective": DPO_OBJECTIVE_VERSION,
        "total_steps": options.steps,
        "batch_size": options.batch_size,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "max_length": max_length,
        "min_context_tokens": options.min_context_tokens,
        "target_mode": target_mode,
        "pair_format": DPO_PAIR_FORMAT_VERSION,
        "truncation_version": DPO_TRUNCATION_VERSION,
        "data_order": DPO_DATA_ORDER_VERSION,
        "deduplicate_exact": options.deduplicate_exact,
        "learning_rate": options.learning_rate,
        "min_learning_rate": options.min_learning_rate,
        "warmup_steps": options.warmup_steps,
        "weight_decay": options.weight_decay,
        "adam_beta1": options.adam_beta1,
        "adam_beta2": options.adam_beta2,
        "adam_eps": options.adam_eps,
        "optimizer_parameter_groups": DPO_OPTIMIZER_GROUPING,
        "beta": options.beta,
        "grad_clip": options.grad_clip,
        "num_workers": options.num_workers,
        "validation_fraction": options.validation_fraction,
        "validation_every": options.validation_every,
        "validation_batches": options.validation_batches,
        "validation_sampling": "fixed_global_pair_sample_v1",
        "seed": options.seed,
        "precision": resolved_precision,
        "world_size": 1,
        "model": asdict(model.config),
        "tokenizer": tokenizer_identity(tokenizer),
        "initial_checkpoint": initial_checkpoint,
        "reference": {"kind": "frozen_initial_policy", "checkpoint": initial_checkpoint},
        "train_data": train_index.identity,
        "validation_data": validation_identity,
    }
