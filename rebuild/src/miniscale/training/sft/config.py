from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

from miniscale.integrity import path_identity, tokenizer_identity
from miniscale.model import MiniScaleForCausalLM
from miniscale.data.sft import (
    SFT_DATA_ORDER_VERSION,
    SFT_EXAMPLE_FORMAT_VERSION,
    SFT_SELECTION_VERSION,
    SFT_TRUNCATION_VERSION,
    SFTCorpusIndex,
)
from miniscale.tokenizer import Tokenizer


SFT_RESUME_SIGNATURE_VERSION = 1
SFT_IMPLEMENTATION_VERSION = 2
SFT_MASK_VERSION = "structured_assistant_span_v1"
SFT_OPTIMIZER_GROUPING = "matrix_weights_decay_norm_embedding_no_decay_v1"


@dataclass(slots=True)
class SFTOptions:
    """Resolved options for production JSONL supervised fine-tuning."""

    steps: int
    batch_size: int = 1
    max_length: int | None = None
    min_context_tokens: int = 32
    target_mode: str = "reasoning_and_response"
    learning_rate: float = 2e-5
    min_learning_rate: float = 2e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 100
    precision: str = "fp32"
    gradient_accumulation_steps: int = 16
    validation_fraction: float = 0.005
    validation_every: int = 200
    validation_batches: int = 100
    save_every: int = 500
    keep_last_checkpoints: int = 3
    generation_every: int = 1000
    generation_max_new_tokens: int = 96
    generation_suite: str | Path | None = Path("data/eval/sft_generation_v1.jsonl")
    early_stopping_patience: int = 0
    early_stopping_min_steps: int = 1000
    early_stopping_validation_min_delta: float = 0.002
    early_stopping_quality_min_delta: float = 0.005
    severe_loop_rate_threshold: float = 0.20
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


@dataclass(slots=True)
class SmokeSFTOptions:
    """Small in-memory integration settings; not a production recipe."""

    steps: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"


def sft_option_default(name: str) -> object:
    option = next((field for field in fields(SFTOptions) if field.name == name), None)
    if option is None:
        raise KeyError(f"unknown SFT option: {name}")
    if option.default is MISSING:
        raise ValueError(f"SFT option has no default: {name}")
    return option.default


def validate_sft_options(
    options: SFTOptions,
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
) -> int:
    if options.steps < 1 or options.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    max_length = options.max_length or model.config.max_position_embeddings
    if max_length < 3 or max_length > model.config.max_position_embeddings:
        raise ValueError("max_length must be between 3 and the model context length")
    if not 1 <= options.min_context_tokens < max_length:
        raise ValueError("min_context_tokens must be in [1, max_length)")
    if options.target_mode not in {"reasoning_and_response", "response_only"}:
        raise ValueError("target_mode must be 'reasoning_and_response' or 'response_only'")
    if options.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if options.log_every < 1 or options.validation_every < 1 or options.validation_batches < 1:
        raise ValueError("logging and validation intervals must be positive")
    if not 0 <= options.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if options.save_every < 0 or options.keep_last_checkpoints < 1:
        raise ValueError("save_every must be non-negative and keep_last_checkpoints must be positive")
    if options.generation_every < 0 or options.generation_max_new_tokens < 1:
        raise ValueError("generation settings must be non-negative with a positive token limit")
    if options.generation_every and options.generation_suite is not None and not Path(
        options.generation_suite
    ).is_file():
        raise FileNotFoundError(f"SFT generation suite does not exist: {options.generation_suite}")
    if options.early_stopping_patience < 0 or options.early_stopping_min_steps < 0:
        raise ValueError("early stopping patience and minimum steps must be non-negative")
    if options.early_stopping_validation_min_delta < 0 or options.early_stopping_quality_min_delta < 0:
        raise ValueError("early stopping minimum deltas must be non-negative")
    if not 0 <= options.severe_loop_rate_threshold <= 1:
        raise ValueError("severe_loop_rate_threshold must be in [0, 1]")
    if options.early_stopping_patience and not options.generation_every:
        raise ValueError("early stopping requires generation evaluation")
    if options.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if options.learning_rate <= 0 or not 0 <= options.min_learning_rate <= options.learning_rate:
        raise ValueError("learning rates must satisfy 0 <= min_learning_rate <= learning_rate")
    if options.weight_decay < 0 or options.grad_clip <= 0 or options.adam_eps <= 0:
        raise ValueError("weight_decay must be non-negative; grad_clip and adam_eps must be positive")
    if not 0 <= options.adam_beta1 < 1 or not 0 <= options.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if options.warmup_steps < 0 or options.wandb_retry_every_steps < 1:
        raise ValueError("warmup_steps must be non-negative and W&B retry interval must be positive")
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


def resolved_sft_options(options: SFTOptions) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(options).items()
        if name != "resume_from"
    }


def sft_resume_signature(
    options: SFTOptions,
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    *,
    max_length: int,
    train_index: SFTCorpusIndex,
    validation_index: SFTCorpusIndex | None,
    resolved_precision: str,
    initial_checkpoint: dict[str, object],
) -> dict[str, object]:
    validation_identity: dict[str, object]
    if validation_index is None:
        validation_identity = {
            "mode": "conversation_hash_split",
            "fraction": options.validation_fraction,
            "source": train_index.identity,
        }
    else:
        validation_identity = {"mode": "dedicated_file", "source": validation_index.identity}
    return {
        "signature_version": SFT_RESUME_SIGNATURE_VERSION,
        "implementation_version": SFT_IMPLEMENTATION_VERSION,
        "total_steps": options.steps,
        "batch_size": options.batch_size,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "max_length": max_length,
        "min_context_tokens": options.min_context_tokens,
        "target_mode": options.target_mode,
        "example_format": SFT_EXAMPLE_FORMAT_VERSION,
        "selection_version": SFT_SELECTION_VERSION,
        "mask_version": SFT_MASK_VERSION,
        "truncation_version": SFT_TRUNCATION_VERSION,
        "data_order": SFT_DATA_ORDER_VERSION,
        "deduplicate_exact": options.deduplicate_exact,
        "learning_rate": options.learning_rate,
        "min_learning_rate": options.min_learning_rate,
        "warmup_steps": options.warmup_steps,
        "weight_decay": options.weight_decay,
        "adam_beta1": options.adam_beta1,
        "adam_beta2": options.adam_beta2,
        "adam_eps": options.adam_eps,
        "optimizer_parameter_groups": SFT_OPTIMIZER_GROUPING,
        "grad_clip": options.grad_clip,
        "num_workers": options.num_workers,
        "validation_fraction": options.validation_fraction,
        "validation_every": options.validation_every,
        "validation_batches": options.validation_batches,
        "validation_sampling": "fixed_global_sample_v1",
        "generation_suite": (
            path_identity(options.generation_suite)
            if options.generation_suite is not None
            else {"kind": "builtin_smoke_v1"}
        ),
        "generation_every": options.generation_every,
        "generation_max_new_tokens": options.generation_max_new_tokens,
        "early_stopping_patience": options.early_stopping_patience,
        "early_stopping_min_steps": options.early_stopping_min_steps,
        "early_stopping_validation_min_delta": options.early_stopping_validation_min_delta,
        "early_stopping_quality_min_delta": options.early_stopping_quality_min_delta,
        "severe_loop_rate_threshold": options.severe_loop_rate_threshold,
        "seed": options.seed,
        "precision": resolved_precision,
        "world_size": 1,
        "model": asdict(model.config),
        "tokenizer": tokenizer_identity(tokenizer),
        "initial_checkpoint": initial_checkpoint,
        "train_data": train_index.identity,
        "validation_data": validation_identity,
    }
