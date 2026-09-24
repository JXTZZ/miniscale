"""Configuration and stable format constants for pretraining."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path


PRETRAIN_RESUME_SIGNATURE_VERSION = 2
PRETRAIN_IMPLEMENTATION_VERSION = 2
PRETRAIN_INITIALIZATION_SCHEME = "normal_0.02_residual_scaled_1_over_sqrt_2L_v1"
PRETRAIN_OPTIMIZER_GROUPING = "matrix_weights_decay_norm_embedding_no_decay_v1"


@dataclass(slots=True)
class PretrainOptions:
    """Resolved options for the production JSONL pretraining path.

    CLI defaults are read from this dataclass so library and command-line
    callers share one source of truth. ``steps`` intentionally has no default:
    a real training run must always choose its token/update budget explicitly.
    """

    steps: int
    batch_size: int = 1
    sequence_length: int = 768
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    precision: str = "fp32"
    gradient_accumulation_steps: int = 16
    log_every: int = 10
    validation_every: int = 200
    validation_batches: int = 20
    num_workers: int = 0
    validation_fraction: float = 0.005
    warmup_steps: int = 200
    min_learning_rate: float = 3e-5
    save_every: int = 500
    keep_last_checkpoints: int = 3
    generation_every: int = 1000
    generation_max_new_tokens: int = 64
    shuffle_buffer_size: int = 8192
    wandb_enabled: bool = False
    wandb_project: str = "MiniScale"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str = "online"
    wandb_retry_every_steps: int = 200
    resume_from: str | Path | None = None
    allow_legacy_resume: bool = False


@dataclass(slots=True)
class SmokePretrainOptions:
    """Small in-memory integration settings; not a production recipe."""

    steps: int = 2
    batch_size: int = 2
    sequence_length: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"


def pretrain_option_default(name: str) -> object:
    """Return a production option default without constructing fake steps."""

    option = next((field for field in fields(PretrainOptions) if field.name == name), None)
    if option is None:
        raise KeyError(f"unknown pretraining option: {name}")
    if option.default is MISSING:
        raise ValueError(f"pretraining option has no default: {name}")
    return option.default
