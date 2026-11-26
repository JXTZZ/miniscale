"""Training setup and progress shared by the language-model stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.file_utils import atomic_write_json, tokenizer_identity
from model.model_miniscale import MiniScaleForCausalLM
from trainer.checkpoint import TRAINING_CHECKPOINT_FORMAT_VERSION
from trainer.runtime.torch_utils import build_adamw_optimizer, build_warmup_cosine_scheduler, seed_worker


OPTIMIZER_GROUPING = "matrix_weights_decay_norm_embedding_no_decay_v1"


@dataclass(slots=True)
class LMProgress:
    step: int = 0
    micro_batches: int = 0
    tokens: int = 0
    targets: int = 0
    best_val_loss: float = float("inf")
    wandb_run_id: str | None = None
    last_metrics: dict[str, float] = field(default_factory=lambda: {"loss": float("nan")})

    @classmethod
    def from_checkpoint(cls, payload, options, *, legacy_pretrain: bool = False) -> LMProgress:
        if payload is None:
            return cls(last_metrics={"loss": float("nan"), "tokens_seen": 0.0} if legacy_pretrain
                       else {"loss": float("nan")})
        state = payload["training_state"]
        step = int(payload["step"])
        batches = int(state.get("micro_batches_seen", step * options.gradient_accumulation_steps))
        tokens = int(state.get("tokens_seen", 0))
        old_target_count = max(tokens - batches * options.batch_size, 0) if legacy_pretrain else 0
        old_best = payload.get("best_val_loss", float("inf")) if legacy_pretrain else float("inf")
        run_id = state.get("wandb_run_id")
        return cls(
            step=step, micro_batches=batches, tokens=tokens,
            targets=int(state.get("target_tokens_seen", old_target_count)),
            best_val_loss=float(state.get("best_val_loss", old_best)),
            wandb_run_id=str(run_id) if run_id is not None else None,
            last_metrics={name: float(value) for name, value in payload.get("metrics", {}).items()},
        )

    def advance(self, step: int, batches: int, tokens: int, targets: int) -> None:
        self.step = step
        self.micro_batches += batches
        self.tokens += tokens
        self.targets += targets

    def state_dict(self, options, signature, **extra) -> dict[str, object]:
        return {
            "tokens_seen": self.tokens, "target_tokens_seen": self.targets,
            "micro_batches_seen": self.micro_batches, "best_val_loss": self.best_val_loss,
            "wandb_run_id": self.wandb_run_id, "resolved_options": options, "resume_signature": signature,
            **extra,
        }

    def loss_metrics(self, loss: float, learning_rate: float, **extra) -> dict[str, float]:
        return {
            "loss": loss, "tokens_seen": float(self.tokens), "target_tokens_seen": float(self.targets),
            "learning_rate": learning_rate, "best_val_loss": self.best_val_loss, **extra,
        }

    def training_metrics(
        self, stage, loss, lr, grad_norm, duration, device, *, grad_clip,
        tokens, targets, samples,
    ) -> dict[str, object]:
        norm = float(grad_norm)
        metrics: dict[str, object] = {
            "stage": stage, "step": self.step, "tokens_seen": self.tokens, "target_tokens_seen": self.targets,
            "train_loss": loss, "learning_rate": lr, "grad_norm": norm, "grad_was_clipped": norm > grad_clip,
        }
        if duration is None:
            return metrics
        rate = max(duration, 1e-12)
        metrics.update(update_seconds=duration, tokens_per_second=tokens / rate)
        if stage != "pretrain":
            metrics["supervised_tokens_per_second"] = targets / rate
        metrics["samples_per_second" if stage != "dpo" else "pairs_per_second"] = samples / rate
        if device.type == "cuda":
            metrics["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
        return metrics


def validate_model_tokenizer(model, tokenizer) -> None:
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary does not match tokenizer")
    mismatches = {
        name: (getattr(model.config, name), getattr(tokenizer, name))
        for name in ("pad_token_id", "bos_token_id", "eos_token_id")
        if getattr(model.config, name) != getattr(tokenizer, name)
    }
    if mismatches:
        raise ValueError(f"model special token ids do not match tokenizer: {mismatches}")


def validate_lm_options(options, *, pretrain: bool = False) -> None:
    if options.steps < 1 or options.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    if options.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if min(options.log_every, options.validation_every, options.validation_batches) < 1:
        raise ValueError("logging and validation intervals must be positive")
    if not 0 <= options.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if options.save_every < 0 or options.keep_last_checkpoints < 1:
        raise ValueError("save_every must be non-negative and keep_last_checkpoints must be positive")
    if options.generation_every < 0 or options.generation_max_new_tokens < 1:
        raise ValueError("generation settings must be non-negative with a positive token limit")
    if options.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if pretrain:
        if options.learning_rate <= 0 or options.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative and peak learning_rate must be positive")
    elif options.learning_rate <= 0 or not 0 <= options.min_learning_rate <= options.learning_rate:
        raise ValueError("learning rates must satisfy 0 <= min_learning_rate <= learning_rate")
    if options.weight_decay < 0 or options.grad_clip <= 0 or options.adam_eps <= 0:
        raise ValueError("weight_decay must be non-negative; grad_clip and adam_eps must be positive")
    if not 0 <= options.adam_beta1 < 1 or not 0 <= options.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if not pretrain and options.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if options.wandb_retry_every_steps < 1:
        raise ValueError("wandb_retry_every_steps must be positive")


def finetune_context(options, model, target_mode: str) -> int:
    max_length = options.max_length or model.config.max_position_embeddings
    if not 3 <= max_length <= model.config.max_position_embeddings:
        raise ValueError("max_length must be between 3 and the model context length")
    if not 1 <= options.min_context_tokens < max_length:
        raise ValueError("min_context_tokens must be in [1, max_length)")
    if target_mode not in {"reasoning_and_response", "response_only"}:
        raise ValueError("target_mode must be 'reasoning_and_response' or 'response_only'")
    return max_length


def resolved_lm_options(options, **overrides) -> dict[str, object]:
    return {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(options).items()
            if key not in {"resume_from", "allow_legacy_resume"}
        },
        **overrides,
    }


def lm_resume_signature(options, model, tokenizer, precision: str) -> dict[str, object]:
    fields = (
        "batch_size", "gradient_accumulation_steps", "learning_rate", "min_learning_rate",
        "warmup_steps", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps", "grad_clip",
        "num_workers", "validation_fraction", "validation_every", "validation_batches", "seed",
    )
    return {
        **{name: getattr(options, name) for name in fields},
        "total_steps": options.steps,
        "optimizer_parameter_groups": OPTIMIZER_GROUPING,
        "precision": precision,
        "world_size": 1,
        "model": asdict(model.config),
        "tokenizer": tokenizer_identity(tokenizer),
    }


def validation_identity(train, validation, fraction: float, split: str) -> dict[str, object]:
    if validation is None:
        return {"mode": split, "fraction": fraction, "source": train}
    return {"mode": "dedicated_file", "source": validation}


def training_loader(dataset, tokenizer, options, collate=None, *, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset, batch_size=options.batch_size, shuffle=shuffle,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id) if collate is not None else None,
        num_workers=options.num_workers, worker_init_fn=seed_worker if options.num_workers else None,
        generator=torch.Generator().manual_seed(options.seed), drop_last=True,
    )


def indexed_training_data(
    build_index, build_dataset, validation_sample, *, options, tokenizer,
    data_path, validation_path, max_length: int, target_mode: str, label: str, unit: str,
):
    index_args = {"target_mode": target_mode, "deduplicate_exact": options.deduplicate_exact}
    train = build_index(
        data_path, **index_args,
        validation_fraction=options.validation_fraction if validation_path is None else 0,
        destination="split" if validation_path is None else "train",
    )
    validation = None
    if validation_path is not None:
        validation = build_index(validation_path, **index_args, validation_fraction=0, destination="validation")
        if validation.identity == train.identity:
            prefix = "DPO " if label == "DPO" else ""
            raise ValueError(f"dedicated {prefix}validation data is identical to training data")
    dataset_args = {
        "max_length": max_length, "min_context_tokens": options.min_context_tokens, "target_mode": target_mode,
    }
    training = build_dataset(train, tokenizer, split="train", **dataset_args)
    held_out = build_dataset(validation or train, tokenizer, split="validation", **dataset_args)
    if len(training) < options.batch_size:
        raise ValueError(f"{label} training split has fewer {unit} than batch_size")
    if (validation_path is not None or options.validation_fraction > 0) and not len(held_out):
        raise ValueError(f"{label} validation split contains no {unit}")
    batches = None
    if len(held_out):
        batches = validation_sample(
            held_out, batch_size=options.batch_size, batches=options.validation_batches,
            pad_token_id=tokenizer.pad_token_id, seed=options.seed + 1,
        )
    return train, validation, training, batches


def build_lm_optimizer(model, options) -> torch.optim.AdamW:
    return build_adamw_optimizer(
        model, learning_rate=options.learning_rate, weight_decay=options.weight_decay,
        beta1=options.adam_beta1, beta2=options.adam_beta2, eps=options.adam_eps,
    )


def lm_scheduler(optimizer, options):
    return build_warmup_cosine_scheduler(
        optimizer, total_steps=options.steps, warmup_steps=options.warmup_steps,
        min_learning_rate=options.min_learning_rate,
    )


def finetune_manifest(
    stage, model, options, precision, max_length, data_path, validation_path,
    train_index, validation_index, checkpoint_source, checkpoint_identity,
    step, signature,
):
    return {
        "schema_version": 1, "stage": stage,
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": signature["implementation_version"],
        "model": asdict(model.config), "num_parameters": model.num_parameters,
        "training": resolved_lm_options(options, target_mode=signature["target_mode"]),
        "resolved": {"precision": precision, "max_length": max_length, "world_size": 1},
        "data": {
            "train_path": str(Path(data_path).resolve()),
            "validation_path": str(Path(validation_path).resolve()) if validation_path else None,
            "train_index": asdict(train_index.stats),
            "validation_index": asdict(validation_index.stats) if validation_index else None,
        },
        "initialization": {"checkpoint": checkpoint_source, "identity": checkpoint_identity},
        "resume": {
            "checkpoint": str(Path(options.resume_from).resolve()) if options.resume_from else None,
            "completed_step": step,
        },
        "resume_identity": signature,
    }


def write_pretrain_manifest(
    path: Path,
    *,
    model: MiniScaleForCausalLM,
    options,
    resume_signature: dict[str, object],
    train_path: str | Path,
    validation_path: str | Path | None,
    resumed_step: int,
    resolved_precision: str,
) -> None:
    examples = options.batch_size * options.gradient_accumulation_steps
    inputs = examples * options.sequence_length
    targets = examples * (options.sequence_length - 1)
    manifest = {
        "schema_version": 1,
        "stage": "pretrain",
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": resume_signature["implementation_version"],
        "parameter_initialization": resume_signature["parameter_initialization"],
        "optimizer_parameter_groups": OPTIMIZER_GROUPING,
        "model": asdict(model.config),
        "num_parameters": model.num_parameters,
        "training": resolved_lm_options(options),
        "resolved_precision": resolved_precision,
        "derived": {
            "world_size": 1,
            "global_batch_sequences": examples,
            "input_tokens_per_update": inputs,
            "target_tokens_per_update": targets,
            "planned_input_tokens": options.steps * inputs,
            "planned_target_tokens": options.steps * targets,
            "warmup_ratio": min(options.warmup_steps, options.steps) / options.steps,
            "tokens_per_parameter": options.steps * inputs / model.num_parameters,
        },
        "inputs": {
            "train": str(Path(train_path).resolve()),
            "validation": str(Path(validation_path).resolve()) if validation_path else None,
        },
        "resume": {
            "checkpoint": str(Path(options.resume_from).resolve()) if options.resume_from else None,
            "completed_step": resumed_step,
        },
        "resume_identity": resume_signature,
    }
    atomic_write_json(path, manifest)


def clip_finetune_gradients(model, grad_clip: float, stage: str, step: int):
    try:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
        raise FloatingPointError(f"non-finite {stage} gradient norm at optimizer step {step}") from error
