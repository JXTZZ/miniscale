"""Fine-tune on assistant turns and evaluate generation quality."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
from pathlib import Path
import sys
import time

from torch import Tensor
import torch
from torch.utils.data import DataLoader

# Direct file execution starts with this subdirectory on sys.path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import atomic_write_json, path_identity, tokenizer_identity
from dataset.lm_dataset import (
    DEFAULT_SFT_DATA,
    IndexedJsonlSFTDataset,
    SFTCorpusIndex,
    SFTDataset,
    SFT_DATA_ORDER_VERSION,
    SFT_EXAMPLE_FORMAT_VERSION,
    SFT_SELECTION_VERSION,
    SFT_TRUNCATION_VERSION,
    collate_lm_batch,
    fixed_validation_batches,
)
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import ByteTokenizer, Tokenizer, load_tokenizer
from trainer.checkpoint import (
    TRAINING_CHECKPOINT_FORMAT_VERSION,
    read_training_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    signature_differences,
)
from trainer.eval_utils import (
    DEFAULT_SFT_GENERATION_SUITE,
    SFT_GENERATION_METRICS_VERSION,
    SFT_GENERATION_PROMPTS,
    evaluate_sft_generation_quality,
    generation_quality_score,
    load_generation_suite,
)
from trainer.tracking import WandbTracker
from trainer.trainer_utils import (
    add_optimizer_arguments,
    add_wandb_arguments,
    append_metric,
    autocast_context,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    format_training_metric,
    infinite_batches,
    load_training_model,
    mirror_checkpoint,
    option_default,
    prune_periodic_checkpoints,
    resolve_autocast_dtype,
    resolve_device,
    seed_everything,
    seed_worker,
    train_smoke_batches,
    training_options,
    training_parser,
    truncate_metrics_after,
)


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
    generation_suite: str | Path | None = DEFAULT_SFT_GENERATION_SUITE
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


@torch.no_grad()
def evaluate_sft(
    model: MiniScaleForCausalLM,
    batches: Iterable[dict[str, Tensor]],
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None,
) -> tuple[float, float, int]:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    targets = correct = 0
    try:
        for batch_index, batch in enumerate(batches):
            device_batch = {name: value.to(device) for name, value in batch.items()}
            with autocast_context(device, autocast_dtype):
                result = model(**device_batch)
            if result.loss is None or not bool(torch.isfinite(result.loss)):
                raise FloatingPointError(f"non-finite SFT validation loss at batch {batch_index}")
            labels = device_batch["labels"][:, 1:]
            mask = labels.ne(-100)
            target_count = int(mask.sum())
            if not target_count:
                continue
            predictions = result.logits[:, :-1].argmax(dim=-1)
            correct += int((predictions.eq(labels) & mask).sum())
            targets += target_count
            weighted_loss += float(result.loss) * target_count
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("SFT validation batches contain no supervised targets")
    return weighted_loss / targets, correct / targets, targets


@torch.no_grad()
def run_sft_generation_evaluation(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    output_dir: Path,
    *,
    step: int,
    device: torch.device,
    max_new_tokens: int,
    autocast_dtype: torch.dtype | None,
    suite_path: str | Path | None = None,
) -> tuple[Path, dict[str, float]]:
    probes = load_generation_suite(suite_path) if suite_path is not None else list(SFT_GENERATION_PROMPTS)
    samples, summary = evaluate_sft_generation_quality(
        model,
        tokenizer,
        probes,
        device=device,
        max_new_tokens=max_new_tokens,
        autocast_dtype=autocast_dtype,
    )
    target = output_dir / "generations" / f"step_{step:08d}.json"
    atomic_write_json(target, {
        "schema_version": 2,
        "metrics_version": SFT_GENERATION_METRICS_VERSION,
        "stage": "sft",
        "step": step,
        "suite": str(Path(suite_path).resolve()) if suite_path is not None else "builtin_smoke_v1",
        "decoding": {
            "profile": "raw_greedy",
            "temperature": 0.0,
            "top_k": None,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
        },
        "summary": summary,
        "samples": samples,
    })
    return target, summary


def run_sft(
    model: MiniScaleForCausalLM,
    tokenizer: ByteTokenizer,
    conversations: list[list[dict[str, str]]],
    output_dir: str | Path,
    options: SmokeSFTOptions | None = None,
) -> dict[str, float | str]:
    """Run the deliberately small in-memory integration SFT stage."""

    options = options or SmokeSFTOptions()
    if options.steps < 1 or options.batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    seed_everything(options.seed)
    device = resolve_device(options.device)
    model.to(device).train()
    dataset = SFTDataset(conversations, tokenizer, model.config.max_position_embeddings)
    if not dataset:
        raise ValueError("SFT requires at least one conversation")
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_lm_batch(rows, tokenizer.pad_token_id),
    )
    metrics = train_smoke_batches(
        model, loader, device,
        steps=options.steps,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        grad_clip=options.grad_clip,
        missing_loss="model did not return an SFT loss",
    )
    checkpoint = save_checkpoint(
        Path(output_dir) / "sft.pt", model, stage="sft", step=options.steps, metrics=metrics
    )
    return {**metrics, "checkpoint": str(checkpoint), "device": str(device)}


def run_sft_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    options: SFTOptions,
    *,
    validation_path: str | Path | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    """Run reproducible assistant-turn SFT over indexed JSONL data."""

    max_length = validate_sft_options(options, model, tokenizer)
    output = Path(output_dir)
    metrics_path = output / "sft_metrics.jsonl"
    checkpoint_dir = output / "checkpoints"
    manifest_path = output / "sft_run.json"
    if options.resume_from is None:
        existing = [
            path
            for path in (
                manifest_path,
                metrics_path,
                output / "best.pt",
                output / "best_loss.pt",
                output / "best_quality.pt",
                output / "sft.pt",
            )
            if path.exists()
        ]
        existing.extend(checkpoint_dir.glob("*.pt") if checkpoint_dir.exists() else ())
        if existing:
            raise FileExistsError(
                f"output already contains SFT artifacts ({existing[0]}); choose a new --output or use --resume"
            )

    seed_everything(options.seed)
    device = resolve_device(options.device)
    autocast_dtype = resolve_autocast_dtype(options.precision, device)
    resolved_precision = "bf16" if autocast_dtype is torch.bfloat16 else "fp32"

    if validation_path is None:
        train_index = SFTCorpusIndex.build(
            data_path,
            validation_fraction=options.validation_fraction,
            target_mode=options.target_mode,
            destination="split",
            deduplicate_exact=options.deduplicate_exact,
        )
        validation_index = None
    else:
        train_index = SFTCorpusIndex.build(
            data_path,
            validation_fraction=0,
            target_mode=options.target_mode,
            destination="train",
            deduplicate_exact=options.deduplicate_exact,
        )
        validation_index = SFTCorpusIndex.build(
            validation_path,
            validation_fraction=0,
            target_mode=options.target_mode,
            destination="validation",
            deduplicate_exact=options.deduplicate_exact,
        )
        if validation_index.identity == train_index.identity:
            raise ValueError("dedicated validation data is identical to training data")

    train_dataset = IndexedJsonlSFTDataset(
        train_index,
        tokenizer,
        split="train",
        max_length=max_length,
        min_context_tokens=options.min_context_tokens,
        target_mode=options.target_mode,
    )
    validation_source = validation_index or train_index
    validation_dataset = IndexedJsonlSFTDataset(
        validation_source,
        tokenizer,
        split="validation",
        max_length=max_length,
        min_context_tokens=options.min_context_tokens,
        target_mode=options.target_mode,
    )
    if len(train_dataset) < options.batch_size:
        raise ValueError("SFT training split has fewer examples than batch_size")
    if (validation_path is not None or options.validation_fraction > 0) and not len(validation_dataset):
        raise ValueError("SFT validation split contains no examples")

    collate = partial(collate_lm_batch, pad_token_id=tokenizer.pad_token_id)
    train_generator = torch.Generator().manual_seed(options.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=options.num_workers,
        worker_init_fn=seed_worker if options.num_workers else None,
        generator=train_generator,
        drop_last=True,
    )
    validation_batches = None
    if len(validation_dataset):
        validation_batches = fixed_validation_batches(
            validation_dataset,
            batch_size=options.batch_size,
            batches=options.validation_batches,
            pad_token_id=tokenizer.pad_token_id,
            seed=options.seed + 1,
        )
    if options.early_stopping_patience and validation_batches is None:
        raise ValueError("early stopping requires a non-empty validation dataset")

    model.to(device).train()
    optimizer = build_adamw_optimizer(
        model,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        beta1=options.adam_beta1,
        beta2=options.adam_beta2,
        eps=options.adam_eps,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=options.steps,
        warmup_steps=options.warmup_steps,
        min_learning_rate=options.min_learning_rate,
    )

    payload: dict[str, object] | None = None
    saved_signature: dict[str, object] | None = None
    initial_checkpoint_source: str | None = None
    if options.resume_from is not None:
        payload = read_training_checkpoint(options.resume_from, device)
        if payload.get("stage") != "sft":
            raise ValueError("--resume requires a full SFT checkpoint")
        state = payload.get("training_state")
        if not isinstance(state, dict) or not isinstance(state.get("resume_signature"), dict):
            raise ValueError("SFT checkpoint does not contain a strict resume signature")
        saved_signature = state["resume_signature"]
        initial_checkpoint = saved_signature.get("initial_checkpoint")
        if not isinstance(initial_checkpoint, dict):
            raise ValueError("SFT checkpoint is missing its initialization identity")
        if state.get("initial_checkpoint_source") is not None:
            initial_checkpoint_source = str(state["initial_checkpoint_source"])
    elif initial_checkpoint_path is not None:
        initial_checkpoint = path_identity(initial_checkpoint_path)
        initial_checkpoint_source = str(Path(initial_checkpoint_path).resolve())
    else:
        initial_checkpoint = {"kind": "in_memory_model"}

    signature = sft_resume_signature(
        options,
        model,
        tokenizer,
        max_length=max_length,
        train_index=train_index,
        validation_index=validation_index,
        resolved_precision=resolved_precision,
        initial_checkpoint=initial_checkpoint,
    )
    if saved_signature is not None:
        differences = signature_differences(saved_signature, signature)
        if differences:
            raise ValueError(f"SFT resume options do not match checkpoint: {differences}")

    completed_step = micro_batches_seen = examples_seen = input_tokens_seen = target_tokens_seen = 0
    best_val_loss = float("inf")
    best_quality_score = float("-inf")
    best_quality_step = 0
    early_best_quality_score = float("-inf")
    stale_quality_evaluations = severe_loop_evaluations = clipped_steps = 0
    recent_validation_losses: list[float] = []
    last_loss = float("nan")
    saved_wandb_run_id: str | None = None
    last_metrics: dict[str, float] = {"loss": last_loss}
    if payload is not None:
        state = payload["training_state"]
        assert isinstance(state, dict)
        restore_training_checkpoint(payload, model, optimizer, scheduler, restore_rng=False)
        completed_step = int(payload["step"])
        micro_batches_seen = int(
            state.get("micro_batches_seen", completed_step * options.gradient_accumulation_steps)
        )
        examples_seen = int(state.get("examples_seen", micro_batches_seen * options.batch_size))
        input_tokens_seen = int(state.get("tokens_seen", 0))
        target_tokens_seen = int(state.get("target_tokens_seen", 0))
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        best_quality_score = float(state.get("best_quality_score", float("-inf")))
        best_quality_step = int(state.get("best_quality_step", 0))
        early_best_quality_score = float(state.get("early_best_quality_score", best_quality_score))
        stale_quality_evaluations = int(state.get("stale_quality_evaluations", 0))
        severe_loop_evaluations = int(state.get("severe_loop_evaluations", 0))
        clipped_steps = int(state.get("clipped_steps", 0))
        recent_validation_losses = [
            float(value) for value in state.get("recent_validation_losses", [])
        ]
        if state.get("wandb_run_id") is not None:
            saved_wandb_run_id = str(state["wandb_run_id"])
        last_metrics = {name: float(value) for name, value in payload.get("metrics", {}).items()}
        last_loss = float(last_metrics.get("loss", last_metrics.get("train_loss", float("nan"))))
        truncate_metrics_after(metrics_path, completed_step)
        print(f"resuming SFT from step={completed_step}; replaying {micro_batches_seen} micro-batches", flush=True)
    if completed_step >= options.steps:
        raise ValueError("resume checkpoint is already at or beyond the requested total steps")

    batches = iter(infinite_batches(loader))
    for _ in range(micro_batches_seen):
        next(batches)
    if payload is not None:
        restore_rng_state(payload.get("rng_state"))

    if options.wandb_run_id and saved_wandb_run_id and options.wandb_run_id != saved_wandb_run_id:
        raise ValueError("--wandb-run-id does not match the run id stored in the checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "stage": "sft",
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": SFT_IMPLEMENTATION_VERSION,
        "model": asdict(model.config),
        "num_parameters": model.num_parameters,
        "training": resolved_sft_options(options),
        "resolved": {"precision": resolved_precision, "max_length": max_length, "world_size": 1},
        "data": {
            "train_path": str(Path(data_path).resolve()),
            "validation_path": str(Path(validation_path).resolve()) if validation_path else None,
            "train_index": asdict(train_index.stats),
            "validation_index": asdict(validation_index.stats) if validation_index else None,
        },
        "initialization": {
            "checkpoint": initial_checkpoint_source,
            "identity": initial_checkpoint,
        },
        "resume": {
            "checkpoint": str(Path(options.resume_from).resolve()) if options.resume_from else None,
            "completed_step": completed_step,
        },
        "resume_identity": signature,
    }
    atomic_write_json(manifest_path, manifest)
    tracker = WandbTracker.start(
        enabled=options.wandb_enabled,
        project=options.wandb_project,
        entity=options.wandb_entity,
        name=options.wandb_run_name,
        run_id=options.wandb_run_id or saved_wandb_run_id,
        mode=options.wandb_mode,
        config={**manifest, "manifest": str(manifest_path)},
        directory=output,
        retry_every_steps=options.wandb_retry_every_steps,
        initial_step=completed_step,
    )
    wandb_run_id = tracker.run_id if tracker is not None else saved_wandb_run_id

    def training_state() -> dict[str, object]:
        return {
            "tokens_seen": input_tokens_seen,
            "target_tokens_seen": target_tokens_seen,
            "examples_seen": examples_seen,
            "micro_batches_seen": micro_batches_seen,
            "best_val_loss": best_val_loss,
            "best_quality_score": best_quality_score,
            "best_quality_step": best_quality_step,
            "early_best_quality_score": early_best_quality_score,
            "stale_quality_evaluations": stale_quality_evaluations,
            "severe_loop_evaluations": severe_loop_evaluations,
            "recent_validation_losses": recent_validation_losses,
            "clipped_steps": clipped_steps,
            "wandb_run_id": wandb_run_id,
            "initial_checkpoint_source": initial_checkpoint_source,
            "resolved_options": resolved_sft_options(options),
            "resume_signature": signature,
        }

    try:
        for step in range(completed_step + 1, options.steps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            update_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            cpu_batches = [next(batches) for _ in range(options.gradient_accumulation_steps)]
            target_counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in cpu_batches]
            total_targets = sum(target_counts)
            if total_targets < 1:
                raise RuntimeError("SFT optimizer update contains no supervised next-token targets")
            weighted_loss = 0.0
            step_input_tokens = 0
            step_examples = 0
            current_lr = float(optimizer.param_groups[0]["lr"])
            for micro_step, (cpu_batch, target_count) in enumerate(
                zip(cpu_batches, target_counts, strict=True), 1
            ):
                batch = {name: value.to(device) for name, value in cpu_batch.items()}
                with autocast_context(device, autocast_dtype):
                    result = model(**batch)
                if result.loss is None:
                    raise RuntimeError("model did not return an SFT loss")
                if not bool(torch.isfinite(result.loss)):
                    raise FloatingPointError(
                        f"non-finite SFT loss at optimizer step {step}, micro step {micro_step}: "
                        f"{float(result.loss.detach())}"
                    )
                (result.loss * (target_count / total_targets)).backward()
                weighted_loss += float(result.loss.detach()) * target_count
                step_input_tokens += int(batch["attention_mask"].sum())
                step_examples += int(batch["input_ids"].shape[0])
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), options.grad_clip, error_if_nonfinite=True
                )
            except RuntimeError as error:
                if "non-finite" not in str(error):
                    raise
                raise FloatingPointError(
                    f"non-finite SFT gradient norm at optimizer step {step}"
                ) from error
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_started
            completed_step = step
            micro_batches_seen += options.gradient_accumulation_steps
            examples_seen += step_examples
            input_tokens_seen += step_input_tokens
            target_tokens_seen += total_targets
            last_loss = weighted_loss / total_targets
            metric: dict[str, object] = {
                "stage": "sft",
                "step": step,
                "examples_seen": examples_seen,
                "tokens_seen": input_tokens_seen,
                "target_tokens_seen": target_tokens_seen,
                "train_loss": last_loss,
                "learning_rate": current_lr,
                "grad_norm": float(grad_norm),
                "grad_was_clipped": bool(float(grad_norm) > options.grad_clip),
                "update_seconds": update_seconds,
                "tokens_per_second": step_input_tokens / max(update_seconds, 1e-12),
                "supervised_tokens_per_second": total_targets / max(update_seconds, 1e-12),
                "samples_per_second": step_examples / max(update_seconds, 1e-12),
            }
            clipped_steps += int(bool(metric["grad_was_clipped"]))
            metric["grad_clip_fraction"] = clipped_steps / completed_step
            if device.type == "cuda":
                metric["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            validation_due = validation_batches is not None and (
                step % options.validation_every == 0 or step == options.steps
            )
            if validation_due:
                val_loss, val_accuracy, val_targets = evaluate_sft(
                    model, validation_batches, device, autocast_dtype=autocast_dtype
                )
                metric.update({
                    "validation_loss": val_loss,
                    "validation_token_accuracy": val_accuracy,
                    "validation_target_tokens": val_targets,
                    "perplexity": math.exp(val_loss) if val_loss < 709 else float("inf"),
                })
                recent_validation_losses.append(val_loss)
                maximum_losses = max(options.early_stopping_patience + 1, 2)
                recent_validation_losses = recent_validation_losses[-maximum_losses:]
                if math.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_metrics = {
                        "loss": last_loss,
                        "validation_loss": val_loss,
                        "validation_token_accuracy": val_accuracy,
                        "best_val_loss": best_val_loss,
                    }
                    best_checkpoint = save_training_checkpoint(
                        output / "best_loss.pt",
                        model,
                        optimizer,
                        scheduler,
                        stage="sft",
                        step=step,
                        metrics=best_metrics,
                        training_state=training_state(),
                    )
                    print(f"saved best-loss SFT checkpoint: {best_checkpoint}", flush=True)
                metric["best_val_loss"] = best_val_loss

            last_metrics = {
                "loss": last_loss,
                "tokens_seen": float(input_tokens_seen),
                "target_tokens_seen": float(target_tokens_seen),
                "examples_seen": float(examples_seen),
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss,
            }
            if validation_due:
                last_metrics.update({
                    "validation_loss": float(metric["validation_loss"]),
                    "validation_token_accuracy": float(metric["validation_token_accuracy"]),
                    "perplexity": float(metric["perplexity"]),
                })

            generation_path: Path | None = None
            if options.generation_every and step % options.generation_every == 0:
                generation_path, generation_summary = run_sft_generation_evaluation(
                    model,
                    tokenizer,
                    output,
                    step=step,
                    device=device,
                    max_new_tokens=options.generation_max_new_tokens,
                    autocast_dtype=autocast_dtype,
                    suite_path=options.generation_suite,
                )
                metric.update(generation_summary)
                quality_score = generation_quality_score(generation_summary)
                metric["generation_quality_score"] = quality_score
                if quality_score >= early_best_quality_score + options.early_stopping_quality_min_delta:
                    early_best_quality_score = quality_score
                    stale_quality_evaluations = 0
                else:
                    stale_quality_evaluations += 1
                if generation_summary["generation_loop_rate"] > options.severe_loop_rate_threshold:
                    severe_loop_evaluations += 1
                else:
                    severe_loop_evaluations = 0
                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_quality_step = step
                    quality_metrics = {
                        "loss": last_loss,
                        "best_val_loss": best_val_loss,
                        "best_quality_score": best_quality_score,
                        "best_quality_step": float(best_quality_step),
                        **generation_summary,
                    }
                    quality_checkpoint = save_training_checkpoint(
                        output / "best_quality.pt",
                        model,
                        optimizer,
                        scheduler,
                        stage="sft",
                        step=step,
                        metrics=quality_metrics,
                        training_state=training_state(),
                    )
                    mirror_checkpoint(quality_checkpoint, output / "best.pt")
                    print(
                        f"saved best-quality SFT checkpoint: {quality_checkpoint} "
                        f"(score={best_quality_score:.4f})",
                        flush=True,
                    )
                metric["best_quality_score"] = best_quality_score
                metric["best_quality_step"] = best_quality_step
                metric["stale_quality_evaluations"] = stale_quality_evaluations
            log_due = (
                step == 1
                or step % options.log_every == 0
                or step == options.steps
                or validation_due
                or generation_path is not None
            )
            if log_due:
                append_metric(metrics_path, metric)
                if tracker is not None:
                    tracker.log(metric, generation_path=generation_path)
                print(format_training_metric(metric), flush=True)
            if options.save_every and step % options.save_every == 0:
                checkpoint = save_training_checkpoint(
                    checkpoint_dir / f"step_{step:08d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    stage="sft",
                    step=step,
                    metrics=last_metrics,
                    training_state=training_state(),
                )
                prune_periodic_checkpoints(checkpoint_dir, options.keep_last_checkpoints)
                print(f"saved SFT checkpoint: {checkpoint}", flush=True)
            if (
                options.early_stopping_patience
                and step >= options.early_stopping_min_steps
                and validation_due
                and generation_path is not None
            ):
                loss_stalled = (
                    len(recent_validation_losses) >= options.early_stopping_patience + 1
                    and recent_validation_losses[0] - recent_validation_losses[-1]
                    < options.early_stopping_validation_min_delta
                )
                patience_exhausted = (
                    stale_quality_evaluations >= options.early_stopping_patience and loss_stalled
                )
                severe_regression = severe_loop_evaluations >= 2
                if patience_exhausted or severe_regression:
                    print(
                        f"early stopping SFT at step={step}: "
                        f"quality_stale={stale_quality_evaluations}, "
                        f"loss_stalled={loss_stalled}, severe_loop={severe_regression}",
                        flush=True,
                    )
                    break
    except (KeyboardInterrupt, FloatingPointError):
        emergency = save_training_checkpoint(
            checkpoint_dir / f"emergency_step_{completed_step:08d}.pt",
            model,
            optimizer,
            scheduler,
            stage="sft",
            step=completed_step,
            metrics=last_metrics,
            training_state=training_state(),
        )
        print(f"SFT interrupted; emergency checkpoint saved: {emergency}", flush=True)
        if tracker is not None:
            tracker.finish(exit_code=1, summary={"interrupted_step": completed_step})
        raise

    metrics = {
        "loss": last_loss,
        "tokens_seen": float(input_tokens_seen),
        "target_tokens_seen": float(target_tokens_seen),
        "examples_seen": float(examples_seen),
        "learning_rate": float(last_metrics["learning_rate"]),
        "best_val_loss": best_val_loss,
        "best_quality_score": best_quality_score,
        "best_quality_step": float(best_quality_step),
        "stopped_early": float(completed_step < options.steps),
        "grad_clip_fraction": clipped_steps / max(completed_step, 1),
    }
    checkpoint = save_training_checkpoint(
        output / "sft.pt",
        model,
        optimizer,
        scheduler,
        stage="sft",
        step=completed_step,
        metrics=metrics,
        training_state=training_state(),
    )
    if tracker is not None:
        tracker.finish(summary={**metrics, "final_step": completed_step})
    result: dict[str, float | str] = {
        **metrics,
        "checkpoint": str(checkpoint),
        "metrics": str(metrics_path),
        "manifest": str(manifest_path),
        "device": str(device),
    }
    if wandb_run_id is not None:
        result["wandb_run_id"] = wandb_run_id
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = training_parser('Fine-tune a pretraining checkpoint.', 'sft', DEFAULT_SFT_DATA, SFTOptions)
    default = partial(option_default, SFTOptions)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="initialize a new SFT run from model weights")
    source.add_argument("--resume", type=Path, help="resume the exact state of a full SFT checkpoint")
    parser.add_argument("--max-length", type=int, help="defaults to the checkpoint context length")
    parser.add_argument("--min-context-tokens", type=int, default=default("min_context_tokens"))
    parser.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default=default("target_mode"),
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=default("gradient_accumulation_steps")
    )
    add_optimizer_arguments(parser, SFTOptions)
    parser.add_argument(
        "--precision", choices=("fp32", "bf16"), default=default("precision")
    )
    parser.add_argument("--validation-fraction", type=float, default=default("validation_fraction"))
    parser.add_argument("--validation-data", type=Path, help="optional dedicated SFT validation JSONL")
    parser.add_argument("--validation-every", type=int, default=default("validation_every"))
    parser.add_argument("--validation-batches", type=int, default=default("validation_batches"))
    parser.add_argument("--save-every", type=int, default=default("save_every"))
    parser.add_argument("--keep-last", type=int, default=default("keep_last_checkpoints"))
    parser.add_argument("--generation-every", type=int, default=default("generation_every"))
    parser.add_argument(
        "--generation-max-new-tokens", type=int, default=default("generation_max_new_tokens")
    )
    parser.add_argument(
        "--generation-suite",
        type=Path,
        default=default("generation_suite"),
        help="fixed JSONL probe suite used for generation quality and checkpoint selection",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=default("early_stopping_patience"),
        help="quality evaluations without improvement before stopping; 0 disables",
    )
    parser.add_argument(
        "--early-stopping-min-steps", type=int,
        default=default("early_stopping_min_steps"),
    )
    parser.add_argument(
        "--early-stopping-validation-min-delta", type=float,
        default=default("early_stopping_validation_min_delta"),
    )
    parser.add_argument(
        "--early-stopping-quality-min-delta", type=float,
        default=default("early_stopping_quality_min_delta"),
    )
    parser.add_argument(
        "--severe-loop-rate-threshold", type=float,
        default=default("severe_loop_rate_threshold"),
    )
    parser.add_argument(
        "--deduplicate-exact",
        action=argparse.BooleanOptionalAction,
        default=default("deduplicate_exact"),
    )
    parser.add_argument("--seed", type=int, default=default("seed"))
    parser.add_argument("--num-workers", type=int, default=default("num_workers"))
    add_wandb_arguments(parser, SFTOptions)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(arguments.tokenizer)
    model = load_training_model(arguments, tokenizer)
    options = training_options(SFTOptions, arguments)
    result = run_sft_jsonl(
        model, tokenizer, arguments.data, arguments.output, options,
        validation_path=arguments.validation_data,
        initial_checkpoint_path=arguments.checkpoint,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
