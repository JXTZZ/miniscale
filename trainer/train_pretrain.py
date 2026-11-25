"""Pretrain the MiniScale base model on packed JSONL text."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import default_collate

# Support running this file directly as well as with python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import path_identity
from dataset.lm_dataset import (
    DEFAULT_PRETRAIN_DATA,
    JsonlPretrainDataset,
    reservoir_sample_lm_batches,
)
from model.model_miniscale import MiniScaleConfig, MiniScaleForCausalLM
from model.tokenizer import load_tokenizer, Tokenizer
from trainer.checkpoint import PRETRAIN_RESUME_SIGNATURE_VERSION, resume_pretrain
from trainer.cli import add_lm_arguments, option_default, training_options, training_parser
from trainer.eval_utils import evaluate_pretrain_generation
from trainer.lm_utils import (
    build_lm_optimizer,
    LMProgress,
    lm_resume_signature,
    lm_scheduler,
    resolved_lm_options,
    training_loader,
    validate_lm_options,
    validate_model_tokenizer,
    validation_identity,
    write_pretrain_manifest,
)
from trainer.run_utils import (
    check_resume_progress,
    emergency_checkpoint,
    restore_batches,
    RunArtifacts,
    should_log,
    start_tracking,
    start_update,
    training_result,
    truncate_metrics_after,
    update_duration,
)
from trainer.trainer_utils import (
    autocast_context as _autocast_context,
    evaluate_lm,
    resolve_autocast_dtype,
    resolve_device,
    seed_everything,
)


PRETRAIN_IMPLEMENTATION_VERSION = 2


PRETRAIN_INITIALIZATION_SCHEME = "normal_0.02_residual_scaled_1_over_sqrt_2L_v1"


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
    generation_every: int = 0
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


def _resume_signature(
    options: PretrainOptions,
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    train_path: str | Path,
    validation_path: str | Path | None,
    resolved_precision: str,
) -> dict[str, object]:
    train_identity = path_identity(train_path)
    dedicated_identity = path_identity(validation_path) if validation_path is not None else None
    if dedicated_identity == train_identity:
        raise ValueError("dedicated validation data is identical to training data")
    return {
        **lm_resume_signature(options, model, tokenizer, resolved_precision),
        "signature_version": PRETRAIN_RESUME_SIGNATURE_VERSION,
        "implementation_version": PRETRAIN_IMPLEMENTATION_VERSION,
        "sequence_length": options.sequence_length,
        "shuffle_buffer_size": options.shuffle_buffer_size,
        "validation_sampling": "fixed_reservoir_v1",
        "parameter_initialization": PRETRAIN_INITIALIZATION_SCHEME,
        "train_data": train_identity,
        "validation_data": validation_identity(
            train_identity, dedicated_identity, options.validation_fraction, "content_hash_split",
        ),
    }


def run_pretrain_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    train_path: str | Path,
    output_dir: str | Path,
    options: PretrainOptions,
    validation_path: str | Path | None = None,
) -> dict[str, float | str]:
    """Train an already-initialized model on a production JSONL stream.

    The CLI seeds before model construction. Library callers that need
    reproducible from-scratch initialization must do the same before creating
    ``model``; this function seeds data order and subsequent operations.
    """

    validate_lm_options(options, pretrain=True)
    validate_model_tokenizer(model, tokenizer)
    if options.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if options.sequence_length > model.config.max_position_embeddings:
        raise ValueError("sequence_length exceeds the model context length")
    if options.shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size must be non-negative")

    artifacts = RunArtifacts(output_dir, "pretrain")
    output = artifacts.output
    artifacts.check_empty(options.resume_from, "pretraining", artifacts.metrics.name, "best.pt", "final.pt")

    seed_everything(options.seed)
    device = resolve_device(options.device)
    autocast_dtype = resolve_autocast_dtype(options.precision, device)
    resolved_precision = "bf16" if autocast_dtype is torch.bfloat16 else "fp32"
    resume_signature = _resume_signature(
        options,
        model,
        tokenizer,
        train_path,
        validation_path,
        resolved_precision,
    )
    model.to(device).train()
    train_dataset = JsonlPretrainDataset(
        train_path, tokenizer, options.sequence_length, split="train",
        validation_fraction=options.validation_fraction,
        shuffle_buffer_size=options.shuffle_buffer_size,
        seed=options.seed,
    )
    # The stream only emits full packs, so no padding mask is required.
    loader = training_loader(train_dataset, tokenizer, options)
    validation_dataset = None
    if validation_path or options.validation_fraction > 0:
        validation_dataset = JsonlPretrainDataset(
            validation_path or train_path,
            tokenizer,
            options.sequence_length,
            split="all" if validation_path else "validation",
            validation_fraction=options.validation_fraction,
        )
    optimizer = build_lm_optimizer(model, options)
    scheduler = lm_scheduler(optimizer, options)
    payload = resume_pretrain(model, optimizer, scheduler, resume_signature, options, device)
    progress = LMProgress.from_checkpoint(payload, options, legacy_pretrain=True)
    if payload is not None:
        check_resume_progress(progress.step, options.steps, options.wandb_run_id, progress.wandb_run_id)
        truncate_metrics_after(artifacts.metrics, progress.step)
        print(f"resuming from step={progress.step}; skipping {progress.micro_batches} consumed micro-batches", flush=True)
    batches = restore_batches(loader, progress.micro_batches, payload)

    validation_loader = None
    if validation_dataset is not None:
        validation_loader = reservoir_sample_lm_batches(
            validation_dataset,
            batch_size=options.batch_size,
            batches=options.validation_batches,
            pad_token_id=tokenizer.pad_token_id,
            seed=options.seed + 1,
            collate=default_collate,
        )

    write_pretrain_manifest(
        artifacts.manifest,
        model=model,
        options=options,
        resume_signature=resume_signature,
        train_path=train_path,
        validation_path=validation_path,
        resumed_step=progress.step,
        resolved_precision=resolved_precision,
    )
    tracker, progress.wandb_run_id = start_tracking(options, output, progress.step, progress.wandb_run_id, {
        "stage": "pretrain", "data": str(Path(train_path)),
        "validation_data": str(Path(validation_path)) if validation_path else None,
        "output": str(output), "num_parameters": model.num_parameters, "model": asdict(model.config),
        "training": resolved_lm_options(options), "resume_identity": resume_signature,
        "manifest": str(artifacts.manifest),
    })

    def training_state() -> dict[str, object]:
        return progress.state_dict(resolved_lm_options(options), resume_signature)

    save = artifacts.checkpoint_writer(model, optimizer, scheduler, training_state)
    try:
        for step in range(progress.step + 1, options.steps + 1):
            validation_due = validation_loader is not None and (
                step % options.validation_every == 0 or step == options.steps
            )
            generation_due = bool(options.generation_every) and step % options.generation_every == 0
            log_due = should_log(step, options, validation_due, generation_due)
            update_started = start_update(device) if log_due else None
            optimizer.zero_grad(set_to_none=True)
            loss_times_targets = 0.0
            step_tokens = 0
            step_target_tokens = 0
            current_lr = float(optimizer.param_groups[0]["lr"])
            for micro_step in range(1, options.gradient_accumulation_steps + 1):
                batch = {name: value.to(device) for name, value in next(batches).items()}
                with _autocast_context(device, autocast_dtype):
                    result = model(**batch)
                if result.loss is None:
                    raise RuntimeError("model did not return a pretraining loss")
                loss = float(result.loss.detach())
                if not math.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite pretraining loss at optimizer step {step}, micro step {micro_step}: "
                        f"{loss}"
                    )
                batch_size, sequence_length = batch["input_ids"].shape
                target_count = batch_size * (sequence_length - 1)
                (result.loss / options.gradient_accumulation_steps).backward()
                loss_times_targets += loss * target_count
                step_tokens += batch_size * sequence_length
                step_target_tokens += target_count
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), options.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            update_seconds = update_duration(update_started, device) if update_started is not None else None
            progress.advance(step, options.gradient_accumulation_steps, step_tokens, step_target_tokens)
            last_loss = loss_times_targets / step_target_tokens
            metric = progress.training_metrics(
                "pretrain", last_loss, current_lr, grad_norm, update_seconds, device, grad_clip=options.grad_clip,
                tokens=step_tokens, targets=step_target_tokens, samples=options.batch_size * options.gradient_accumulation_steps,
            )
            val_loss = None
            if validation_due:
                val_loss = evaluate_lm(
                    model,
                    validation_loader,
                    device,
                    options.validation_batches,
                    autocast_dtype=autocast_dtype,
                )
                metric["validation_loss"] = val_loss
                metric["perplexity"] = math.exp(val_loss) if val_loss < 709.0 else float("inf")
            progress.last_metrics = progress.loss_metrics(last_loss, current_lr)
            if val_loss is not None:
                progress.last_metrics["validation_loss"] = val_loss
                progress.last_metrics["perplexity"] = float(metric["perplexity"])
                if math.isfinite(val_loss) and val_loss < progress.best_val_loss:
                    progress.best_val_loss = val_loss
                    progress.last_metrics["best_val_loss"] = progress.best_val_loss
                    best_checkpoint = save(output / "best.pt", step, progress.last_metrics)
                    print(
                        f"saved best checkpoint: {best_checkpoint} (validation_loss={progress.best_val_loss:.6f})",
                        flush=True,
                    )
                metric["best_val_loss"] = progress.best_val_loss
            generation_path: Path | None = None
            if generation_due:
                generation_path = evaluate_pretrain_generation(
                    model,
                    tokenizer,
                    output,
                    step=step,
                    device=device,
                    max_new_tokens=options.generation_max_new_tokens,
                    autocast_dtype=autocast_dtype,
                )
                print(f"saved generation evaluation: {generation_path}", flush=True)
            if log_due:
                artifacts.log(metric, tracker, generation_path)
            artifacts.save_periodic(save, step, progress.last_metrics, options, label='')
    except KeyboardInterrupt:
        emergency_checkpoint(artifacts, save, progress.step, progress.last_metrics, tracker)
        raise

    metrics = progress.loss_metrics(last_loss, progress.last_metrics["learning_rate"])
    checkpoint = save(output / "final.pt", options.steps, metrics)
    return training_result(artifacts, checkpoint, metrics, device, tracker, progress.wandb_run_id, options.steps)


def build_parser() -> argparse.ArgumentParser:
    parser = training_parser('Pretrain the 64M base model.', 'pretrain', DEFAULT_PRETRAIN_DATA, PretrainOptions)
    add_lm_arguments(parser, PretrainOptions)
    default = partial(option_default, PretrainOptions)
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=20,
        help="decoder layer count for a new model (default: 20; must match when resuming)",
    )
    parser.add_argument("--sequence-length", type=int, default=default("sequence_length"))
    parser.add_argument(
        "--shuffle-buffer-size", type=int, default=default("shuffle_buffer_size"),
        help="use 0 to preserve JSONL order",
    )
    parser.add_argument("--resume", type=Path, help="resume from a full training checkpoint")
    parser.add_argument(
        "--allow-legacy-resume", action="store_true",
        help="accept a pre-v2 checkpoint without strict data/tokenizer identity checks",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    tokenizer = load_tokenizer(arguments.tokenizer)
    # The seed controls initial weights as well as the training trajectory.
    seed_everything(arguments.seed)
    model = MiniScaleForCausalLM(MiniScaleConfig.small_64m(
        tokenizer.vocab_size,
        arguments.sequence_length,
        num_hidden_layers=arguments.num_hidden_layers,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    options = training_options(PretrainOptions, arguments)
    result = run_pretrain_jsonl(
        model, tokenizer, arguments.data, arguments.output, options,
        validation_path=arguments.validation_data,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
