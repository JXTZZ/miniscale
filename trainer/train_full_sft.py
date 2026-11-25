"""Fine-tune on assistant turns and evaluate generation quality."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import math
from pathlib import Path
import sys

import torch

# Support running this file directly as well as with python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import atomic_write_json, path_identity
from dataset.lm_dataset import collate_lm_batch, sample_indexed_batches
from dataset.sft_dataset import (
    DEFAULT_SFT_DATA,
    IndexedJsonlSFTDataset,
    SFT_DATA_ORDER_VERSION,
    SFT_EXAMPLE_FORMAT_VERSION,
    SFT_SELECTION_VERSION,
    SFT_TRUNCATION_VERSION,
    SFTCorpusIndex,
)
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.checkpoint import check_resume_signature, read_strict_checkpoint, restore_training_checkpoint
from trainer.cli import add_finetune_arguments, option_default, run_checkpoint_stage, training_parser
from trainer.eval_utils import DEFAULT_SFT_GENERATION_SUITE
from trainer.lm_utils import (
    build_lm_optimizer,
    clip_finetune_gradients,
    finetune_context,
    finetune_manifest,
    indexed_training_data,
    LMProgress,
    lm_resume_signature,
    lm_scheduler,
    resolved_lm_options,
    training_loader,
    validate_lm_options,
    validate_model_tokenizer,
    validation_identity,
)
from trainer.run_utils import (
    check_resume_progress,
    emergency_checkpoint,
    mirror_checkpoint,
    restore_batches,
    RunArtifacts,
    should_log,
    start_tracking,
    start_update,
    training_result,
    truncate_metrics_after,
    update_duration,
)
from trainer.sft_eval import evaluate_sft, run_sft_generation_evaluation, SFTQuality
from trainer.trainer_utils import autocast_context, resolve_autocast_dtype, resolve_device, seed_everything


SFT_RESUME_SIGNATURE_VERSION = 1


SFT_IMPLEMENTATION_VERSION = 2


SFT_MASK_VERSION = "structured_assistant_span_v1"


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


def validate_sft_options(options: SFTOptions, model, tokenizer) -> int:
    validate_lm_options(options)
    validate_model_tokenizer(model, tokenizer)
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
    return finetune_context(options, model, options.target_mode)


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
    return {
        **lm_resume_signature(options, model, tokenizer, resolved_precision),
        "signature_version": SFT_RESUME_SIGNATURE_VERSION,
        "implementation_version": SFT_IMPLEMENTATION_VERSION,
        "max_length": max_length,
        "min_context_tokens": options.min_context_tokens,
        "target_mode": options.target_mode,
        "example_format": SFT_EXAMPLE_FORMAT_VERSION,
        "selection_version": SFT_SELECTION_VERSION,
        "mask_version": SFT_MASK_VERSION,
        "truncation_version": SFT_TRUNCATION_VERSION,
        "data_order": SFT_DATA_ORDER_VERSION,
        "deduplicate_exact": options.deduplicate_exact,
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
        "initial_checkpoint": initial_checkpoint,
        "train_data": train_index.identity,
        "validation_data": validation_identity(
            train_index.identity, validation_index.identity if validation_index else None,
            options.validation_fraction, "conversation_hash_split",
        ),
    }


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
    artifacts = RunArtifacts(output_dir, "sft")
    output = artifacts.output
    artifacts.check_empty(options.resume_from, "SFT", artifacts.manifest.name, artifacts.metrics.name, "best.pt", "best_loss.pt", "best_quality.pt", "sft.pt")

    seed_everything(options.seed)
    device = resolve_device(options.device)
    autocast_dtype = resolve_autocast_dtype(options.precision, device)
    resolved_precision = "bf16" if autocast_dtype is torch.bfloat16 else "fp32"

    train_index, validation_index, train_dataset, validation_batches = indexed_training_data(
        SFTCorpusIndex.build, IndexedJsonlSFTDataset, sample_indexed_batches,
        options=options, tokenizer=tokenizer, data_path=data_path, validation_path=validation_path,
        max_length=max_length, target_mode=options.target_mode, label="SFT", unit="examples",
    )
    loader = training_loader(train_dataset, tokenizer, options, collate_lm_batch, shuffle=True)
    if options.early_stopping_patience and validation_batches is None:
        raise ValueError("early stopping requires a non-empty validation dataset")

    model.to(device).train()
    optimizer = build_lm_optimizer(model, options)
    scheduler = lm_scheduler(optimizer, options)

    payload, state = read_strict_checkpoint(options.resume_from, device, "sft", "SFT")
    saved_signature = state.get("resume_signature")
    initial_checkpoint_source = None
    if saved_signature is not None:
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
    check_resume_signature(saved_signature, signature, "SFT")

    progress = LMProgress.from_checkpoint(payload, options)
    examples_seen = int(state.get("examples_seen", progress.micro_batches * options.batch_size))
    clipped_steps = int(state.get("clipped_steps", 0))
    quality = SFTQuality.from_checkpoint(state)
    if payload is not None:
        check_resume_progress(progress.step, options.steps, options.wandb_run_id, progress.wandb_run_id)
        restore_training_checkpoint(payload, model, optimizer, scheduler, restore_rng=False)
        truncate_metrics_after(artifacts.metrics, progress.step)
        print(f"resuming SFT from step={progress.step}; replaying {progress.micro_batches} micro-batches", flush=True)
    batches = restore_batches(loader, progress.micro_batches, payload)

    manifest = finetune_manifest(
        "sft", model, options, resolved_precision, max_length, data_path, validation_path,
        train_index, validation_index, initial_checkpoint_source, initial_checkpoint, progress.step, signature,
    )
    atomic_write_json(artifacts.manifest, manifest)
    tracker, progress.wandb_run_id = start_tracking(
        options, output, progress.step, progress.wandb_run_id,
        {**manifest, "manifest": str(artifacts.manifest)},
    )

    def training_state() -> dict[str, object]:
        return progress.state_dict(
            resolved_lm_options(options), signature, examples_seen=examples_seen,
            **asdict(quality),
            clipped_steps=clipped_steps, initial_checkpoint_source=initial_checkpoint_source,
        )

    save = artifacts.checkpoint_writer(model, optimizer, scheduler, training_state)
    try:
        for step in range(progress.step + 1, options.steps + 1):
            validation_due = validation_batches is not None and (
                step % options.validation_every == 0 or step == options.steps
            )
            generation_due = bool(options.generation_every) and step % options.generation_every == 0
            log_due = should_log(step, options, validation_due, generation_due)
            update_started = start_update(device) if log_due else None
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
                loss = float(result.loss.detach())
                if not math.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite SFT loss at optimizer step {step}, micro step {micro_step}: "
                        f"{loss}"
                    )
                (result.loss * (target_count / total_targets)).backward()
                weighted_loss += loss * target_count
                step_input_tokens += int(cpu_batch["attention_mask"].sum())
                step_examples += cpu_batch["input_ids"].shape[0]
            grad_norm = clip_finetune_gradients(model, options.grad_clip, "SFT", step)
            optimizer.step()
            scheduler.step()
            update_seconds = update_duration(update_started, device) if update_started is not None else None
            progress.advance(step, options.gradient_accumulation_steps, step_input_tokens, total_targets)
            examples_seen += step_examples
            last_loss = weighted_loss / total_targets
            metric = progress.training_metrics(
                "sft", last_loss, current_lr, grad_norm, update_seconds, device, grad_clip=options.grad_clip,
                tokens=step_input_tokens, targets=total_targets, samples=step_examples,
            )
            metric["examples_seen"] = examples_seen
            clipped_steps += int(bool(metric["grad_was_clipped"]))
            metric["grad_clip_fraction"] = clipped_steps / progress.step
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
                quality.record_loss(val_loss, options.early_stopping_patience)
                if math.isfinite(val_loss) and val_loss < progress.best_val_loss:
                    progress.best_val_loss = val_loss
                    best_metrics = {
                        "loss": last_loss,
                        "validation_loss": val_loss,
                        "validation_token_accuracy": val_accuracy,
                        "best_val_loss": progress.best_val_loss,
                    }
                    best_checkpoint = save(output / "best_loss.pt", step, best_metrics)
                    print(f"saved best-loss SFT checkpoint: {best_checkpoint}", flush=True)
                metric["best_val_loss"] = progress.best_val_loss

            progress.last_metrics = progress.loss_metrics(last_loss, current_lr, examples_seen=float(examples_seen))
            if validation_due:
                progress.last_metrics.update({
                    "validation_loss": float(metric["validation_loss"]),
                    "validation_token_accuracy": float(metric["validation_token_accuracy"]),
                    "perplexity": float(metric["perplexity"]),
                })

            generation_path: Path | None = None
            if generation_due:
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
                quality_score, improved = quality.record_generation(generation_summary, options, step)
                metric["generation_quality_score"] = quality_score
                if improved:
                    quality_metrics = {
                        "loss": last_loss,
                        "best_val_loss": progress.best_val_loss,
                        "best_quality_score": quality.best_quality_score,
                        "best_quality_step": float(quality.best_quality_step),
                        **generation_summary,
                    }
                    quality_checkpoint = save(output / "best_quality.pt", step, quality_metrics)
                    mirror_checkpoint(quality_checkpoint, output / "best.pt")
                    print(
                        f"saved best-quality SFT checkpoint: {quality_checkpoint} "
                        f"(score={quality.best_quality_score:.4f})",
                        flush=True,
                    )
                metric["best_quality_score"] = quality.best_quality_score
                metric["best_quality_step"] = quality.best_quality_step
                metric["stale_quality_evaluations"] = quality.stale_quality_evaluations
            if log_due:
                artifacts.log(metric, tracker, generation_path)
            artifacts.save_periodic(save, step, progress.last_metrics, options, label='SFT')
            if validation_due and generation_path is not None:
                reason = quality.stop_reason(options, step)
                if reason is not None:
                    print(f"early stopping SFT at step={step}: {reason}", flush=True)
                    break
    except (KeyboardInterrupt, FloatingPointError):
        emergency_checkpoint(artifacts, save, progress.step, progress.last_metrics, tracker, 'SFT')
        raise

    metrics = progress.loss_metrics(
        last_loss, float(progress.last_metrics["learning_rate"]), examples_seen=float(examples_seen),
        best_quality_score=quality.best_quality_score, best_quality_step=float(quality.best_quality_step),
        stopped_early=float(progress.step < options.steps), grad_clip_fraction=clipped_steps / max(progress.step, 1),
    )
    checkpoint = save(output / "sft.pt", progress.step, metrics)
    return training_result(artifacts, checkpoint, metrics, device, tracker, progress.wandb_run_id, progress.step)


def build_parser() -> argparse.ArgumentParser:
    parser = training_parser('Fine-tune a pretraining checkpoint.', 'sft', DEFAULT_SFT_DATA, SFTOptions)
    add_finetune_arguments(parser, SFTOptions)
    default = partial(option_default, SFTOptions)
    parser.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default=default("target_mode"),
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
    return parser


def main(argv: list[str] | None = None) -> None:
    run_checkpoint_stage(build_parser(), SFTOptions, run_sft_jsonl, argv)


if __name__ == "__main__":
    main()
