"""Optimize paired preferences against a frozen reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys

import torch

# Support running this file directly as well as with python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import atomic_write_json, path_identity
from dataset.lm_dataset import sample_indexed_batches
from dataset.preference_dataset import (
    collate_preference_batch,
    DEFAULT_DPO_DATA,
    DPO_DATA_ORDER_VERSION,
    DPO_PAIR_FORMAT_VERSION,
    DPO_TRUNCATION_VERSION,
    IndexedPreferenceDataset,
    PreferenceCorpusIndex,
)
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.checkpoint import (
    check_resume_signature,
    freeze_reference,
    load_reference,
    read_checkpoint,
    read_strict_checkpoint,
    restore_training_checkpoint,
    save_checkpoint,
)
from trainer.cli import add_finetune_arguments, option_default, run_checkpoint_stage, training_parser
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
from trainer.preference_objective import (
    checked_dpo_metrics,
    completion_log_probability,
    dpo_batch_metrics,
    evaluate_dpo,
    run_dpo_generation_evaluation,
)
from trainer.runtime.run import (
    check_resume_progress,
    TrainingStep,
    restore_batches,
    RunArtifacts,
    should_log,
    start_tracking,
    start_update,
    training_result,
    truncate_metrics_after,
    update_duration,
)
from trainer.runtime.torch_utils import autocast_context, resolve_autocast_dtype, resolve_device, seed_everything


DPO_RESUME_SIGNATURE_VERSION = 1


DPO_IMPLEMENTATION_VERSION = 2


DPO_OBJECTIVE_VERSION = "sigmoid_sum_completion_logp_v1"


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
    generation_every: int = 0
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


def validate_dpo_options(options: DPOOptions, model, tokenizer, *, target_mode: str) -> int:
    validate_lm_options(options)
    validate_model_tokenizer(model, tokenizer)
    if options.beta <= 0:
        raise ValueError("beta must be positive")
    return finetune_context(options, model, target_mode)


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
    return {
        **lm_resume_signature(options, model, tokenizer, resolved_precision),
        "signature_version": DPO_RESUME_SIGNATURE_VERSION,
        "implementation_version": DPO_IMPLEMENTATION_VERSION,
        "objective": DPO_OBJECTIVE_VERSION,
        "max_length": max_length,
        "min_context_tokens": options.min_context_tokens,
        "target_mode": target_mode,
        "pair_format": DPO_PAIR_FORMAT_VERSION,
        "truncation_version": DPO_TRUNCATION_VERSION,
        "data_order": DPO_DATA_ORDER_VERSION,
        "deduplicate_exact": options.deduplicate_exact,
        "beta": options.beta,
        "validation_sampling": "fixed_global_pair_sample_v1",
        "initial_checkpoint": initial_checkpoint,
        "reference": {"kind": "frozen_initial_policy", "checkpoint": initial_checkpoint},
        "train_data": train_index.identity,
        "validation_data": validation_identity(
            train_index.identity, validation_index.identity if validation_index else None,
            options.validation_fraction, "prompt_hash_split",
        ),
    }


def _checkpoint_target_mode(payload: dict[str, object]) -> str | None:
    state = payload.get("training_state")
    if not isinstance(state, dict):
        return None
    resolved = state.get("resolved_options")
    if isinstance(resolved, dict) and resolved.get("target_mode") in {
        "reasoning_and_response",
        "response_only",
    }:
        return str(resolved["target_mode"])
    signature = state.get("resume_signature")
    if isinstance(signature, dict) and signature.get("target_mode") in {
        "reasoning_and_response",
        "response_only",
    }:
        return str(signature["target_mode"])
    return None


def _inspect_sft_checkpoint(path: str | Path) -> tuple[dict[str, object], str | None]:
    payload = read_checkpoint(path, "cpu")
    if payload.get("stage") != "sft":
        raise ValueError("--checkpoint must be an SFT checkpoint")
    return path_identity(path), _checkpoint_target_mode(payload)


def run_dpo_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    options: DPOOptions,
    *,
    validation_path: str | Path | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    """Run reproducible DPO over indexed, shared-prompt preference pairs."""

    artifacts = RunArtifacts(output_dir, "dpo")
    output = artifacts.output
    artifacts.check_empty(options.resume_from, "DPO", artifacts.manifest.name, artifacts.metrics.name, "best.pt", "dpo.pt", "reference.pt")

    seed_everything(options.seed)
    device = resolve_device(options.device)
    autocast_dtype = resolve_autocast_dtype(options.precision, device)
    resolved_precision = "bf16" if autocast_dtype is torch.bfloat16 else "fp32"

    payload, state = read_strict_checkpoint(options.resume_from, device, "dpo", "DPO")
    saved_signature = state.get("resume_signature")
    parent_target_mode = initial_checkpoint_source = None
    if saved_signature is not None:
        parent_target_mode = str(saved_signature.get("target_mode"))
        initial_checkpoint = saved_signature.get("initial_checkpoint")
        if not isinstance(initial_checkpoint, dict):
            raise ValueError("DPO checkpoint is missing its SFT initialization identity")
        if state.get("initial_checkpoint_source") is not None:
            initial_checkpoint_source = str(state["initial_checkpoint_source"])
    elif initial_checkpoint_path is not None:
        initial_checkpoint, parent_target_mode = _inspect_sft_checkpoint(initial_checkpoint_path)
        initial_checkpoint_source = str(Path(initial_checkpoint_path).resolve())
    else:
        initial_checkpoint = {"kind": "in_memory_model"}

    target_mode = options.target_mode or parent_target_mode or "reasoning_and_response"
    if parent_target_mode is not None and target_mode != parent_target_mode:
        raise ValueError(
            f"DPO target_mode {target_mode!r} does not match parent SFT mode {parent_target_mode!r}"
        )
    max_length = validate_dpo_options(options, model, tokenizer, target_mode=target_mode)

    train_index, validation_index, train_dataset, validation_batches = indexed_training_data(
        PreferenceCorpusIndex.build, IndexedPreferenceDataset,
        partial(sample_indexed_batches, collate=collate_preference_batch),
        options=options, tokenizer=tokenizer, data_path=data_path, validation_path=validation_path,
        max_length=max_length, target_mode=target_mode, label="DPO", unit="pairs",
    )
    loader = training_loader(train_dataset, tokenizer, options, collate_preference_batch, shuffle=True)
    model.to(device).train()
    reference = freeze_reference(model, device)
    optimizer = build_lm_optimizer(model, options)
    scheduler = lm_scheduler(optimizer, options)

    signature = dpo_resume_signature(
        options,
        model,
        tokenizer,
        max_length=max_length,
        target_mode=target_mode,
        train_index=train_index,
        validation_index=validation_index,
        resolved_precision=resolved_precision,
        initial_checkpoint=initial_checkpoint,
    )
    check_resume_signature(saved_signature, signature, "DPO")

    progress = LMProgress.from_checkpoint(payload, options)
    pairs_seen = int(state.get("pairs_seen", progress.micro_batches * options.batch_size))
    reference_path = output / "reference.pt"
    if payload is not None:
        check_resume_progress(progress.step, options.steps, options.wandb_run_id, progress.wandb_run_id)
        reference_identity = state.get("reference_identity")
        if reference_identity is None:
            raise ValueError("DPO checkpoint is missing its frozen reference identity")
        load_reference(reference_path, "dpo", reference_identity, device, reference, "DPO")
        restore_training_checkpoint(payload, model, optimizer, scheduler, restore_rng=False)
        truncate_metrics_after(artifacts.metrics, progress.step)
        print(f"resuming DPO from step={progress.step}; replaying {progress.micro_batches} micro-batches", flush=True)
    else:
        save_checkpoint(reference_path, reference, stage="dpo_reference", step=0, metrics={})
        reference_identity = path_identity(reference_path)
    batches = restore_batches(loader, progress.micro_batches, payload)

    manifest = finetune_manifest(
        "dpo", model, options, resolved_precision, max_length, data_path, validation_path,
        train_index, validation_index, initial_checkpoint_source, initial_checkpoint, progress.step, signature,
    )
    manifest["reference"] = {"checkpoint": "reference.pt", "identity": reference_identity}
    atomic_write_json(artifacts.manifest, manifest)
    tracker, progress.wandb_run_id = start_tracking(
        options, output, progress.step, progress.wandb_run_id,
        {**manifest, "manifest": str(artifacts.manifest)},
    )

    def training_state() -> dict[str, object]:
        return progress.state_dict(
            resolved_lm_options(options, target_mode=target_mode), signature, pairs_seen=pairs_seen,
            initial_checkpoint_source=initial_checkpoint_source, reference_checkpoint="reference.pt",
            reference_identity=reference_identity,
        )

    save = artifacts.checkpoint_writer(model, optimizer, scheduler, training_state)
    boundary = TrainingStep()
    try:
        for step in range(progress.step + 1, options.steps + 1):
            boundary.begin()
            validation_due = validation_batches is not None and (
                step % options.validation_every == 0 or step == options.steps
            )
            generation_due = bool(options.generation_every) and step % options.generation_every == 0
            log_due = should_log(step, options, validation_due, generation_due)
            update_started = start_update(device) if log_due else None
            optimizer.zero_grad(set_to_none=True)
            cpu_batches = [next(batches) for _ in range(options.gradient_accumulation_steps)]
            pair_counts = [batch["input_ids"].shape[0] // 2 for batch in cpu_batches]
            total_pairs = sum(pair_counts)
            if total_pairs < 1:
                raise RuntimeError("DPO optimizer update contains no preference pairs")
            accumulated: dict[str, float] = {}
            step_input_tokens = step_target_tokens = 0
            current_lr = float(optimizer.param_groups[0]["lr"])
            for micro_step, (cpu_batch, pair_count) in enumerate(
                zip(cpu_batches, pair_counts, strict=True), 1
            ):
                batch = {name: value.to(device) for name, value in cpu_batch.items()}
                with torch.no_grad(), autocast_context(device, autocast_dtype):
                    reference_chosen, reference_rejected = completion_log_probability(reference, batch).chunk(2)
                with autocast_context(device, autocast_dtype):
                    policy_chosen, policy_rejected = completion_log_probability(model, batch).chunk(2)
                loss, batch_metrics = dpo_batch_metrics(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    options.beta,
                )
                values = checked_dpo_metrics(batch_metrics, f"optimizer step {step}, micro step {micro_step}")
                (loss * (pair_count / total_pairs)).backward()
                for name, value in values.items():
                    accumulated[name] = accumulated.get(name, 0.0) + value * pair_count
                step_input_tokens += int(cpu_batch["attention_mask"].sum())
                step_target_tokens += int(cpu_batch["labels"][:, 1:].ne(-100).sum())
            grad_norm = clip_finetune_gradients(model, options.grad_clip, "DPO", step)
            boundary.updating()
            optimizer.step()
            scheduler.step()
            update_seconds = update_duration(update_started, device) if update_started is not None else None
            progress.advance(step, options.gradient_accumulation_steps, step_input_tokens, step_target_tokens)
            pairs_seen += total_pairs
            averaged = {name: value / total_pairs for name, value in accumulated.items()}
            last_loss = averaged["loss"]
            metric = progress.training_metrics(
                "dpo", last_loss, current_lr, grad_norm, update_seconds, device, grad_clip=options.grad_clip,
                tokens=step_input_tokens, targets=step_target_tokens, samples=total_pairs,
            )
            metric.update({
                "pairs_seen": pairs_seen,
                "preference_accuracy": averaged["reward_accuracy"],
                **{name: value for name, value in averaged.items() if name != "loss"},
            })

            if validation_due:
                validation_metrics = evaluate_dpo(
                    model,
                    reference,
                    validation_batches,
                    device,
                    beta=options.beta,
                    autocast_dtype=autocast_dtype,
                )
                metric.update(validation_metrics)
                validation_loss = float(validation_metrics["validation_loss"])
                if validation_loss < progress.best_val_loss:
                    progress.best_val_loss = validation_loss
                    best_checkpoint = save(output / "best.pt", step, {
                        "loss": last_loss, "validation_loss": validation_loss,
                        "validation_reward_accuracy": float(validation_metrics["validation_reward_accuracy"]),
                        "best_val_loss": progress.best_val_loss,
                    })
                    print(f"saved best DPO checkpoint: {best_checkpoint}", flush=True)
                metric["best_val_loss"] = progress.best_val_loss

            progress.last_metrics = progress.loss_metrics(
                last_loss, current_lr, preference_accuracy=averaged["reward_accuracy"], pairs_seen=float(pairs_seen),
            )
            if validation_due:
                progress.last_metrics.update({
                    "validation_loss": float(metric["validation_loss"]),
                    "validation_reward_accuracy": float(metric["validation_reward_accuracy"]),
                })

            generation_path: Path | None = None
            if generation_due:
                generation_path = run_dpo_generation_evaluation(
                    model,
                    reference,
                    tokenizer,
                    output,
                    step=step,
                    device=device,
                    max_new_tokens=options.generation_max_new_tokens,
                    autocast_dtype=autocast_dtype,
                )
            boundary.complete()
            if log_due:
                artifacts.log(metric, tracker, generation_path)
            artifacts.save_periodic(save, step, progress.last_metrics, options, label='DPO')
    except (KeyboardInterrupt, FloatingPointError):
        boundary.save_emergency(artifacts, save, progress.step, progress.last_metrics, tracker, 'DPO')
        raise

    metrics = progress.loss_metrics(
        last_loss, float(progress.last_metrics["learning_rate"]),
        preference_accuracy=float(progress.last_metrics["preference_accuracy"]), pairs_seen=float(pairs_seen),
    )
    checkpoint = save(output / "dpo.pt", options.steps, metrics)
    return training_result(artifacts, checkpoint, metrics, device, tracker, progress.wandb_run_id, options.steps)


def build_parser() -> argparse.ArgumentParser:
    parser = training_parser('Optimize preferences from an SFT checkpoint.', 'dpo', DEFAULT_DPO_DATA, DPOOptions)
    add_finetune_arguments(parser, DPOOptions)
    default = partial(option_default, DPOOptions)
    parser.add_argument("--beta", type=float, default=default("beta"))
    parser.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default=None,
        help="defaults to and must match the completion semantics stored by the parent SFT run",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    run_checkpoint_stage(build_parser(), DPOOptions, run_dpo_jsonl, argv)


if __name__ == "__main__":
    main()
