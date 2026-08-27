from __future__ import annotations

from dataclasses import asdict
from functools import partial
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from miniscale.data import SFTDataset, collate_lm_batch
from miniscale.integrity import path_identity
from miniscale.model import MiniScaleForCausalLM
from miniscale.sft_data import (
    IndexedJsonlSFTDataset,
    SFTCorpusIndex,
    fixed_validation_batches,
)
from miniscale.tokenizer import ByteTokenizer, Tokenizer
from miniscale.tracking import WandbTracker
from .common import (
    TRAINING_CHECKPOINT_FORMAT_VERSION,
    append_metric,
    atomic_write_json,
    autocast_context,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    infinite_batches,
    prune_periodic_checkpoints,
    read_training_checkpoint,
    resolve_autocast_dtype,
    resolve_device,
    restore_rng_state,
    restore_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    seed_everything,
    seed_worker,
    signature_differences,
    truncate_metrics_after,
)
from .sft_config import (
    SFT_IMPLEMENTATION_VERSION,
    SFTOptions,
    SmokeSFTOptions,
    resolved_sft_options,
    sft_option_default,
    sft_resume_signature,
    validate_sft_options,
)
from .sft_evaluation import evaluate_sft, run_sft_generation_evaluation


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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    batches = iter(infinite_batches(loader))
    losses: list[float] = []
    for _ in range(options.steps):
        batch = {name: value.to(device) for name, value in next(batches).items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if output.loss is None:
            raise RuntimeError("model did not return an SFT loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        losses.append(float(output.loss.detach()))
    metrics = {"loss": losses[-1], "mean_loss": sum(losses) / len(losses)}
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
            for path in (manifest_path, metrics_path, output / "best.pt", output / "sft.pt")
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
                if math.isfinite(val_loss) and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_metrics = {
                        "loss": last_loss,
                        "validation_loss": val_loss,
                        "validation_token_accuracy": val_accuracy,
                        "best_val_loss": best_val_loss,
                    }
                    best_checkpoint = save_training_checkpoint(
                        output / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        stage="sft",
                        step=step,
                        metrics=best_metrics,
                        training_state=training_state(),
                    )
                    print(f"saved best SFT checkpoint: {best_checkpoint}", flush=True)
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
                generation_path = run_sft_generation_evaluation(
                    model,
                    tokenizer,
                    output,
                    step=step,
                    device=device,
                    max_new_tokens=options.generation_max_new_tokens,
                    autocast_dtype=autocast_dtype,
                )
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
                print(metric, flush=True)
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
    }
    checkpoint = save_training_checkpoint(
        output / "sft.pt",
        model,
        optimizer,
        scheduler,
        stage="sft",
        step=options.steps,
        metrics=metrics,
        training_state=training_state(),
    )
    if tracker is not None:
        tracker.finish(summary={**metrics, "final_step": options.steps})
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
