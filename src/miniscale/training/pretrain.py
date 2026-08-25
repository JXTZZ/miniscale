from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from miniscale.data import JsonlPretrainDataset, PretrainDataset, collate_lm_batch
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer, Tokenizer
from miniscale.tracking import WandbTracker
from .common import (
    append_metric,
    evaluate_lm,
    infinite_batches,
    load_training_checkpoint,
    resolve_device,
    restore_rng_state,
    save_checkpoint,
    save_training_checkpoint,
    seed_everything,
)


@dataclass(slots=True)
class PretrainOptions:
    steps: int = 100
    batch_size: int = 8
    sequence_length: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    gradient_accumulation_steps: int = 1
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
    resume_from: str | Path | None = None


GENERATION_EVAL_PROMPTS: tuple[dict[str, str], ...] = (
    {"name": "chinese", "language": "zh", "prompt": "人工智能的发展将会"},
    {"name": "english", "language": "en", "prompt": "The future of artificial intelligence is"},
    {
        "name": "code",
        "language": "python",
        "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    },
)


def warmup_cosine_multiplier(
    step_index: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """LR multiplier for the update at zero-based ``step_index``."""

    if warmup_steps > 0 and step_index < warmup_steps:
        return (step_index + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return 1.0
    if warmup_steps == 0:
        progress = step_index / max(total_steps - 1, 1)
    else:
        progress = (step_index - warmup_steps + 1) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    peak_lr = float(optimizer.defaults["lr"])
    if peak_lr <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= min_learning_rate <= peak_lr:
        raise ValueError("min_learning_rate must be between zero and learning_rate")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    effective_warmup_steps = min(warmup_steps, total_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_index: warmup_cosine_multiplier(
            step_index,
            total_steps=total_steps,
            warmup_steps=effective_warmup_steps,
            min_lr_ratio=min_learning_rate / peak_lr,
        ),
    )


def _resume_signature(options: PretrainOptions) -> dict[str, object]:
    return {
        "total_steps": options.steps,
        "batch_size": options.batch_size,
        "sequence_length": options.sequence_length,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "learning_rate": options.learning_rate,
        "min_learning_rate": options.min_learning_rate,
        "warmup_steps": options.warmup_steps,
        "shuffle_buffer_size": options.shuffle_buffer_size,
        "seed": options.seed,
    }


def _validate_resume_signature(saved: object, current: dict[str, object]) -> None:
    if not isinstance(saved, dict):
        raise ValueError("checkpoint resume_signature must be a mapping")
    # Older checkpoints do not contain newly introduced data-order fields.
    # Validate every field they did record, while allowing new fields to use
    # their current defaults during a one-time migration.
    mismatches = {
        name: (value, current.get(name))
        for name, value in saved.items()
        if name not in current or current[name] != value
    }
    if mismatches:
        raise ValueError(f"resume options do not match checkpoint: {mismatches}")


def _prune_periodic_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink()


def _truncate_metrics_after(path: Path, step: int) -> None:
    if not path.exists():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            metric = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(metric.get("step", -1)) <= step:
            retained.append(json.dumps(metric, ensure_ascii=False))
    path.write_text("".join(f"{line}\n" for line in retained), encoding="utf-8")


@torch.no_grad()
def run_generation_evaluation(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    step: int,
    device: torch.device,
    max_new_tokens: int,
) -> Path:
    """Generate fixed multilingual probes with deterministic greedy decoding."""

    was_training = model.training
    model.eval()
    samples: list[dict[str, object]] = []
    try:
        for probe in GENERATION_EVAL_PROMPTS:
            prompt_ids = tokenizer.encode(probe["prompt"], bos=True)
            if len(prompt_ids) >= model.config.max_position_embeddings:
                prompt_ids = prompt_ids[-(model.config.max_position_embeddings - 1) :]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=None,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            completion_ids = generated[0, len(prompt_ids) :].tolist()
            samples.append({
                **probe,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(completion_ids),
                "response": tokenizer.decode(completion_ids),
            })
    finally:
        model.train(was_training)

    target = Path(output_dir) / "generations" / f"step_{step:08d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "stage": "pretrain",
                "step": step,
                "decoding": {"do_sample": False, "strategy": "greedy", "temperature": 0.0},
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def run_pretrain(
    model: MiniScaleForCausalLM,
    tokenizer: ByteTokenizer,
    texts: list[str],
    output_dir: str | Path,
    options: PretrainOptions | None = None,
) -> dict[str, float | str]:
    options = options or PretrainOptions()
    if options.steps < 1:
        raise ValueError("steps must be positive")
    seed_everything(options.seed)
    device = resolve_device(options.device)
    model.to(device).train()
    dataset = PretrainDataset(texts, tokenizer, options.sequence_length)
    if not dataset:
        raise ValueError("pretraining corpus produced no examples")
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_lm_batch(rows, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    losses: list[float] = []
    batches = iter(infinite_batches(loader))
    for _ in range(options.steps):
        batch = {name: value.to(device) for name, value in next(batches).items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if output.loss is None:
            raise RuntimeError("model did not return a pretraining loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        losses.append(float(output.loss.detach()))

    metrics = {
        "loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
    }
    checkpoint = save_checkpoint(
        Path(output_dir) / "pretrain.pt",
        model,
        stage="pretrain",
        step=options.steps,
        metrics=metrics,
    )
    return {**metrics, "checkpoint": str(checkpoint), "device": str(device)}


def run_pretrain_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    train_path: str | Path,
    output_dir: str | Path,
    options: PretrainOptions,
    validation_path: str | Path | None = None,
) -> dict[str, float | str]:
    if options.steps < 1:
        raise ValueError("steps must be positive")
    if options.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if options.validation_every < 1 or options.validation_batches < 1:
        raise ValueError("validation_every and validation_batches must be positive")
    if options.save_every < 0 or options.keep_last_checkpoints < 1:
        raise ValueError("save_every must be non-negative and keep_last_checkpoints must be positive")
    if options.generation_every < 0 or options.generation_max_new_tokens < 1:
        raise ValueError("generation_every must be non-negative and generation_max_new_tokens must be positive")
    if options.shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size must be non-negative")
    seed_everything(options.seed)
    device = resolve_device(options.device)
    model.to(device).train()
    train_dataset = JsonlPretrainDataset(
        train_path, tokenizer, options.sequence_length, split="train",
        validation_fraction=options.validation_fraction,
        shuffle_buffer_size=options.shuffle_buffer_size,
        seed=options.seed,
    )
    collate = lambda rows: collate_lm_batch(rows, tokenizer.pad_token_id)
    loader = DataLoader(train_dataset, batch_size=options.batch_size, collate_fn=collate, num_workers=options.num_workers)
    validation_loader = None
    if validation_path or options.validation_fraction > 0:
        validation_loader = DataLoader(
            JsonlPretrainDataset(
                validation_path or train_path, tokenizer, options.sequence_length,
                split="all" if validation_path else "validation",
                validation_fraction=options.validation_fraction,
            ),
            batch_size=options.batch_size,
            collate_fn=collate,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay, betas=(0.9, 0.95))
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=options.steps,
        warmup_steps=options.warmup_steps,
        min_learning_rate=options.min_learning_rate,
    )
    output = Path(output_dir)
    metrics_path = output / "pretrain_metrics.jsonl"
    checkpoint_dir = output / "checkpoints"
    last_loss = float("nan")
    tokens_seen = 0
    completed_step = 0
    micro_batches_seen = 0
    best_val_loss = float("inf")
    saved_wandb_run_id: str | None = None
    last_metrics: dict[str, float] = {"loss": last_loss, "tokens_seen": float(tokens_seen)}
    if options.resume_from is not None:
        payload = load_training_checkpoint(options.resume_from, model, optimizer, scheduler, device)
        state = payload["training_state"]
        if not isinstance(state, dict):
            raise ValueError("checkpoint training_state must be a mapping")
        saved_signature = state.get("resume_signature")
        _validate_resume_signature(saved_signature, _resume_signature(options))
        completed_step = int(payload["step"])
        tokens_seen = int(state.get("tokens_seen", 0))
        micro_batches_seen = int(state.get("micro_batches_seen", completed_step * options.gradient_accumulation_steps))
        best_val_loss = float(state.get("best_val_loss", payload.get("best_val_loss", float("inf"))))
        if state.get("wandb_run_id") is not None:
            saved_wandb_run_id = str(state["wandb_run_id"])
        last_metrics = {name: float(value) for name, value in payload.get("metrics", {}).items()}
        last_loss = float(last_metrics.get("loss", last_metrics.get("train_loss", float("nan"))))
        _truncate_metrics_after(metrics_path, completed_step)
        print(f"resuming from step={completed_step}; skipping {micro_batches_seen} consumed micro-batches", flush=True)
    if completed_step >= options.steps:
        raise ValueError("resume checkpoint is already at or beyond the requested total steps")

    batches = iter(infinite_batches(loader))
    for _ in range(micro_batches_seen):
        next(batches)
    # Constructing/skipping DataLoader iterators can consume RNG. Restore the
    # exact post-checkpoint RNG state after positioning the data stream.
    if options.resume_from is not None:
        restore_rng_state(payload.get("rng_state"))

    if options.wandb_run_id and saved_wandb_run_id and options.wandb_run_id != saved_wandb_run_id:
        raise ValueError("--wandb-run-id does not match the run id stored in the checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    tracker = WandbTracker.start(
        enabled=options.wandb_enabled,
        project=options.wandb_project,
        entity=options.wandb_entity,
        name=options.wandb_run_name,
        run_id=options.wandb_run_id or saved_wandb_run_id,
        mode=options.wandb_mode,
        config={
            "stage": "pretrain",
            "data": str(Path(train_path)),
            "validation_data": str(Path(validation_path)) if validation_path else None,
            "output": str(output),
            "num_parameters": model.num_parameters,
            "model": asdict(model.config),
            "training": {
                name: str(value) if isinstance(value, Path) else value
                for name, value in asdict(options).items()
                if name not in {"wandb_run_id", "resume_from"}
            },
        },
        directory=output,
    )
    wandb_run_id = tracker.run_id if tracker is not None else saved_wandb_run_id

    def training_state() -> dict[str, object]:
        return {
            "tokens_seen": tokens_seen,
            "micro_batches_seen": micro_batches_seen,
            "best_val_loss": best_val_loss,
            "wandb_run_id": wandb_run_id,
            "resume_signature": _resume_signature(options),
        }

    try:
        for step in range(completed_step + 1, options.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[float] = []
            step_tokens = 0
            current_lr = float(optimizer.param_groups[0]["lr"])
            for _ in range(options.gradient_accumulation_steps):
                batch = {name: value.to(device) for name, value in next(batches).items()}
                result = model(**batch)
                if result.loss is None:
                    raise RuntimeError("model did not return a pretraining loss")
                (result.loss / options.gradient_accumulation_steps).backward()
                micro_losses.append(float(result.loss.detach()))
                step_tokens += int(batch["attention_mask"].sum())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
            optimizer.step()
            scheduler.step()
            completed_step = step
            micro_batches_seen += options.gradient_accumulation_steps
            tokens_seen += step_tokens
            last_loss = sum(micro_losses) / len(micro_losses)
            metric: dict[str, object] = {
                "stage": "pretrain", "step": step, "tokens_seen": tokens_seen,
                "train_loss": last_loss, "learning_rate": current_lr,
                "grad_norm": float(grad_norm),
            }
            validation_due = validation_loader is not None and (
                step % options.validation_every == 0 or step == options.steps
            )
            if validation_due:
                val_loss = evaluate_lm(model, validation_loader, device, options.validation_batches)
                metric["validation_loss"] = val_loss
                metric["perplexity"] = math.exp(val_loss) if val_loss < 709.0 else float("inf")
            last_metrics = {
                "loss": last_loss,
                "tokens_seen": float(tokens_seen),
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss,
            }
            if validation_due:
                last_metrics["validation_loss"] = float(metric["validation_loss"])
                last_metrics["perplexity"] = float(metric["perplexity"])
                if math.isfinite(last_metrics["validation_loss"]) and last_metrics["validation_loss"] < best_val_loss:
                    best_val_loss = last_metrics["validation_loss"]
                    last_metrics["best_val_loss"] = best_val_loss
                    best_checkpoint = save_training_checkpoint(
                        output / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        stage="pretrain",
                        step=step,
                        metrics=last_metrics,
                        training_state=training_state(),
                    )
                    print(
                        f"saved best checkpoint: {best_checkpoint} (validation_loss={best_val_loss:.6f})",
                        flush=True,
                    )
                metric["best_val_loss"] = best_val_loss
            generation_path: Path | None = None
            if options.generation_every and step % options.generation_every == 0:
                generation_path = run_generation_evaluation(
                    model,
                    tokenizer,
                    output,
                    step=step,
                    device=device,
                    max_new_tokens=options.generation_max_new_tokens,
                )
                print(f"saved generation evaluation: {generation_path}", flush=True)
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
                    stage="pretrain",
                    step=step,
                    metrics=last_metrics,
                    training_state=training_state(),
                )
                _prune_periodic_checkpoints(checkpoint_dir, options.keep_last_checkpoints)
                print(f"saved checkpoint: {checkpoint}", flush=True)
    except KeyboardInterrupt:
        emergency = save_training_checkpoint(
            checkpoint_dir / f"emergency_step_{completed_step:08d}.pt",
            model,
            optimizer,
            scheduler,
            stage="pretrain",
            step=completed_step,
            metrics=last_metrics,
            training_state=training_state(),
        )
        print(f"training interrupted; emergency checkpoint saved: {emergency}", flush=True)
        if tracker is not None:
            tracker.finish(exit_code=1, summary={"interrupted_step": completed_step})
        raise

    metrics = {
        "loss": last_loss,
        "tokens_seen": float(tokens_seen),
        "learning_rate": last_metrics["learning_rate"],
        "best_val_loss": best_val_loss,
    }
    checkpoint = save_training_checkpoint(
        output / "final.pt",
        model,
        optimizer,
        scheduler,
        stage="pretrain",
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
        "device": str(device),
    }
    if wandb_run_id is not None:
        result["wandb_run_id"] = wandb_run_id
    return result
