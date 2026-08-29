from __future__ import annotations

from dataclasses import asdict
from functools import partial
import math
from pathlib import Path
import time
import warnings

import torch
from torch.utils.data import DataLoader

from miniscale.data import (
    JsonlPretrainDataset,
    PretrainDataset,
    collate_lm_batch,
    reservoir_sample_lm_batches,
)
from miniscale.integrity import atomic_write_json, path_identity, tokenizer_identity
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer, Tokenizer
from miniscale.tracking import WandbTracker
from ..core.artifacts import append_metric, prune_periodic_checkpoints, truncate_metrics_after
from ..core.logging import format_training_metric
from ..core.checkpoint import (
    TRAINING_CHECKPOINT_FORMAT_VERSION,
    read_training_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    signature_differences,
)
from ..core.runtime import (
    autocast_context as _autocast_context,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    evaluate_lm,
    infinite_batches,
    resolve_autocast_dtype,
    resolve_device,
    seed_everything,
    seed_worker,
)
from ..configs.pretrain import (
    PRETRAIN_IMPLEMENTATION_VERSION,
    PRETRAIN_INITIALIZATION_SCHEME,
    PRETRAIN_OPTIMIZER_GROUPING,
    PRETRAIN_RESUME_SIGNATURE_VERSION,
    PretrainOptions,
    SmokePretrainOptions,
    pretrain_option_default,
)


GENERATION_EVAL_PROMPTS: tuple[dict[str, str], ...] = (
    {"name": "chinese", "language": "zh", "prompt": "人工智能的发展将会"},
    {"name": "english", "language": "en", "prompt": "The future of artificial intelligence is"},
    {
        "name": "code",
        "language": "python",
        "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    },
)


def build_pretrain_optimizer(
    model: MiniScaleForCausalLM,
    options: PretrainOptions,
) -> torch.optim.AdamW:
    return build_adamw_optimizer(
        model,
        learning_rate=options.learning_rate,
        weight_decay=options.weight_decay,
        beta1=options.adam_beta1,
        beta2=options.adam_beta2,
        eps=options.adam_eps,
    )


def _migrate_legacy_single_group_optimizer(
    payload: dict[str, object],
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    """Map an old model-order AdamW state onto the current named groups."""

    saved_optimizer = payload.get("optimizer")
    if not isinstance(saved_optimizer, dict):
        raise ValueError("checkpoint optimizer state must be a mapping")
    saved_groups = saved_optimizer.get("param_groups")
    saved_state = saved_optimizer.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        raise ValueError("checkpoint optimizer state has invalid groups or state")
    if len(saved_groups) == len(optimizer.param_groups):
        return payload
    if len(saved_groups) != 1 or not isinstance(saved_groups[0], dict):
        raise ValueError(
            "legacy optimizer parameter groups cannot be migrated automatically; "
            f"checkpoint has {len(saved_groups)} groups, expected 1"
        )

    saved_ids = saved_groups[0].get("params")
    model_parameters = list(model.parameters())
    if not isinstance(saved_ids, list) or len(saved_ids) != len(model_parameters):
        raise ValueError("legacy optimizer parameter order does not match the current model")
    saved_id_by_parameter = {
        id(parameter): saved_id for parameter, saved_id in zip(model_parameters, saved_ids, strict=True)
    }

    current_optimizer = optimizer.state_dict()
    current_groups = current_optimizer["param_groups"]
    migrated_state: dict[object, object] = {}
    migrated_groups: list[dict[str, object]] = []
    source_group = saved_groups[0]
    for live_group, serialized_group in zip(optimizer.param_groups, current_groups, strict=True):
        current_ids = serialized_group["params"]
        for parameter, current_id in zip(live_group["params"], current_ids, strict=True):
            saved_id = saved_id_by_parameter[id(parameter)]
            if saved_id in saved_state:
                migrated_state[current_id] = saved_state[saved_id]
        migrated_group = dict(serialized_group)
        for name, value in source_group.items():
            if name not in {"params", "weight_decay", "group_name"}:
                migrated_group[name] = value
        migrated_groups.append(migrated_group)

    scheduler_state = payload.get("scheduler")
    migrated_scheduler = dict(scheduler_state) if isinstance(scheduler_state, dict) else scheduler_state
    if isinstance(migrated_scheduler, dict):
        group_count = len(migrated_groups)
        for name in ("base_lrs", "_last_lr", "lr_lambdas"):
            value = migrated_scheduler.get(name)
            if isinstance(value, list) and len(value) == 1:
                migrated_scheduler[name] = value * group_count

    migrated = dict(payload)
    migrated["optimizer"] = {"state": migrated_state, "param_groups": migrated_groups}
    migrated["scheduler"] = migrated_scheduler
    warnings.warn(
        "migrated legacy single-group AdamW state to decay/no-decay parameter groups",
        RuntimeWarning,
        stacklevel=2,
    )
    return migrated


def _resume_signature(
    options: PretrainOptions,
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    train_path: str | Path,
    validation_path: str | Path | None,
    resolved_precision: str,
) -> dict[str, object]:
    train_identity = path_identity(train_path)
    validation_identity: dict[str, object]
    if validation_path is None:
        validation_identity = {
            "mode": "content_hash_split",
            "fraction": options.validation_fraction,
            "source": train_identity,
        }
    else:
        dedicated_identity = path_identity(validation_path)
        if dedicated_identity == train_identity:
            raise ValueError("dedicated validation data is identical to training data")
        validation_identity = {"mode": "dedicated_file", "source": dedicated_identity}
    return {
        "signature_version": PRETRAIN_RESUME_SIGNATURE_VERSION,
        "implementation_version": PRETRAIN_IMPLEMENTATION_VERSION,
        "total_steps": options.steps,
        "batch_size": options.batch_size,
        "sequence_length": options.sequence_length,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "learning_rate": options.learning_rate,
        "min_learning_rate": options.min_learning_rate,
        "warmup_steps": options.warmup_steps,
        "weight_decay": options.weight_decay,
        "adam_beta1": options.adam_beta1,
        "adam_beta2": options.adam_beta2,
        "adam_eps": options.adam_eps,
        "grad_clip": options.grad_clip,
        "shuffle_buffer_size": options.shuffle_buffer_size,
        "num_workers": options.num_workers,
        "validation_fraction": options.validation_fraction,
        "validation_every": options.validation_every,
        "validation_batches": options.validation_batches,
        "validation_sampling": "fixed_reservoir_v1",
        "seed": options.seed,
        "precision": resolved_precision,
        "parameter_initialization": PRETRAIN_INITIALIZATION_SCHEME,
        "optimizer_parameter_groups": PRETRAIN_OPTIMIZER_GROUPING,
        "world_size": 1,
        "model": asdict(model.config),
        "tokenizer": tokenizer_identity(tokenizer),
        "train_data": train_identity,
        "validation_data": validation_identity,
    }


def _validate_resume_signature(
    saved: object,
    current: dict[str, object],
    *,
    allow_legacy: bool = False,
) -> None:
    if not isinstance(saved, dict):
        raise ValueError("checkpoint resume_signature must be a mapping")
    if saved.get("signature_version") != PRETRAIN_RESUME_SIGNATURE_VERSION:
        if not allow_legacy:
            raise ValueError(
                "checkpoint predates strict resume identity checks; pass "
                "--allow-legacy-resume once to accept the documented migration risk"
            )
        warnings.warn(
            "resuming a legacy checkpoint without verified data/tokenizer/model identity; "
            "the next checkpoint will be upgraded to the current format",
            RuntimeWarning,
            stacklevel=2,
        )
        comparable = {
            name: value
            for name, value in saved.items()
            if name in current and name not in {"signature_version", "implementation_version"}
        }
        mismatches = signature_differences(comparable, {name: current[name] for name in comparable})
    else:
        mismatches = signature_differences(saved, current)
    if mismatches:
        raise ValueError(f"resume options do not match checkpoint: {mismatches}")


def _resolved_options(options: PretrainOptions) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(options).items()
        if name not in {"allow_legacy_resume", "resume_from"}
    }


def _write_run_manifest(
    path: Path,
    *,
    model: MiniScaleForCausalLM,
    options: PretrainOptions,
    resume_signature: dict[str, object],
    train_path: str | Path,
    validation_path: str | Path | None,
    resumed_step: int,
    resolved_precision: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "stage": "pretrain",
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": PRETRAIN_IMPLEMENTATION_VERSION,
        "parameter_initialization": PRETRAIN_INITIALIZATION_SCHEME,
        "optimizer_parameter_groups": PRETRAIN_OPTIMIZER_GROUPING,
        "model": asdict(model.config),
        "num_parameters": model.num_parameters,
        "training": _resolved_options(options),
        "resolved_precision": resolved_precision,
        "derived": {
            "world_size": 1,
            "global_batch_sequences": options.batch_size * options.gradient_accumulation_steps,
            "input_tokens_per_update": (
                options.batch_size * options.gradient_accumulation_steps * options.sequence_length
            ),
            "target_tokens_per_update": (
                options.batch_size
                * options.gradient_accumulation_steps
                * (options.sequence_length - 1)
            ),
            "planned_input_tokens": (
                options.steps
                * options.batch_size
                * options.gradient_accumulation_steps
                * options.sequence_length
            ),
            "planned_target_tokens": (
                options.steps
                * options.batch_size
                * options.gradient_accumulation_steps
                * (options.sequence_length - 1)
            ),
            "warmup_ratio": min(options.warmup_steps, options.steps) / options.steps,
            "tokens_per_parameter": (
                options.steps
                * options.batch_size
                * options.gradient_accumulation_steps
                * options.sequence_length
                / model.num_parameters
            ),
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


@torch.no_grad()
def run_generation_evaluation(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    step: int,
    device: torch.device,
    max_new_tokens: int,
    autocast_dtype: torch.dtype | None = None,
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
            with _autocast_context(device, autocast_dtype):
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
    atomic_write_json(target, {
        "stage": "pretrain",
        "step": step,
        "decoding": {"do_sample": False, "strategy": "greedy", "temperature": 0.0},
        "samples": samples,
    })
    return target


def run_pretrain(
    model: MiniScaleForCausalLM,
    tokenizer: ByteTokenizer,
    texts: list[str],
    output_dir: str | Path,
    options: SmokePretrainOptions | None = None,
) -> dict[str, float | str]:
    """Run the deliberately tiny in-memory smoke path.

    Production callers should use :func:`run_pretrain_jsonl`; keeping a
    separate options type prevents smoke defaults from becoming an accidental
    real training recipe.
    """

    options = options or SmokePretrainOptions()
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
        collate_fn=partial(collate_lm_batch, pad_token_id=tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(options.seed),
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
    """Train an already-initialized model on a production JSONL stream.

    The CLI seeds before model construction. Library callers that need
    reproducible from-scratch initialization must do the same before creating
    ``model``; this function seeds data order and subsequent operations.
    """

    if options.steps < 1:
        raise ValueError("steps must be positive")
    if options.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if options.sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if options.sequence_length > model.config.max_position_embeddings:
        raise ValueError("sequence_length exceeds the model context length")
    if options.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if options.log_every < 1:
        raise ValueError("log_every must be positive")
    if options.validation_every < 1 or options.validation_batches < 1:
        raise ValueError("validation_every and validation_batches must be positive")
    if not 0 <= options.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if options.save_every < 0 or options.keep_last_checkpoints < 1:
        raise ValueError("save_every must be non-negative and keep_last_checkpoints must be positive")
    if options.generation_every < 0 or options.generation_max_new_tokens < 1:
        raise ValueError("generation_every must be non-negative and generation_max_new_tokens must be positive")
    if options.shuffle_buffer_size < 0:
        raise ValueError("shuffle_buffer_size must be non-negative")
    if options.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if options.learning_rate <= 0 or options.min_learning_rate < 0:
        raise ValueError("learning rates must be non-negative and peak learning_rate must be positive")
    if options.weight_decay < 0 or options.grad_clip <= 0 or options.adam_eps <= 0:
        raise ValueError("weight_decay must be non-negative; grad_clip and adam_eps must be positive")
    if not 0 <= options.adam_beta1 < 1 or not 0 <= options.adam_beta2 < 1:
        raise ValueError("Adam beta values must be in [0, 1)")
    if options.wandb_retry_every_steps < 1:
        raise ValueError("wandb_retry_every_steps must be positive")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary does not match tokenizer")
    special_ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    special_mismatches = {
        name: (getattr(model.config, name), value)
        for name, value in special_ids.items()
        if getattr(model.config, name) != value
    }
    if special_mismatches:
        raise ValueError(f"model special token ids do not match tokenizer: {special_mismatches}")

    output = Path(output_dir)
    metrics_path = output / "pretrain_metrics.jsonl"
    checkpoint_dir = output / "checkpoints"
    manifest_path = output / "pretrain_run.json"
    if options.resume_from is None:
        existing = [path for path in (metrics_path, output / "best.pt", output / "final.pt") if path.exists()]
        existing.extend(checkpoint_dir.glob("*.pt") if checkpoint_dir.exists() else ())
        if existing:
            raise FileExistsError(
                f"output already contains pretraining artifacts ({existing[0]}); "
                "choose a new --output or use --resume"
            )

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
    collate = partial(collate_lm_batch, pad_token_id=tokenizer.pad_token_id)
    train_generator = torch.Generator().manual_seed(options.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        collate_fn=collate,
        num_workers=options.num_workers,
        worker_init_fn=seed_worker if options.num_workers else None,
        generator=train_generator,
        drop_last=True,
    )
    validation_dataset = None
    if validation_path or options.validation_fraction > 0:
        validation_dataset = JsonlPretrainDataset(
            validation_path or train_path,
            tokenizer,
            options.sequence_length,
            split="all" if validation_path else "validation",
            validation_fraction=options.validation_fraction,
        )
    optimizer = build_pretrain_optimizer(model, options)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=options.steps,
        warmup_steps=options.warmup_steps,
        min_learning_rate=options.min_learning_rate,
    )
    last_loss = float("nan")
    tokens_seen = 0
    target_tokens_seen = 0
    completed_step = 0
    micro_batches_seen = 0
    best_val_loss = float("inf")
    saved_wandb_run_id: str | None = None
    last_metrics: dict[str, float] = {"loss": last_loss, "tokens_seen": float(tokens_seen)}
    payload: dict[str, object] | None = None
    if options.resume_from is not None:
        payload = read_training_checkpoint(options.resume_from, device)
        state = payload["training_state"]
        if not isinstance(state, dict):
            raise ValueError("checkpoint training_state must be a mapping")
        saved_signature = state.get("resume_signature")
        _validate_resume_signature(
            saved_signature,
            resume_signature,
            allow_legacy=options.allow_legacy_resume,
        )
        if not isinstance(saved_signature, dict) or (
            saved_signature.get("signature_version") != PRETRAIN_RESUME_SIGNATURE_VERSION
        ):
            payload = _migrate_legacy_single_group_optimizer(payload, model, optimizer)
        restore_training_checkpoint(payload, model, optimizer, scheduler, restore_rng=False)
        completed_step = int(payload["step"])
        tokens_seen = int(state.get("tokens_seen", 0))
        micro_batches_seen = int(state.get("micro_batches_seen", completed_step * options.gradient_accumulation_steps))
        target_tokens_seen = int(
            state.get(
                "target_tokens_seen",
                max(tokens_seen - micro_batches_seen * options.batch_size, 0),
            )
        )
        best_val_loss = float(state.get("best_val_loss", payload.get("best_val_loss", float("inf"))))
        if state.get("wandb_run_id") is not None:
            saved_wandb_run_id = str(state["wandb_run_id"])
        last_metrics = {name: float(value) for name, value in payload.get("metrics", {}).items()}
        last_loss = float(last_metrics.get("loss", last_metrics.get("train_loss", float("nan"))))
        truncate_metrics_after(metrics_path, completed_step)
        print(f"resuming from step={completed_step}; skipping {micro_batches_seen} consumed micro-batches", flush=True)
    if completed_step >= options.steps:
        raise ValueError("resume checkpoint is already at or beyond the requested total steps")

    batches = iter(infinite_batches(loader))
    for _ in range(micro_batches_seen):
        next(batches)
    # Constructing/skipping DataLoader iterators can consume RNG. Restore the
    # exact post-checkpoint RNG state after positioning the data stream.
    if options.resume_from is not None:
        assert payload is not None
        restore_rng_state(payload.get("rng_state"))

    validation_loader = None
    if validation_dataset is not None:
        validation_loader = reservoir_sample_lm_batches(
            validation_dataset,
            batch_size=options.batch_size,
            batches=options.validation_batches,
            pad_token_id=tokenizer.pad_token_id,
            seed=options.seed + 1,
        )

    if options.wandb_run_id and saved_wandb_run_id and options.wandb_run_id != saved_wandb_run_id:
        raise ValueError("--wandb-run-id does not match the run id stored in the checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(
        manifest_path,
        model=model,
        options=options,
        resume_signature=resume_signature,
        train_path=train_path,
        validation_path=validation_path,
        resumed_step=completed_step,
        resolved_precision=resolved_precision,
    )
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
            "training": _resolved_options(options),
            "resume_identity": resume_signature,
            "manifest": str(manifest_path),
        },
        directory=output,
        retry_every_steps=options.wandb_retry_every_steps,
        initial_step=completed_step,
    )
    wandb_run_id = tracker.run_id if tracker is not None else saved_wandb_run_id

    def training_state() -> dict[str, object]:
        return {
            "tokens_seen": tokens_seen,
            "target_tokens_seen": target_tokens_seen,
            "micro_batches_seen": micro_batches_seen,
            "best_val_loss": best_val_loss,
            "wandb_run_id": wandb_run_id,
            "resolved_options": _resolved_options(options),
            "resume_signature": resume_signature,
        }

    try:
        for step in range(completed_step + 1, options.steps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            update_started = time.perf_counter()
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
                if not bool(torch.isfinite(result.loss)):
                    raise FloatingPointError(
                        f"non-finite pretraining loss at optimizer step {step}, micro step {micro_step}: "
                        f"{float(result.loss.detach())}"
                    )
                target_count = int((batch["labels"][:, 1:] != -100).sum())
                if target_count < 1:
                    raise RuntimeError("pretraining micro batch contains no supervised next-token targets")
                (result.loss / options.gradient_accumulation_steps).backward()
                loss_times_targets += float(result.loss.detach()) * target_count
                step_tokens += int(batch["attention_mask"].sum())
                step_target_tokens += target_count
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), options.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                # CUDA kernels are asynchronous; synchronize before deriving
                # throughput so the metric measures completed update work.
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_started
            completed_step = step
            micro_batches_seen += options.gradient_accumulation_steps
            tokens_seen += step_tokens
            target_tokens_seen += step_target_tokens
            last_loss = loss_times_targets / step_target_tokens
            metric: dict[str, object] = {
                "stage": "pretrain", "step": step, "tokens_seen": tokens_seen,
                "target_tokens_seen": target_tokens_seen,
                "train_loss": last_loss, "learning_rate": current_lr,
                "grad_norm": float(grad_norm),
                "grad_was_clipped": bool(float(grad_norm) > options.grad_clip),
                "update_seconds": update_seconds,
                "tokens_per_second": step_tokens / max(update_seconds, 1e-12),
                "samples_per_second": (
                    options.batch_size * options.gradient_accumulation_steps / max(update_seconds, 1e-12)
                ),
            }
            if device.type == "cuda":
                metric["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            validation_due = validation_loader is not None and (
                step % options.validation_every == 0 or step == options.steps
            )
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
            last_metrics = {
                "loss": last_loss,
                "tokens_seen": float(tokens_seen),
                "target_tokens_seen": float(target_tokens_seen),
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
                    autocast_dtype=autocast_dtype,
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
                print(format_training_metric(metric), flush=True)
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
                prune_periodic_checkpoints(checkpoint_dir, options.keep_last_checkpoints)
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
        "target_tokens_seen": float(target_tokens_seen),
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
        "manifest": str(manifest_path),
        "device": str(device),
    }
    if wandb_run_id is not None:
        result["wandb_run_id"] = wandb_run_id
    return result
