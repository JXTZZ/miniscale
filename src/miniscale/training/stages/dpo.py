from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from miniscale.integrity import atomic_write_json, path_identity
from miniscale.model import MiniScaleForCausalLM
from miniscale.data.preference import (
    IndexedPreferenceDataset,
    PreferenceCorpusIndex,
    collate_preference_batch,
    fixed_preference_validation_batches,
)
from miniscale.tokenizer import Tokenizer
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
    autocast_context,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    infinite_batches,
    resolve_autocast_dtype,
    resolve_device,
    seed_everything,
    seed_worker,
)
from ..configs.dpo import (
    DPO_IMPLEMENTATION_VERSION,
    DPOOptions,
    dpo_option_default,
    dpo_resume_signature,
    resolved_dpo_options,
    validate_dpo_options,
)
from ..evaluators.dpo import evaluate_dpo, move_preference_batch, run_dpo_generation_evaluation
from ..objectives.dpo import (
    completion_log_probability,
    concatenated_completion_log_probabilities,
    dpo_batch_metrics,
    dpo_loss,
)


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
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != "sft":
        raise ValueError("--checkpoint must be an SFT checkpoint")
    return path_identity(path), _checkpoint_target_mode(payload)


def _save_dpo_checkpoint(
    path: str | Path,
    policy: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    metrics: dict[str, float],
    training_state: dict[str, object],
) -> Path:
    return save_training_checkpoint(
        path,
        policy,
        optimizer,
        scheduler,
        stage="dpo",
        step=step,
        metrics=metrics,
        training_state=training_state,
    )


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

    output = Path(output_dir)
    metrics_path = output / "dpo_metrics.jsonl"
    checkpoint_dir = output / "checkpoints"
    manifest_path = output / "dpo_run.json"
    if options.resume_from is None:
        existing = [
            path
            for path in (
                manifest_path,
                metrics_path,
                output / "best.pt",
                output / "dpo.pt",
                output / "reference.pt",
            )
            if path.exists()
        ]
        existing.extend(checkpoint_dir.glob("*.pt") if checkpoint_dir.exists() else ())
        if existing:
            raise FileExistsError(
                f"output already contains DPO artifacts ({existing[0]}); choose a new --output or use --resume"
            )

    seed_everything(options.seed)
    device = resolve_device(options.device)
    autocast_dtype = resolve_autocast_dtype(options.precision, device)
    resolved_precision = "bf16" if autocast_dtype is torch.bfloat16 else "fp32"

    payload: dict[str, object] | None = None
    saved_signature: dict[str, object] | None = None
    saved_reference_identity: dict[str, object] | None = None
    parent_target_mode: str | None = None
    initial_checkpoint_source: str | None = None
    if options.resume_from is not None:
        payload = read_training_checkpoint(options.resume_from, device)
        if payload.get("stage") != "dpo":
            raise ValueError("--resume requires a full DPO checkpoint")
        state = payload.get("training_state")
        if not isinstance(state, dict) or not isinstance(state.get("resume_signature"), dict):
            raise ValueError("DPO checkpoint does not contain a strict resume signature")
        saved_signature = state["resume_signature"]
        if isinstance(state.get("reference_identity"), dict):
            saved_reference_identity = state["reference_identity"]
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

    if validation_path is None:
        train_index = PreferenceCorpusIndex.build(
            data_path,
            validation_fraction=options.validation_fraction,
            target_mode=target_mode,
            destination="split",
            deduplicate_exact=options.deduplicate_exact,
        )
        validation_index = None
    else:
        train_index = PreferenceCorpusIndex.build(
            data_path,
            validation_fraction=0,
            target_mode=target_mode,
            destination="train",
            deduplicate_exact=options.deduplicate_exact,
        )
        validation_index = PreferenceCorpusIndex.build(
            validation_path,
            validation_fraction=0,
            target_mode=target_mode,
            destination="validation",
            deduplicate_exact=options.deduplicate_exact,
        )
        if validation_index.identity == train_index.identity:
            raise ValueError("dedicated DPO validation data is identical to training data")

    train_dataset = IndexedPreferenceDataset(
        train_index,
        tokenizer,
        split="train",
        max_length=max_length,
        min_context_tokens=options.min_context_tokens,
        target_mode=target_mode,
    )
    validation_dataset = IndexedPreferenceDataset(
        validation_index or train_index,
        tokenizer,
        split="validation",
        max_length=max_length,
        min_context_tokens=options.min_context_tokens,
        target_mode=target_mode,
    )
    if len(train_dataset) < options.batch_size:
        raise ValueError("DPO training split has fewer pairs than batch_size")
    if (validation_path is not None or options.validation_fraction > 0) and not len(validation_dataset):
        raise ValueError("DPO validation split contains no pairs")

    collate = partial(collate_preference_batch, pad_token_id=tokenizer.pad_token_id)
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
        validation_batches = fixed_preference_validation_batches(
            validation_dataset,
            batch_size=options.batch_size,
            batches=options.validation_batches,
            pad_token_id=tokenizer.pad_token_id,
            seed=options.seed + 1,
        )

    model.to(device).train()
    reference = deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
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
    if saved_signature is not None:
        differences = signature_differences(saved_signature, signature)
        if differences:
            raise ValueError(f"DPO resume options do not match checkpoint: {differences}")

    completed_step = micro_batches_seen = pairs_seen = input_tokens_seen = target_tokens_seen = 0
    best_val_loss = float("inf")
    last_loss = float("nan")
    saved_wandb_run_id: str | None = None
    last_metrics: dict[str, float] = {"loss": last_loss}
    reference_path = output / "reference.pt"
    if payload is not None:
        if saved_reference_identity is None:
            raise ValueError("DPO checkpoint is missing its frozen reference identity")
        current_reference_identity = path_identity(reference_path)
        if current_reference_identity != saved_reference_identity:
            raise ValueError("DPO frozen reference snapshot does not match the resume checkpoint")
        reference_payload = torch.load(reference_path, map_location=device, weights_only=False)
        if reference_payload.get("stage") != "dpo_reference":
            raise ValueError("DPO reference snapshot has an invalid stage")
        restore_training_checkpoint(payload, model, optimizer, scheduler, restore_rng=False)
        reference.load_state_dict(reference_payload["model"])
        state = payload["training_state"]
        assert isinstance(state, dict)
        completed_step = int(payload["step"])
        micro_batches_seen = int(
            state.get("micro_batches_seen", completed_step * options.gradient_accumulation_steps)
        )
        pairs_seen = int(state.get("pairs_seen", micro_batches_seen * options.batch_size))
        input_tokens_seen = int(state.get("tokens_seen", 0))
        target_tokens_seen = int(state.get("target_tokens_seen", 0))
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        if state.get("wandb_run_id") is not None:
            saved_wandb_run_id = str(state["wandb_run_id"])
        last_metrics = {name: float(value) for name, value in payload.get("metrics", {}).items()}
        last_loss = float(last_metrics.get("loss", last_metrics.get("train_loss", float("nan"))))
        truncate_metrics_after(metrics_path, completed_step)
        print(
            f"resuming DPO from step={completed_step}; replaying {micro_batches_seen} micro-batches",
            flush=True,
        )
        reference_identity = current_reference_identity
    else:
        output.mkdir(parents=True, exist_ok=True)
        save_checkpoint(reference_path, reference, stage="dpo_reference", step=0, metrics={})
        reference_identity = path_identity(reference_path)
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
        "stage": "dpo",
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": DPO_IMPLEMENTATION_VERSION,
        "model": asdict(model.config),
        "num_parameters": model.num_parameters,
        "training": resolved_dpo_options(options, target_mode=target_mode),
        "resolved": {"precision": resolved_precision, "max_length": max_length, "world_size": 1},
        "data": {
            "train_path": str(Path(data_path).resolve()),
            "validation_path": str(Path(validation_path).resolve()) if validation_path else None,
            "train_index": asdict(train_index.stats),
            "validation_index": asdict(validation_index.stats) if validation_index else None,
        },
        "initialization": {"checkpoint": initial_checkpoint_source, "identity": initial_checkpoint},
        "reference": {"checkpoint": "reference.pt", "identity": reference_identity},
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
            "pairs_seen": pairs_seen,
            "micro_batches_seen": micro_batches_seen,
            "best_val_loss": best_val_loss,
            "wandb_run_id": wandb_run_id,
            "initial_checkpoint_source": initial_checkpoint_source,
            "reference_checkpoint": "reference.pt",
            "reference_identity": reference_identity,
            "resolved_options": resolved_dpo_options(options, target_mode=target_mode),
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
            pair_counts = [int(batch["chosen"]["input_ids"].shape[0]) for batch in cpu_batches]
            total_pairs = sum(pair_counts)
            if total_pairs < 1:
                raise RuntimeError("DPO optimizer update contains no preference pairs")
            accumulated: dict[str, float] = {}
            step_input_tokens = step_target_tokens = 0
            current_lr = float(optimizer.param_groups[0]["lr"])
            for micro_step, (cpu_batch, pair_count) in enumerate(
                zip(cpu_batches, pair_counts, strict=True), 1
            ):
                batch = move_preference_batch(cpu_batch, device)
                with torch.no_grad(), autocast_context(device, autocast_dtype):
                    reference_chosen, reference_rejected, _, _ = (
                        concatenated_completion_log_probabilities(reference, batch)
                    )
                with autocast_context(device, autocast_dtype):
                    policy_chosen, policy_rejected, chosen_counts, rejected_counts = (
                        concatenated_completion_log_probabilities(model, batch)
                    )
                loss, batch_metrics = dpo_batch_metrics(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    options.beta,
                )
                if not bool(torch.isfinite(loss)) or any(
                    not bool(torch.isfinite(value)) for value in batch_metrics.values()
                ):
                    raise FloatingPointError(
                        f"non-finite DPO metric at optimizer step {step}, micro step {micro_step}"
                    )
                (loss * (pair_count / total_pairs)).backward()
                for name, value in batch_metrics.items():
                    accumulated[name] = accumulated.get(name, 0.0) + float(value) * pair_count
                step_input_tokens += int(batch["chosen"]["attention_mask"].sum())
                step_input_tokens += int(batch["rejected"]["attention_mask"].sum())
                step_target_tokens += int(chosen_counts.sum() + rejected_counts.sum())
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), options.grad_clip, error_if_nonfinite=True
                )
            except RuntimeError as error:
                if "non-finite" not in str(error):
                    raise
                raise FloatingPointError(
                    f"non-finite DPO gradient norm at optimizer step {step}"
                ) from error
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_started
            completed_step = step
            micro_batches_seen += options.gradient_accumulation_steps
            pairs_seen += total_pairs
            input_tokens_seen += step_input_tokens
            target_tokens_seen += step_target_tokens
            averaged = {name: value / total_pairs for name, value in accumulated.items()}
            last_loss = averaged["loss"]
            metric: dict[str, object] = {
                "stage": "dpo",
                "step": step,
                "pairs_seen": pairs_seen,
                "tokens_seen": input_tokens_seen,
                "target_tokens_seen": target_tokens_seen,
                "train_loss": last_loss,
                "preference_accuracy": averaged["reward_accuracy"],
                **{name: value for name, value in averaged.items() if name != "loss"},
                "learning_rate": current_lr,
                "grad_norm": float(grad_norm),
                "grad_was_clipped": bool(float(grad_norm) > options.grad_clip),
                "update_seconds": update_seconds,
                "pairs_per_second": total_pairs / max(update_seconds, 1e-12),
                "tokens_per_second": step_input_tokens / max(update_seconds, 1e-12),
                "supervised_tokens_per_second": step_target_tokens / max(update_seconds, 1e-12),
            }
            if device.type == "cuda":
                metric["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20

            validation_due = validation_batches is not None and (
                step % options.validation_every == 0 or step == options.steps
            )
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
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    best_checkpoint = _save_dpo_checkpoint(
                        output / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        step=step,
                        metrics={
                            "loss": last_loss,
                            "validation_loss": validation_loss,
                            "validation_reward_accuracy": float(
                                validation_metrics["validation_reward_accuracy"]
                            ),
                            "best_val_loss": best_val_loss,
                        },
                        training_state=training_state(),
                    )
                    print(f"saved best DPO checkpoint: {best_checkpoint}", flush=True)
                metric["best_val_loss"] = best_val_loss

            last_metrics = {
                "loss": last_loss,
                "preference_accuracy": averaged["reward_accuracy"],
                "pairs_seen": float(pairs_seen),
                "tokens_seen": float(input_tokens_seen),
                "target_tokens_seen": float(target_tokens_seen),
                "learning_rate": current_lr,
                "best_val_loss": best_val_loss,
            }
            if validation_due:
                last_metrics.update({
                    "validation_loss": float(metric["validation_loss"]),
                    "validation_reward_accuracy": float(metric["validation_reward_accuracy"]),
                })

            generation_path: Path | None = None
            if options.generation_every and step % options.generation_every == 0:
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
                checkpoint = _save_dpo_checkpoint(
                    checkpoint_dir / f"step_{step:08d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    metrics=last_metrics,
                    training_state=training_state(),
                )
                prune_periodic_checkpoints(checkpoint_dir, options.keep_last_checkpoints)
                print(f"saved DPO checkpoint: {checkpoint}", flush=True)
    except (KeyboardInterrupt, FloatingPointError):
        emergency = _save_dpo_checkpoint(
            checkpoint_dir / f"emergency_step_{completed_step:08d}.pt",
            model,
            optimizer,
            scheduler,
            step=completed_step,
            metrics=last_metrics,
            training_state=training_state(),
        )
        print(f"DPO interrupted; emergency checkpoint saved: {emergency}", flush=True)
        if tracker is not None:
            tracker.finish(exit_code=1, summary={"interrupted_step": completed_step})
        raise

    metrics = {
        "loss": last_loss,
        "preference_accuracy": float(last_metrics["preference_accuracy"]),
        "pairs_seen": float(pairs_seen),
        "tokens_seen": float(input_tokens_seen),
        "target_tokens_seen": float(target_tokens_seen),
        "learning_rate": float(last_metrics["learning_rate"]),
        "best_val_loss": best_val_loss,
    }
    checkpoint = _save_dpo_checkpoint(
        output / "dpo.pt",
        model,
        optimizer,
        scheduler,
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
