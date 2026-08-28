from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import Tensor

from miniscale.integrity import atomic_write_json, path_identity, tokenizer_identity
from miniscale.model import MiniScaleForCausalLM
from miniscale.rewards import score_math_answer
from miniscale.data.rl import RLTask, build_rl_corpus, load_rl_tasks
from miniscale.tokenizer import Tokenizer
from miniscale.tracking import WandbTracker
from ..core.artifacts import (
    append_metric,
    mirror_checkpoint,
    prune_periodic_checkpoints,
    truncate_metrics_after,
)
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
    resolve_autocast_dtype,
    resolve_device,
    seed_everything,
)
from ..configs.rl import GRPOOptions, resolved_rl_options, rl_resume_options, validate_rl_options
from ..core.rl_runtime import optimize_policy_epochs
from ..objectives.grpo import grpo_objective, normalize_group_rewards, sequence_token_log_probs


GRPO_IMPLEMENTATION_VERSION = 2
GRPO_OBJECTIVE_VERSION = "sequence_balanced_clipped_grpo_v2"


def math_reward(completion: str, answer: str | tuple[str, ...]) -> float:
    """Compatibility wrapper around the structured verifier reward."""

    return score_math_answer(completion, answer).total


def _task_identity(tasks: list[RLTask] | tuple[RLTask, ...]) -> dict[str, object]:
    digest = hashlib.sha256()
    for task in tasks:
        answers = (task.answer,) if isinstance(task.answer, str) else task.answer
        digest.update(json.dumps([task.prompt, answers], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return {"kind": "in_memory_tasks", "sha256": digest.hexdigest(), "tasks": len(tasks)}


def _collate_rollouts(
    sequences: list[list[int]],
    prompt_lengths: list[int],
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    action_mask = torch.zeros((len(sequences), max_length - 1), dtype=torch.float32, device=device)
    for row, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths, strict=True)):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, device=device)
        attention_mask[row, :length] = 1
        action_mask[row, max(prompt_length - 1, 0) : length - 1] = 1
    return input_ids, attention_mask, action_mask


@torch.no_grad()
def collect_rollouts(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[RLTask],
    options: GRPOOptions,
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    rewards: list[float] = []
    prompt_budget = model.config.max_position_embeddings - options.max_new_tokens
    if prompt_budget < 2:
        raise ValueError("max_new_tokens leaves no room for a prompt")
    with autocast_context(device, autocast_dtype):
        for task in tasks:
            prompt = tokenizer.format_messages(
                [{"role": "user", "content": task.prompt}], generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt, bos=True)[-prompt_budget:]
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            for _ in range(options.group_size):
                generated = model.generate(
                    prompt_tensor,
                    max_new_tokens=options.max_new_tokens,
                    temperature=options.temperature,
                    top_k=options.top_k,
                )[0].tolist()
                completion = tokenizer.decode(generated[len(prompt_ids) :])
                sequences.append(generated)
                prompt_lengths.append(len(prompt_ids))
                rewards.append(math_reward(completion, task.answer))
    input_ids, attention_mask, action_mask = _collate_rollouts(
        sequences, prompt_lengths, tokenizer.pad_token_id, device
    )
    return input_ids, attention_mask, action_mask, torch.tensor(
        rewards, dtype=torch.float32, device=device
    )


@torch.no_grad()
def evaluate_grpo(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[RLTask],
    options: GRPOOptions,
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    eval_options = replace(options, group_size=1, temperature=0.0, top_k=None)
    rewards: list[float] = []
    exact = 0
    try:
        for task in tasks:
            _, _, _, task_rewards = collect_rollouts(
                model, tokenizer, [task], eval_options, device, autocast_dtype=autocast_dtype
            )
            value = float(task_rewards[0])
            rewards.append(value)
            exact += int(value >= 1.0)
    finally:
        model.train(was_training)
    if not rewards:
        raise ValueError("GRPO validation contains no tasks")
    return {
        "validation_reward": sum(rewards) / len(rewards),
        "validation_exact_match": exact / len(rewards),
        "validation_prompts": float(len(rewards)),
    }


def _reference_log_probs(
    reference: MiniScaleForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    reference_device: torch.device,
    policy_device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> Tensor:
    with torch.no_grad(), autocast_context(
        reference_device, autocast_dtype if reference_device.type == "cuda" else None
    ):
        values = sequence_token_log_probs(
            reference, input_ids.to(reference_device), attention_mask.to(reference_device)
        )
    return values.to(policy_device)


def _save_grpo_checkpoint(
    path: str | Path,
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    metrics: dict[str, float],
    training_state: dict[str, object],
) -> Path:
    return save_training_checkpoint(
        path, model, optimizer, scheduler, stage="grpo", step=step,
        metrics=metrics, training_state=training_state,
    )


def run_grpo(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[RLTask],
    output_dir: str | Path,
    options: GRPOOptions | None = None,
    *,
    validation_tasks: list[RLTask] | None = None,
    data_identity: dict[str, object] | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    options = options or GRPOOptions()
    validate_rl_options(options)
    if not tasks:
        raise ValueError("GRPO requires at least one training task")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary does not match tokenizer")

    output = Path(output_dir)
    metrics_path = output / "grpo_metrics.jsonl"
    manifest_path = output / "grpo_run.json"
    reference_path = output / "reference.pt"
    checkpoint_dir = output / "checkpoints"
    if options.resume_from is None:
        existing = [path for path in (
            metrics_path, manifest_path, reference_path, output / "rl.pt",
            output / "last.pt", output / "best.pt"
        ) if path.exists()]
        existing.extend(checkpoint_dir.glob("*.pt") if checkpoint_dir.exists() else ())
        if existing:
            raise FileExistsError(
                f"output already contains GRPO artifacts ({existing[0]}); choose a new --output or use --resume"
            )

    seed_everything(options.seed)
    device = resolve_device(options.device)
    autocast_dtype = resolve_autocast_dtype(options.precision, device)
    resolved_precision = "bf16" if autocast_dtype is torch.bfloat16 else "fp32"
    reference_device = device if options.reference_device == "same" else torch.device("cpu")

    payload: dict[str, object] | None = None
    saved_signature: dict[str, object] | None = None
    saved_reference_identity: dict[str, object] | None = None
    saved_wandb_run_id: str | None = None
    if options.resume_from is not None:
        payload = read_training_checkpoint(options.resume_from, device)
        if payload.get("stage") != "grpo":
            raise ValueError("--resume requires a full GRPO checkpoint")
        state = payload.get("training_state")
        if not isinstance(state, dict) or not isinstance(state.get("resume_signature"), dict):
            raise ValueError("GRPO checkpoint does not contain a strict resume signature")
        saved_signature = state["resume_signature"]
        if isinstance(state.get("reference_identity"), dict):
            saved_reference_identity = state["reference_identity"]
        if state.get("wandb_run_id") is not None:
            saved_wandb_run_id = str(state["wandb_run_id"])

    model.to(device)
    reference = deepcopy(model).to(reference_device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = build_adamw_optimizer(
        model, learning_rate=options.learning_rate, weight_decay=options.weight_decay,
        beta1=options.adam_beta1, beta2=options.adam_beta2, eps=options.adam_eps,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=options.steps * options.policy_epochs,
        warmup_steps=options.warmup_steps, min_learning_rate=options.min_learning_rate,
    )

    if initial_checkpoint_path is not None:
        initialization = path_identity(initial_checkpoint_path)
    elif saved_signature is not None:
        initialization = saved_signature.get("initial_checkpoint", {"kind": "in_memory_model"})
    else:
        initialization = {"kind": "in_memory_model"}
    task_data_identity = data_identity or {
        "train": _task_identity(tasks), "validation": _task_identity(validation_tasks or [])
    }
    signature = {
        "signature_version": 1,
        "implementation_version": GRPO_IMPLEMENTATION_VERSION,
        "objective": GRPO_OBJECTIVE_VERSION,
        "model": asdict(model.config),
        "tokenizer": tokenizer_identity(tokenizer),
        "options": rl_resume_options(options),
        "precision": resolved_precision,
        "data": task_data_identity,
        "initial_checkpoint": initialization,
    }
    if saved_signature is not None:
        differences = signature_differences(saved_signature, signature)
        if differences:
            raise ValueError(f"GRPO resume options do not match checkpoint: {differences}")

    completed_step = task_cursor = prompts_seen = rollouts_seen = action_tokens_seen = 0
    optimizer_steps = 0
    best_validation_reward = float("-inf")
    last_metrics: dict[str, float] = {"loss": float("nan")}
    if payload is not None:
        if saved_reference_identity is None or path_identity(reference_path) != saved_reference_identity:
            raise ValueError("GRPO frozen reference snapshot does not match the resume checkpoint")
        reference_payload = torch.load(reference_path, map_location=reference_device, weights_only=False)
        if reference_payload.get("stage") != "grpo_reference":
            raise ValueError("GRPO reference snapshot has an invalid stage")
        reference.load_state_dict(reference_payload["model"])
        restore_training_checkpoint(payload, model, optimizer, scheduler, restore_rng=False)
        state = payload["training_state"]
        assert isinstance(state, dict)
        completed_step = int(payload["step"])
        task_cursor = int(state.get("task_cursor", completed_step * options.batch_size)) % len(tasks)
        prompts_seen = int(state.get("prompts_seen", completed_step * options.batch_size))
        rollouts_seen = int(state.get("rollouts_seen", prompts_seen * options.group_size))
        action_tokens_seen = int(state.get("action_tokens_seen", 0))
        optimizer_steps = int(state.get("optimizer_steps", completed_step * options.policy_epochs))
        best_validation_reward = float(state.get("best_validation_reward", float("-inf")))
        last_metrics = {name: float(value) for name, value in payload.get("metrics", {}).items()}
        truncate_metrics_after(metrics_path, completed_step)
        restore_rng_state(payload.get("rng_state"))
        reference_identity = saved_reference_identity
    else:
        output.mkdir(parents=True, exist_ok=True)
        save_checkpoint(reference_path, reference, stage="grpo_reference", step=0, metrics={})
        reference_identity = path_identity(reference_path)
    if completed_step >= options.steps:
        raise ValueError("resume checkpoint is already at or beyond the requested total steps")

    if options.wandb_run_id and saved_wandb_run_id and options.wandb_run_id != saved_wandb_run_id:
        raise ValueError("--wandb-run-id does not match the run id stored in the checkpoint")
    fixed_validation = list(validation_tasks or [])
    if len(fixed_validation) > options.validation_prompts:
        fixed_validation = random.Random(options.seed + 1).sample(fixed_validation, options.validation_prompts)
    manifest = {
        "schema_version": 1,
        "stage": "grpo",
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": GRPO_IMPLEMENTATION_VERSION,
        "model": asdict(model.config),
        "num_parameters": model.num_parameters,
        "training": resolved_rl_options(options),
        "resolved": {"precision": resolved_precision, "device": str(device),
                     "reference_device": str(reference_device)},
        "data": task_data_identity,
        "initialization": initialization,
        "reference": {"checkpoint": "reference.pt", "identity": reference_identity},
        "resume_identity": signature,
    }
    atomic_write_json(manifest_path, manifest)
    tracker = WandbTracker.start(
        enabled=options.wandb_enabled, project=options.wandb_project,
        entity=options.wandb_entity, name=options.wandb_run_name,
        run_id=options.wandb_run_id or saved_wandb_run_id, mode=options.wandb_mode,
        config={**manifest, "manifest": str(manifest_path)}, directory=output,
        retry_every_steps=options.wandb_retry_every_steps, initial_step=completed_step,
    )
    wandb_run_id = tracker.run_id if tracker is not None else saved_wandb_run_id

    def training_state() -> dict[str, object]:
        return {
            "task_cursor": task_cursor, "prompts_seen": prompts_seen,
            "rollouts_seen": rollouts_seen, "action_tokens_seen": action_tokens_seen,
            "optimizer_steps": optimizer_steps, "tokens_seen": action_tokens_seen,
            "best_validation_reward": best_validation_reward, "wandb_run_id": wandb_run_id,
            "reference_identity": reference_identity,
            "resolved_options": resolved_rl_options(options), "resume_signature": signature,
        }

    try:
        for step in range(completed_step + 1, options.steps + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            selected = [tasks[(task_cursor + index) % len(tasks)] for index in range(options.batch_size)]
            task_cursor = (task_cursor + options.batch_size) % len(tasks)
            model.eval()
            input_ids, attention_mask, action_mask, rewards = collect_rollouts(
                model, tokenizer, selected, options, device, autocast_dtype=autocast_dtype
            )
            with torch.no_grad(), autocast_context(device, autocast_dtype):
                old_log_probs = sequence_token_log_probs(model, input_ids, attention_mask)
            reference_log_probs = _reference_log_probs(
                reference, input_ids, attention_mask, reference_device, device, autocast_dtype
            )
            advantages = normalize_group_rewards(rewards, options.group_size)
            zero_variance = rewards.view(-1, options.group_size).var(dim=1, unbiased=False) < 1e-12
            current_lr = float(optimizer.param_groups[0]["lr"])
            update = optimize_policy_epochs(
                model, optimizer, scheduler, input_ids=input_ids,
                attention_mask=attention_mask, action_mask=action_mask,
                old_log_probs=old_log_probs, reference_log_probs=reference_log_probs,
                advantages=advantages, policy_epochs=options.policy_epochs,
                clip_epsilon=options.clip_epsilon, beta=options.beta,
                grad_clip=options.grad_clip, device=device,
                autocast_dtype=autocast_dtype, step=step,
            )
            optimizer_steps += update.optimizer_steps
            epoch_metrics = update.metrics
            grad_norm_value = update.grad_norm
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            completed_step = step
            step_actions = int(action_mask.sum())
            prompts_seen += len(selected)
            rollouts_seen += len(selected) * options.group_size
            action_tokens_seen += step_actions
            metric: dict[str, object] = {
                "stage": "grpo", "step": step, "train_loss": epoch_metrics["loss"],
                **{name: value for name, value in epoch_metrics.items() if name != "loss"},
                "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std(unbiased=False)),
                "reward_max": float(rewards.max()),
                "zero_advantage_group_rate": float(zero_variance.float().mean()),
                "prompts_seen": prompts_seen, "rollouts_seen": rollouts_seen,
                "action_tokens_seen": action_tokens_seen, "tokens_seen": action_tokens_seen,
                "optimizer_steps": optimizer_steps, "learning_rate": current_lr,
                "grad_norm": grad_norm_value, "grad_was_clipped": grad_norm_value > options.grad_clip,
                "update_seconds": duration,
                "rollouts_per_second": len(rewards) / max(duration, 1e-12),
            }
            if device.type == "cuda":
                metric["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            validation_due = bool(fixed_validation) and (
                step % options.validation_every == 0 or step == options.steps
            )
            score_for_best = float(rewards.mean())
            if validation_due:
                validation = evaluate_grpo(
                    model, tokenizer, fixed_validation, options, device,
                    autocast_dtype=autocast_dtype,
                )
                metric.update(validation)
                score_for_best = validation["validation_reward"]
            best_candidate = validation_due or not fixed_validation
            if best_candidate and score_for_best > best_validation_reward:
                best_validation_reward = score_for_best
                metric["best_validation_reward"] = best_validation_reward
                _save_grpo_checkpoint(
                    output / "best.pt", model, optimizer, scheduler, step=step,
                    metrics={"loss": epoch_metrics["loss"], "reward_mean": float(rewards.mean()),
                             "best_validation_reward": best_validation_reward},
                    training_state=training_state(),
                )
            elif math.isfinite(best_validation_reward):
                metric["best_validation_reward"] = best_validation_reward
            last_metrics = {
                "loss": epoch_metrics["loss"], "reward_mean": float(rewards.mean()),
                "reward_max": float(rewards.max()), "best_validation_reward": best_validation_reward,
                "learning_rate": current_lr,
            }
            if validation_due:
                last_metrics.update({
                    "validation_reward": float(metric["validation_reward"]),
                    "validation_exact_match": float(metric["validation_exact_match"]),
                })
            if step == 1 or step % options.log_every == 0 or step == options.steps or validation_due:
                append_metric(metrics_path, metric)
                if tracker is not None:
                    tracker.log(metric)
                print(metric, flush=True)
            if options.save_every and step % options.save_every == 0:
                _save_grpo_checkpoint(
                    checkpoint_dir / f"step_{step:08d}.pt", model, optimizer, scheduler,
                    step=step, metrics=last_metrics, training_state=training_state(),
                )
                prune_periodic_checkpoints(checkpoint_dir, options.keep_last_checkpoints)
    except (KeyboardInterrupt, FloatingPointError):
        emergency = _save_grpo_checkpoint(
            checkpoint_dir / f"emergency_step_{completed_step:08d}.pt", model, optimizer, scheduler,
            step=completed_step, metrics=last_metrics, training_state=training_state(),
        )
        print(f"GRPO interrupted; emergency checkpoint saved: {emergency}", flush=True)
        if tracker is not None:
            tracker.finish(exit_code=1, summary={"interrupted_step": completed_step})
        raise

    checkpoint = _save_grpo_checkpoint(
        output / "rl.pt", model, optimizer, scheduler, step=options.steps,
        metrics=last_metrics, training_state=training_state(),
    )
    mirror_checkpoint(checkpoint, output / "last.pt")
    if tracker is not None:
        tracker.finish(summary={**last_metrics, "final_step": options.steps})
    result: dict[str, float | str] = {
        **last_metrics, "checkpoint": str(checkpoint), "best_checkpoint": str(output / "best.pt"),
        "metrics": str(metrics_path), "manifest": str(manifest_path), "device": str(device),
    }
    if wandb_run_id is not None:
        result["wandb_run_id"] = wandb_run_id
    return result


def run_grpo_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    options: GRPOOptions | None = None,
    *,
    validation_path: str | Path | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    options = options or GRPOOptions()
    if validation_path is None:
        corpus = build_rl_corpus(
            data_path, validation_fraction=options.validation_fraction,
            train_limit=options.data_limit, seed=options.seed,
        )
        validation = list(corpus.validation)
        identity: dict[str, object] = {"train": corpus.identity, "stats": asdict(corpus.stats)}
    else:
        corpus = build_rl_corpus(
            data_path, validation_fraction=0, train_limit=options.data_limit, seed=options.seed
        )
        validation_corpus = build_rl_corpus(validation_path, validation_fraction=0, seed=options.seed)
        if corpus.identity["source"] == validation_corpus.identity["source"]:
            raise ValueError("dedicated GRPO validation data is identical to training data")
        validation = list(validation_corpus.train)
        identity = {
            "train": corpus.identity, "train_stats": asdict(corpus.stats),
            "validation": validation_corpus.identity,
            "validation_stats": asdict(validation_corpus.stats),
        }
    return run_grpo(
        model, tokenizer, list(corpus.train), output_dir, options,
        validation_tasks=validation, data_identity=identity,
        initial_checkpoint_path=initial_checkpoint_path,
    )


__all__ = [
    "GRPOOptions", "RLTask", "collect_rollouts", "evaluate_grpo", "grpo_objective",
    "load_rl_tasks", "math_reward", "normalize_group_rewards", "run_grpo",
    "run_grpo_jsonl", "sequence_token_log_probs",
]
