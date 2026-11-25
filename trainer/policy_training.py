"""Shared policy optimization loop for GRPO and calculator rollouts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import TypeVar

import torch

from dataset.file_utils import atomic_write_json, path_identity, tokenizer_identity
from dataset.rl_dataset import CalculatorTask, RLTask
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.checkpoint import (
    check_resume_signature,
    freeze_reference,
    load_reference,
    read_strict_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_checkpoint,
    TRAINING_CHECKPOINT_FORMAT_VERSION,
)
from trainer.rl_utils import (
    compute_reference_log_probs,
    GRPOOptions,
    normalize_group_rewards,
    optimize_policy_epochs,
    resolved_rl_options,
    rl_resume_options,
    sequence_token_log_probs,
    validate_rl_options,
)
from trainer.run_utils import (
    check_resume_progress,
    emergency_checkpoint,
    mirror_checkpoint,
    RunArtifacts,
    should_log,
    start_tracking,
    start_update,
    training_result,
    truncate_metrics_after,
    update_duration,
)
from trainer.trainer_utils import (
    autocast_context,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    resolve_autocast_dtype,
    resolve_device,
    seed_everything,
)


Task = TypeVar("Task", RLTask, CalculatorTask)


def task_identity(tasks: list[RLTask] | list[CalculatorTask], *, agent: bool) -> dict[str, object]:
    digest = hashlib.sha256()
    for task in tasks:
        answers = (task.answer,) if isinstance(task.answer, str) else task.answer
        prompt = task.question if isinstance(task, CalculatorTask) else task.prompt
        digest.update(json.dumps([prompt, answers], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    kind = "in_memory_agent_tasks" if agent else "in_memory_tasks"
    return {"kind": kind, "sha256": digest.hexdigest(), "tasks": len(tasks)}


def run_policy(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[Task],
    output_dir: str | Path,
    options: GRPOOptions,
    *,
    stage: str,
    collect_batch,
    evaluate,
    validation_tasks: list[Task] | None = None,
    data_identity: dict[str, object] | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    agent = stage == "agent_rl"
    label = "Agent-RL" if agent else "GRPO"
    final_name = "agent_rl.pt" if agent else "rl.pt"
    validation_metric = "validation_success_rate" if agent else "validation_exact_match"
    result_keys = ("success_rate", "tool_call_rate", "invalid_call_rate") if agent else ("reward_max",)
    validate_rl_options(options)
    if not tasks:
        raise ValueError(f"{label} requires at least one training task")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary does not match tokenizer")

    artifacts = RunArtifacts(output_dir, stage)
    output = artifacts.output
    reference_path = output / "reference.pt"
    artifacts.check_empty(
        options.resume_from, label, artifacts.metrics.name, artifacts.manifest.name,
        reference_path.name, final_name, "last.pt", "best.pt",
    )

    seed_everything(options.seed)
    device = resolve_device(options.device)
    autocast_dtype = resolve_autocast_dtype(options.precision, device)
    resolved_precision = "bf16" if autocast_dtype is torch.bfloat16 else "fp32"
    reference_device = device if options.reference_device == "same" else torch.device("cpu")

    payload, state = read_strict_checkpoint(options.resume_from, device, stage, label)
    saved_signature = state.get("resume_signature")
    saved_reference_identity = state.get("reference_identity")
    saved_run_id = state.get("wandb_run_id")
    saved_wandb_run_id = str(saved_run_id) if saved_run_id is not None else None

    model.to(device)
    reference = freeze_reference(model, reference_device)
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
        "train": task_identity(tasks, agent=agent), "validation": task_identity(validation_tasks or [], agent=agent)
    }
    signature = {
        "signature_version": 1,
        "implementation_version": 2,
        "objective": "sequence_balanced_clipped_agent_grpo_v2" if agent else "sequence_balanced_clipped_grpo_v2",
        "model": asdict(model.config),
        "tokenizer": tokenizer_identity(tokenizer),
        "options": rl_resume_options(options),
        "precision": resolved_precision,
        "data": task_data_identity,
        "initial_checkpoint": initialization,
    }
    check_resume_signature(saved_signature, signature, label)

    completed_step = task_cursor = prompts_seen = rollouts_seen = action_tokens_seen = 0
    optimizer_steps = 0
    best_validation_reward = float("-inf")
    last_metrics: dict[str, float] = {"loss": float("nan")}
    if payload is not None:
        load_reference(reference_path, stage, saved_reference_identity, reference_device, reference, label)
        restore_training_checkpoint(payload, model, optimizer, scheduler, restore_rng=False)
        completed_step = int(payload["step"])
        task_cursor = int(state.get("task_cursor", completed_step * options.batch_size)) % len(tasks)
        prompts_seen = int(state.get("prompts_seen", completed_step * options.batch_size))
        rollouts_seen = int(state.get("rollouts_seen", prompts_seen * options.group_size))
        action_tokens_seen = int(state.get("action_tokens_seen", 0))
        optimizer_steps = int(state.get("optimizer_steps", completed_step * options.policy_epochs))
        best_validation_reward = float(state.get("best_validation_reward", float("-inf")))
        last_metrics = {name: float(value) for name, value in payload.get("metrics", {}).items()}
        check_resume_progress(completed_step, options.steps, None if agent else options.wandb_run_id, saved_wandb_run_id)
        truncate_metrics_after(artifacts.metrics, completed_step)
        restore_rng_state(payload.get("rng_state"))
        reference_identity = saved_reference_identity
    else:
        save_checkpoint(reference_path, reference, stage=f"{stage}_reference", step=0, metrics={})
        reference_identity = path_identity(reference_path)
    fixed_validation = list(validation_tasks or [])
    if len(fixed_validation) > options.validation_prompts:
        fixed_validation = random.Random(options.seed + 1).sample(fixed_validation, options.validation_prompts)
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": 2,
        "model": asdict(model.config),
        "num_parameters": model.num_parameters,
        "training": resolved_rl_options(options),
        "resolved": {"precision": resolved_precision, "device": str(device),
                     "reference_device": str(reference_device),
                     **({"tool_registry": "calculator_v1"} if agent else {})},
        "data": task_data_identity,
        "initialization": initialization,
        "reference": {"checkpoint": "reference.pt", "identity": reference_identity},
        "resume_identity": signature,
    }
    atomic_write_json(artifacts.manifest, manifest)
    tracker, wandb_run_id = start_tracking(
        options, output, completed_step, saved_wandb_run_id,
        {**manifest, "manifest": str(artifacts.manifest)},
    )

    def training_state() -> dict[str, object]:
        return {
            "task_cursor": task_cursor, "prompts_seen": prompts_seen,
            "rollouts_seen": rollouts_seen, "action_tokens_seen": action_tokens_seen,
            "optimizer_steps": optimizer_steps, "tokens_seen": action_tokens_seen,
            "best_validation_reward": best_validation_reward, "wandb_run_id": wandb_run_id,
            "reference_identity": reference_identity,
            "resolved_options": resolved_rl_options(options), "resume_signature": signature,
        }

    save = artifacts.checkpoint_writer(model, optimizer, scheduler, training_state)
    try:
        for step in range(completed_step + 1, options.steps + 1):
            validation_due = bool(fixed_validation) and (
                step % options.validation_every == 0 or step == options.steps
            )
            log_due = should_log(step, options, validation_due)
            started = start_update(device) if log_due else None
            selected = [tasks[(task_cursor + index) % len(tasks)] for index in range(options.batch_size)]
            task_cursor = (task_cursor + options.batch_size) % len(tasks)
            model.eval()
            (input_ids, attention_mask, action_mask, rewards), rollout_metrics = collect_batch(
                model, tokenizer, selected, options, device, autocast_dtype
            )
            with torch.no_grad(), autocast_context(device, autocast_dtype):
                old_log_probs = sequence_token_log_probs(model, input_ids, attention_mask)
            reference_log_probs = compute_reference_log_probs(
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
            duration = update_duration(started, device) if started is not None else None
            completed_step = step
            step_actions = int(action_mask.sum())
            prompts_seen += len(selected)
            rollouts_seen += len(rewards)
            action_tokens_seen += step_actions
            metric: dict[str, object] = {
                "stage": stage, "step": step, "train_loss": epoch_metrics["loss"],
                **{name: value for name, value in epoch_metrics.items() if name != "loss"},
                "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std(unbiased=False)),
                **rollout_metrics,
                "zero_advantage_group_rate": float(zero_variance.float().mean()),
                "prompts_seen": prompts_seen, "rollouts_seen": rollouts_seen,
                "action_tokens_seen": action_tokens_seen, "tokens_seen": action_tokens_seen,
                "optimizer_steps": optimizer_steps, "learning_rate": current_lr,
                "grad_norm": grad_norm_value, "grad_was_clipped": grad_norm_value > options.grad_clip,
            }
            if duration is not None:
                metric.update(update_seconds=duration, rollouts_per_second=len(rewards) / max(duration, 1e-12))
                if device.type == "cuda":
                    metric["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            score_for_best = float(rewards.mean())
            if validation_due:
                validation = evaluate(
                    model, tokenizer, fixed_validation, options, device,
                    autocast_dtype=autocast_dtype,
                )
                metric.update(validation)
                score_for_best = validation["validation_reward"]
            best_candidate = validation_due or not fixed_validation
            if best_candidate and score_for_best > best_validation_reward:
                best_validation_reward = score_for_best
                metric["best_validation_reward"] = best_validation_reward
                save(output / "best.pt", step, {
                    "loss": epoch_metrics["loss"], "reward_mean": float(rewards.mean()),
                    "best_validation_reward": best_validation_reward,
                })
            elif math.isfinite(best_validation_reward):
                metric["best_validation_reward"] = best_validation_reward
            last_metrics = {
                "loss": epoch_metrics["loss"], "reward_mean": float(rewards.mean()),
                **{key: rollout_metrics[key] for key in result_keys},
                "best_validation_reward": best_validation_reward,
                "learning_rate": current_lr,
            }
            if validation_due:
                last_metrics.update({
                    "validation_reward": float(metric["validation_reward"]),
                    validation_metric: float(metric[validation_metric]),
                })
            if log_due:
                artifacts.log(metric, tracker)
            artifacts.save_periodic(save, step, last_metrics, options)
    except (KeyboardInterrupt, FloatingPointError):
        emergency_checkpoint(artifacts, save, completed_step, last_metrics, tracker, label)
        raise

    checkpoint = save(output / final_name, options.steps, last_metrics)
    mirror_checkpoint(checkpoint, output / "last.pt")
    return training_result(artifacts, checkpoint, last_metrics, device, tracker, wandb_run_id, options.steps)



def load_policy_data(build_corpus, data_path, validation_path, options, label):
    corpus = build_corpus(
        data_path, validation_fraction=options.validation_fraction if validation_path is None else 0,
        train_limit=options.data_limit, seed=options.seed,
    )
    if validation_path is None:
        return list(corpus.train), list(corpus.validation), {"train": corpus.identity, "stats": asdict(corpus.stats)}
    validation = build_corpus(validation_path, validation_fraction=0, seed=options.seed)
    if validation.identity["source"] == corpus.identity["source"]:
        raise ValueError(f"dedicated {label} validation data is identical to training data")
    identity = {
        "train": corpus.identity, "train_stats": asdict(corpus.stats),
        "validation": validation.identity, "validation_stats": asdict(validation.stats),
    }
    return list(corpus.train), list(validation.train), identity
