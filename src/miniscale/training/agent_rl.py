from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import Tensor

from miniscale.agent_data import build_agent_corpus, load_agent_tasks
from miniscale.agent_env import CalculatorEnv, CalculatorTask, filter_calculator_tools
from miniscale.integrity import atomic_write_json, path_identity, tokenizer_identity
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from miniscale.tracking import WandbTracker
from .common import (
    TRAINING_CHECKPOINT_FORMAT_VERSION,
    append_metric,
    autocast_context,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    mirror_checkpoint,
    prune_periodic_checkpoints,
    read_training_checkpoint,
    resolve_autocast_dtype,
    resolve_device,
    restore_rng_state,
    restore_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    seed_everything,
    signature_differences,
    truncate_metrics_after,
)
from .grpo_objective import normalize_group_rewards, sequence_token_log_probs
from .rl_config import AgentRLOptions, resolved_rl_options, rl_resume_options, validate_rl_options
from .rl_runtime import optimize_policy_epochs


AGENT_RL_IMPLEMENTATION_VERSION = 2


@dataclass(slots=True)
class AgentTrajectory:
    input_ids: list[int]
    action_mask: list[int]
    reward: float
    transcript: str
    observation_tokens: int
    observation_ranges: list[tuple[int, int]]
    valid_calls: int = 0
    invalid_calls: int = 0
    exact: bool = False
    turns: int = 0
    final_answer: str = ""


ResponseFunction = Callable[[str, int], str]


@torch.no_grad()
def rollout_agent(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    task: CalculatorTask,
    options: AgentRLOptions,
    device: torch.device,
    response_fn: ResponseFunction | None = None,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> AgentTrajectory:
    env = CalculatorEnv(task)
    system_content = task.system_prompt or env.tool_prompt
    if task.system_prompt and task.tools is None:
        system_content = f"{task.system_prompt.rstrip()}\n\n{env.tool_prompt}"
    system_message: dict[str, object] = {"role": "system", "content": system_content}
    filtered_tools = filter_calculator_tools(task.tools)
    if filtered_tools is not None:
        system_message["tools"] = filtered_tools
    messages = [system_message, {"role": "user", "content": task.question}]
    transcript = tokenizer.format_messages(messages, generation_prompt=True)
    input_ids = tokenizer.encode(transcript, bos=True)
    if len(input_ids) >= model.config.max_position_embeddings:
        raise ValueError(
            "agent system/tool prompt does not fit the model context; shorten the schema or use a longer context"
        )
    action_mask = [0] * len(input_ids)
    observation_tokens = 0
    observation_ranges: list[tuple[int, int]] = []
    final_answer = ""
    turns = 0
    for turn in range(options.max_turns):
        remaining = model.config.max_position_embeddings - len(input_ids)
        if remaining <= 0:
            break
        turns = turn + 1
        if response_fn is None:
            prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
            with autocast_context(device, autocast_dtype):
                generated = model.generate(
                    prompt,
                    max_new_tokens=min(options.max_new_tokens, remaining),
                    temperature=options.temperature,
                    top_k=options.top_k,
                )[0].tolist()
            response_ids = generated[len(input_ids) :]
            response = tokenizer.decode(response_ids)
        else:
            response = response_fn(transcript, turn)
            response_ids = tokenizer.encode(response, eos=True)[:remaining]
        input_ids.extend(response_ids)
        action_mask.extend([1] * len(response_ids))
        transcript += response
        final_answer = response
        execution = env.step(response)
        if execution.observation is None or turn + 1 >= options.max_turns:
            break
        observation_text = tokenizer.format_tool_observation(
            execution.observation,
            assistant_closed=bool(response_ids) and response_ids[-1] == tokenizer.eos_token_id,
        )
        observation_ids = tokenizer.encode(observation_text)
        observation_ids = observation_ids[: model.config.max_position_embeddings - len(input_ids)]
        observation_start = len(input_ids)
        input_ids.extend(observation_ids)
        action_mask.extend([0] * len(observation_ids))
        observation_tokens += len(observation_ids)
        observation_ranges.append((observation_start, len(input_ids)))
        transcript += observation_text
    components = env.reward_components(final_answer)
    return AgentTrajectory(
        input_ids=input_ids,
        action_mask=action_mask,
        reward=components["total"],
        transcript=transcript,
        observation_tokens=observation_tokens,
        observation_ranges=observation_ranges,
        valid_calls=env.valid_calls,
        invalid_calls=env.invalid_calls,
        exact=bool(components["exact"]),
        turns=turns,
        final_answer=final_answer,
    )


def _collate_trajectories(
    trajectories: list[AgentTrajectory], pad_token_id: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    max_length = max(len(item.input_ids) for item in trajectories)
    input_ids = torch.full((len(trajectories), max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    action_mask = torch.zeros((len(trajectories), max_length - 1), dtype=torch.float32, device=device)
    for row, trajectory in enumerate(trajectories):
        length = len(trajectory.input_ids)
        input_ids[row, :length] = torch.tensor(trajectory.input_ids, device=device)
        attention_mask[row, :length] = 1
        action_mask[row, : length - 1] = torch.tensor(trajectory.action_mask[1:], device=device)
    rewards = torch.tensor([item.reward for item in trajectories], dtype=torch.float32, device=device)
    return input_ids, attention_mask, action_mask, rewards


def _task_identity(tasks: list[CalculatorTask]) -> dict[str, object]:
    digest = hashlib.sha256()
    for task in tasks:
        answers = (task.answer,) if isinstance(task.answer, str) else task.answer
        digest.update(json.dumps([task.question, answers], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return {"kind": "in_memory_agent_tasks", "sha256": digest.hexdigest(), "tasks": len(tasks)}


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


@torch.no_grad()
def evaluate_agent(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[CalculatorTask],
    options: AgentRLOptions,
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    deterministic = replace(options, temperature=0.0, top_k=None)
    try:
        trajectories = [
            rollout_agent(
                model, tokenizer, task, deterministic, device, autocast_dtype=autocast_dtype
            )
            for task in tasks
        ]
    finally:
        model.train(was_training)
    if not trajectories:
        raise ValueError("Agent-RL validation contains no tasks")
    return {
        "validation_reward": sum(item.reward for item in trajectories) / len(trajectories),
        "validation_success_rate": sum(item.exact for item in trajectories) / len(trajectories),
        "validation_tool_call_rate": sum(item.valid_calls > 0 for item in trajectories) / len(trajectories),
        "validation_invalid_call_rate": sum(item.invalid_calls > 0 for item in trajectories) / len(trajectories),
        "validation_mean_turns": sum(item.turns for item in trajectories) / len(trajectories),
        "validation_prompts": float(len(trajectories)),
    }


def _save_agent_checkpoint(
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
        path, model, optimizer, scheduler, stage="agent_rl", step=step,
        metrics=metrics, training_state=training_state,
    )


def run_agent_grpo(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[CalculatorTask],
    output_dir: str | Path,
    options: AgentRLOptions | None = None,
    *,
    validation_tasks: list[CalculatorTask] | None = None,
    data_identity: dict[str, object] | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    options = options or AgentRLOptions()
    validate_rl_options(options)
    if not tasks:
        raise ValueError("Agent GRPO requires at least one training task")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary does not match tokenizer")
    output = Path(output_dir)
    metrics_path = output / "agent_rl_metrics.jsonl"
    manifest_path = output / "agent_rl_run.json"
    reference_path = output / "reference.pt"
    checkpoint_dir = output / "checkpoints"
    if options.resume_from is None:
        existing = [path for path in (
            metrics_path, manifest_path, reference_path, output / "agent_rl.pt",
            output / "last.pt", output / "best.pt"
        ) if path.exists()]
        existing.extend(checkpoint_dir.glob("*.pt") if checkpoint_dir.exists() else ())
        if existing:
            raise FileExistsError(
                f"output already contains Agent-RL artifacts ({existing[0]}); choose a new --output or use --resume"
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
        if payload.get("stage") != "agent_rl":
            raise ValueError("--resume requires a full Agent-RL checkpoint")
        state = payload.get("training_state")
        if not isinstance(state, dict) or not isinstance(state.get("resume_signature"), dict):
            raise ValueError("Agent-RL checkpoint does not contain a strict resume signature")
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
        "implementation_version": AGENT_RL_IMPLEMENTATION_VERSION,
        "objective": "sequence_balanced_clipped_agent_grpo_v2",
        "model": asdict(model.config), "tokenizer": tokenizer_identity(tokenizer),
        "options": rl_resume_options(options), "precision": resolved_precision,
        "data": task_data_identity, "initial_checkpoint": initialization,
    }
    if saved_signature is not None:
        differences = signature_differences(saved_signature, signature)
        if differences:
            raise ValueError(f"Agent-RL resume options do not match checkpoint: {differences}")

    completed_step = task_cursor = prompts_seen = rollouts_seen = action_tokens_seen = 0
    optimizer_steps = 0
    best_validation_reward = float("-inf")
    last_metrics: dict[str, float] = {"loss": float("nan")}
    if payload is not None:
        if saved_reference_identity is None or path_identity(reference_path) != saved_reference_identity:
            raise ValueError("Agent-RL frozen reference snapshot does not match the resume checkpoint")
        reference_payload = torch.load(reference_path, map_location=reference_device, weights_only=False)
        if reference_payload.get("stage") != "agent_rl_reference":
            raise ValueError("Agent-RL reference snapshot has an invalid stage")
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
        save_checkpoint(reference_path, reference, stage="agent_rl_reference", step=0, metrics={})
        reference_identity = path_identity(reference_path)
    if completed_step >= options.steps:
        raise ValueError("resume checkpoint is already at or beyond the requested total steps")

    fixed_validation = list(validation_tasks or [])
    if len(fixed_validation) > options.validation_prompts:
        fixed_validation = random.Random(options.seed + 1).sample(fixed_validation, options.validation_prompts)
    manifest = {
        "schema_version": 1, "stage": "agent_rl",
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": AGENT_RL_IMPLEMENTATION_VERSION,
        "model": asdict(model.config), "num_parameters": model.num_parameters,
        "training": resolved_rl_options(options),
        "resolved": {"precision": resolved_precision, "device": str(device),
                     "reference_device": str(reference_device), "tool_registry": "calculator_v1"},
        "data": task_data_identity, "initialization": initialization,
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
            "best_validation_reward": best_validation_reward,
            "wandb_run_id": wandb_run_id, "reference_identity": reference_identity,
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
            trajectories = [
                rollout_agent(
                    model, tokenizer, task, options, device, autocast_dtype=autocast_dtype
                )
                for task in selected for _ in range(options.group_size)
            ]
            input_ids, attention_mask, action_mask, rewards = _collate_trajectories(
                trajectories, tokenizer.pad_token_id, device
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            completed_step = step
            prompts_seen += len(selected)
            rollouts_seen += len(trajectories)
            action_tokens_seen += int(action_mask.sum())
            metric: dict[str, object] = {
                "stage": "agent_rl", "step": step, "train_loss": update.metrics["loss"],
                **{name: value for name, value in update.metrics.items() if name != "loss"},
                "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std(unbiased=False)),
                "success_rate": sum(item.exact for item in trajectories) / len(trajectories),
                "tool_call_rate": sum(item.valid_calls > 0 for item in trajectories) / len(trajectories),
                "invalid_call_rate": sum(item.invalid_calls > 0 for item in trajectories) / len(trajectories),
                "mean_turns": sum(item.turns for item in trajectories) / len(trajectories),
                "zero_advantage_group_rate": float(zero_variance.float().mean()),
                "prompts_seen": prompts_seen, "rollouts_seen": rollouts_seen,
                "action_tokens_seen": action_tokens_seen, "tokens_seen": action_tokens_seen,
                "optimizer_steps": optimizer_steps, "learning_rate": current_lr,
                "grad_norm": update.grad_norm, "grad_was_clipped": update.grad_norm > options.grad_clip,
                "update_seconds": duration,
                "rollouts_per_second": len(trajectories) / max(duration, 1e-12),
            }
            if device.type == "cuda":
                metric["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
            validation_due = bool(fixed_validation) and (
                step % options.validation_every == 0 or step == options.steps
            )
            score_for_best = float(rewards.mean())
            if validation_due:
                validation = evaluate_agent(
                    model, tokenizer, fixed_validation, options, device,
                    autocast_dtype=autocast_dtype,
                )
                metric.update(validation)
                score_for_best = validation["validation_reward"]
            best_candidate = validation_due or not fixed_validation
            if best_candidate and score_for_best > best_validation_reward:
                best_validation_reward = score_for_best
                _save_agent_checkpoint(
                    output / "best.pt", model, optimizer, scheduler, step=step,
                    metrics={"loss": update.metrics["loss"], "reward_mean": float(rewards.mean()),
                             "best_validation_reward": best_validation_reward},
                    training_state=training_state(),
                )
            if math.isfinite(best_validation_reward):
                metric["best_validation_reward"] = best_validation_reward
            last_metrics = {
                "loss": update.metrics["loss"], "reward_mean": float(rewards.mean()),
                "success_rate": float(metric["success_rate"]),
                "tool_call_rate": float(metric["tool_call_rate"]),
                "invalid_call_rate": float(metric["invalid_call_rate"]),
                "best_validation_reward": best_validation_reward, "learning_rate": current_lr,
            }
            if validation_due:
                last_metrics.update({
                    "validation_reward": float(metric["validation_reward"]),
                    "validation_success_rate": float(metric["validation_success_rate"]),
                })
            if step == 1 or step % options.log_every == 0 or step == options.steps or validation_due:
                append_metric(metrics_path, metric)
                if tracker is not None:
                    tracker.log(metric)
                print(metric, flush=True)
            if options.save_every and step % options.save_every == 0:
                _save_agent_checkpoint(
                    checkpoint_dir / f"step_{step:08d}.pt", model, optimizer, scheduler,
                    step=step, metrics=last_metrics, training_state=training_state(),
                )
                prune_periodic_checkpoints(checkpoint_dir, options.keep_last_checkpoints)
    except (KeyboardInterrupt, FloatingPointError):
        emergency = _save_agent_checkpoint(
            checkpoint_dir / f"emergency_step_{completed_step:08d}.pt", model, optimizer, scheduler,
            step=completed_step, metrics=last_metrics, training_state=training_state(),
        )
        print(f"Agent-RL interrupted; emergency checkpoint saved: {emergency}", flush=True)
        if tracker is not None:
            tracker.finish(exit_code=1, summary={"interrupted_step": completed_step})
        raise

    checkpoint = _save_agent_checkpoint(
        output / "agent_rl.pt", model, optimizer, scheduler, step=options.steps,
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


def run_agent_grpo_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    options: AgentRLOptions | None = None,
    *,
    validation_path: str | Path | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    options = options or AgentRLOptions()
    if validation_path is None:
        corpus = build_agent_corpus(
            data_path, validation_fraction=options.validation_fraction,
            train_limit=options.data_limit, seed=options.seed,
        )
        validation = list(corpus.validation)
        identity: dict[str, object] = {"train": corpus.identity, "stats": asdict(corpus.stats)}
    else:
        corpus = build_agent_corpus(
            data_path, validation_fraction=0, train_limit=options.data_limit, seed=options.seed
        )
        validation_corpus = build_agent_corpus(validation_path, validation_fraction=0, seed=options.seed)
        if corpus.identity["source"] == validation_corpus.identity["source"]:
            raise ValueError("dedicated Agent-RL validation data is identical to training data")
        validation = list(validation_corpus.train)
        identity = {
            "train": corpus.identity, "train_stats": asdict(corpus.stats),
            "validation": validation_corpus.identity,
            "validation_stats": asdict(validation_corpus.stats),
        }
    return run_agent_grpo(
        model, tokenizer, list(corpus.train), output_dir, options,
        validation_tasks=validation, data_identity=identity,
        initial_checkpoint_path=initial_checkpoint_path,
    )


__all__ = [
    "AgentRLOptions", "AgentTrajectory", "evaluate_agent", "load_agent_tasks",
    "rollout_agent", "run_agent_grpo", "run_agent_grpo_jsonl",
]
