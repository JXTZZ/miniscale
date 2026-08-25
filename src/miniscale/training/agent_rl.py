from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import Tensor

from miniscale.agent_env import CalculatorEnv, CalculatorTask
from miniscale.data import load_jsonl_rows
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from .common import append_metric, resolve_device, save_checkpoint, seed_everything
from .grpo import grpo_objective, normalize_group_rewards, sequence_token_log_probs


@dataclass(slots=True)
class AgentRLOptions:
    steps: int = 20
    batch_size: int = 1
    group_size: int = 4
    max_turns: int = 6
    max_new_tokens: int = 64
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    beta: float = 0.01
    temperature: float = 1.0
    top_k: int | None = 50
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    data_limit: int | None = 1000
    log_every: int = 10


@dataclass(slots=True)
class AgentTrajectory:
    input_ids: list[int]
    action_mask: list[int]
    reward: float
    transcript: str
    observation_tokens: int
    observation_ranges: list[tuple[int, int]]


ResponseFunction = Callable[[str, int], str]


@torch.no_grad()
def rollout_agent(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    task: CalculatorTask,
    options: AgentRLOptions,
    device: torch.device,
    response_fn: ResponseFunction | None = None,
) -> AgentTrajectory:
    env = CalculatorEnv(task)
    system_message: dict[str, object] = {"role": "system", "content": task.system_prompt or env.tool_prompt}
    if task.tools is not None:
        system_message["tools"] = task.tools
    messages = [system_message, {"role": "user", "content": task.question}]
    transcript = tokenizer.format_messages(messages, generation_prompt=True)
    input_ids = tokenizer.encode(transcript, bos=True)
    if len(input_ids) >= model.config.max_position_embeddings:
        input_ids = input_ids[-(model.config.max_position_embeddings - 1) :]
    action_mask = [0] * len(input_ids)
    observation_tokens = 0
    observation_ranges: list[tuple[int, int]] = []
    final_answer = ""
    for turn in range(options.max_turns):
        remaining = model.config.max_position_embeddings - len(input_ids)
        if remaining <= 0:
            break
        if response_fn is None:
            prompt = torch.tensor([input_ids], dtype=torch.long, device=device)
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
        observation = env.execute(response)
        if observation is None or turn + 1 >= options.max_turns:
            break
        observation_text = tokenizer.format_tool_observation(
            observation,
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
    return AgentTrajectory(
        input_ids=input_ids,
        action_mask=action_mask,
        reward=env.reward(final_answer),
        transcript=transcript,
        observation_tokens=observation_tokens,
        observation_ranges=observation_ranges,
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
        # Shift because log-probability i predicts token i + 1.
        action_mask[row, : length - 1] = torch.tensor(trajectory.action_mask[1:], device=device)
    rewards = torch.tensor([item.reward for item in trajectories], dtype=torch.float32, device=device)
    return input_ids, attention_mask, action_mask, rewards


def run_agent_grpo(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[CalculatorTask],
    output_dir: str | Path,
    options: AgentRLOptions | None = None,
) -> dict[str, float | str]:
    options = options or AgentRLOptions()
    if options.steps < 1 or options.group_size < 2 or not tasks:
        raise ValueError("Agent GRPO requires tasks, positive steps, and group_size >= 2")
    seed_everything(options.seed)
    device = resolve_device(options.device)
    model.to(device)
    reference = deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)
    last: dict[str, float] = {}
    metrics_path = Path(output_dir) / "agent_rl_metrics.jsonl"
    task_cursor = 0
    for step in range(1, options.steps + 1):
        selected_tasks = [tasks[(task_cursor + index) % len(tasks)] for index in range(options.batch_size)]
        task_cursor = (task_cursor + options.batch_size) % len(tasks)
        model.eval()
        trajectories = [
            rollout_agent(model, tokenizer, task, options, device)
            for task in selected_tasks
            for _ in range(options.group_size)
        ]
        input_ids, attention_mask, action_mask, rewards = _collate_trajectories(
            trajectories, tokenizer.pad_token_id, device
        )
        with torch.no_grad():
            old_log_probs = sequence_token_log_probs(model, input_ids, attention_mask)
            reference_log_probs = sequence_token_log_probs(reference, input_ids, attention_mask)
        advantages = normalize_group_rewards(rewards, options.group_size)
        model.train()
        policy_log_probs = sequence_token_log_probs(model, input_ids, attention_mask)
        loss, stats = grpo_objective(
            policy_log_probs,
            old_log_probs,
            reference_log_probs,
            action_mask,
            advantages,
            clip_epsilon=options.clip_epsilon,
            beta=options.beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        last = {
            "step": float(step),
            "loss": float(loss.detach()),
            "reward_mean": float(rewards.mean()),
            "success_rate": float((rewards >= 1.0).float().mean()),
            "tool_call_rate": sum(item.observation_tokens > 0 for item in trajectories) / len(trajectories),
            **{name: float(value.detach()) for name, value in stats.items()},
        }
        if step == 1 or step % options.log_every == 0 or step == options.steps:
            append_metric(metrics_path, last)
            print(
                f"agent-rl step={step} loss={last['loss']:.4f} "
                f"reward={last['reward_mean']:.4f} success={last['success_rate']:.3f}"
            )
    checkpoint = save_checkpoint(
        Path(output_dir) / "agent_rl.pt", model, stage="agent_grpo", step=options.steps, metrics=last
    )
    return {**last, "checkpoint": str(checkpoint), "metrics": str(metrics_path), "device": str(device)}


def load_agent_tasks(data_path: str | Path, limit: int | None = None) -> list[CalculatorTask]:
    tasks: list[CalculatorTask] = []
    for row in load_jsonl_rows(data_path, limit):
        messages = row.get("conversations")
        if not isinstance(messages, list):
            continue
        users = [str(message.get("content") or "") for message in messages if message.get("role") == "user"]
        systems = [message for message in messages if message.get("role") == "system"]
        ground_truth = row.get("gt")
        answers = tuple(str(item) for item in ground_truth) if isinstance(ground_truth, list) else ()
        if not users or not answers:
            continue
        system_prompt = None
        tools = None
        if systems:
            system = systems[-1]
            system_prompt = str(system.get("content") or "").strip() or None
            tools = system.get("tools")
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except json.JSONDecodeError:
                    pass
        tasks.append(CalculatorTask(users[-1], "", answers, system_prompt, tools))
    return tasks


def run_agent_grpo_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    options: AgentRLOptions | None = None,
) -> dict[str, float | str]:
    options = options or AgentRLOptions()
    tasks = load_agent_tasks(data_path, options.data_limit)
    return run_agent_grpo(model, tokenizer, tasks, output_dir, options)
