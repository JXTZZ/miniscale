from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from miniscale.agent_env import CalculatorEnv, CalculatorTask
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer
from .common import resolve_device, save_checkpoint, seed_everything
from .grpo import grpo_objective, normalize_group_rewards, sequence_token_log_probs


@dataclass(slots=True)
class AgentRLOptions:
    steps: int = 20
    group_size: int = 4
    max_turns: int = 2
    max_new_tokens: int = 64
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    beta: float = 0.01
    temperature: float = 1.0
    top_k: int | None = 50
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"


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
    tokenizer: ByteTokenizer,
    task: CalculatorTask,
    options: AgentRLOptions,
    device: torch.device,
    response_fn: ResponseFunction | None = None,
) -> AgentTrajectory:
    env = CalculatorEnv(task)
    messages = [
        {"role": "system", "content": env.tool_prompt},
        {"role": "user", "content": task.question},
    ]
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
        observation_text = f"\n<|tool|>\n{observation}<|end|>\n<|assistant|>\n"
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
    tokenizer: ByteTokenizer,
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
    for _ in range(options.steps):
        model.eval()
        trajectories = [
            rollout_agent(model, tokenizer, task, options, device)
            for task in tasks
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
            "loss": float(loss.detach()),
            "reward_mean": float(rewards.mean()),
            "success_rate": float((rewards >= 1.0).float().mean()),
            "tool_call_rate": sum(item.observation_tokens > 0 for item in trajectories) / len(trajectories),
            **{name: float(value.detach()) for name, value in stats.items()},
        }
    checkpoint = save_checkpoint(
        Path(output_dir) / "agent_rl.pt", model, stage="agent_grpo", step=options.steps, metrics=last
    )
    return {**last, "checkpoint": str(checkpoint), "device": str(device)}
