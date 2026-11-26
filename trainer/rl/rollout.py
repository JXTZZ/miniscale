"""Sample model completions and calculator trajectories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from torch import Tensor
import torch

from dataset.rl_dataset import CalculatorTask, RLTask, filter_calculator_tools
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.rl.environment import CalculatorEnv
from trainer.rl.rewards import score_math_answer
from trainer.rl.options import AgentRLOptions, GRPOOptions
from trainer.runtime.torch_utils import autocast_context


def collate_rollouts(
    sequences: list[list[int]],
    action_masks: list[list[int]],
    rewards: list[float],
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    action_mask = torch.zeros((len(sequences), max_length - 1), dtype=torch.float32, device=device)
    for row, (sequence, actions) in enumerate(zip(sequences, action_masks, strict=True)):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(sequence, device=device)
        attention_mask[row, :length] = 1
        action_mask[row, :length - 1] = torch.tensor(actions[1:], device=device)
    return input_ids, attention_mask, action_mask, torch.tensor(rewards, dtype=torch.float32, device=device)


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
    action_masks: list[list[int]] = []
    rewards: list[float] = []
    eos_token_id = model.config.eos_token_id
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
            group = prompt_tensor.expand(options.group_size, -1) if options.batch_rollouts else prompt_tensor
            for _ in range(1 if options.batch_rollouts else options.group_size):
                generated_batch = model.generate(
                    group,
                    max_new_tokens=options.max_new_tokens,
                    temperature=options.temperature,
                    top_k=options.top_k,
                    use_cache=options.use_cache,
                ).tolist()
                for generated in generated_batch:
                    completion = generated[len(prompt_ids):]
                    if eos_token_id in completion:
                        completion = completion[:completion.index(eos_token_id) + 1]
                    sequences.append(prompt_ids + completion)
                    action_masks.append([0] * len(prompt_ids) + [1] * len(completion))
                    rewards.append(score_math_answer(tokenizer.decode(completion), task.answer).total)
    return collate_rollouts(sequences, action_masks, rewards, tokenizer.pad_token_id, device)


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


class AgentSamplingOptions(Protocol):
    max_turns: int
    max_new_tokens: int
    temperature: float
    top_k: int | None
    use_cache: bool


@torch.no_grad()
def rollout_agent(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    task: CalculatorTask,
    options: AgentSamplingOptions,
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
                    use_cache=options.use_cache,
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
