"""Sample model completions and calculator trajectories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor
import torch

from dataset.rl_dataset import CalculatorTask, RLTask, filter_calculator_tools
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.agent_env import CalculatorEnv
from trainer.rewards import score_math_answer
from trainer.rl_utils import AgentRLOptions, GRPOOptions
from trainer.trainer_utils import autocast_context


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
                rewards.append(score_math_answer(completion, task.answer).total)
    input_ids, attention_mask, action_mask = _collate_rollouts(
        sequences, prompt_lengths, tokenizer.pad_token_id, device
    )
    return input_ids, attention_mask, action_mask, torch.tensor(
        rewards, dtype=torch.float32, device=device
    )


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
