from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .env import CalculatorEnv, CalculatorTask, filter_calculator_tools
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from ..core.runtime import autocast_context
from .config import AgentRLOptions

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
    generator: torch.Generator | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
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
                    generator=generator,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
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
