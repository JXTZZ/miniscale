from __future__ import annotations

from dataclasses import replace

import torch

from miniscale.agent_env import CalculatorTask
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from .config import AgentRLOptions
from .rollout import rollout_agent

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
