from __future__ import annotations

from dataclasses import replace

import torch

from miniscale.data.rl import RLTask
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from .config import GRPOOptions
from .rollout import collect_rollouts

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
