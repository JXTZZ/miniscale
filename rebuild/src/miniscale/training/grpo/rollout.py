from __future__ import annotations

import torch
from torch import Tensor

from miniscale.data.rl import RLTask
from miniscale.model import MiniScaleForCausalLM
from miniscale.rewards import score_math_answer
from miniscale.tokenizer import Tokenizer
from ..core.runtime import autocast_context
from .config import GRPOOptions

def math_reward(completion: str, answer: str | tuple[str, ...]) -> float:
    """Compatibility wrapper around the structured verifier reward."""

    return score_math_answer(completion, answer).total



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
