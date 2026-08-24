from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

import torch
from torch import Tensor
import torch.nn.functional as F

from miniscale.model import MiniScaleForCausalLM
from miniscale.data import load_jsonl_rows
from miniscale.tokenizer import Tokenizer
from .common import append_metric, resolve_device, save_checkpoint, seed_everything


@dataclass(frozen=True, slots=True)
class RLTask:
    prompt: str
    answer: str | tuple[str, ...]


@dataclass(slots=True)
class GRPOOptions:
    steps: int = 20
    batch_size: int = 1
    group_size: int = 4
    max_new_tokens: int = 24
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


def sequence_token_log_probs(
    model: MiniScaleForCausalLM,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets = input_ids[:, 1:]
    return F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def normalize_group_rewards(rewards: Tensor, group_size: int, eps: float = 1e-4) -> Tensor:
    if rewards.numel() % group_size:
        raise ValueError("reward count must be divisible by group_size")
    groups = rewards.view(-1, group_size)
    return ((groups - groups.mean(dim=1, keepdim=True)) / (groups.std(dim=1, keepdim=True, unbiased=False) + eps)).view(-1)


def grpo_objective(
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    action_mask: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float,
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    token_advantages = advantages[:, None]
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = -torch.minimum(ratio * token_advantages, clipped_ratio * token_advantages)
    log_ratio = reference_log_probs - policy_log_probs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    denominator = action_mask.sum().clamp_min(1.0)
    loss = ((policy_loss + beta * kl) * action_mask).sum() / denominator
    return loss, {
        "policy_loss": (policy_loss * action_mask).sum() / denominator,
        "kl": (kl * action_mask).sum() / denominator,
        "clip_fraction": (((ratio - 1.0).abs() > clip_epsilon) * action_mask.bool()).float().sum() / denominator,
    }


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def _normalized_number(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value.lstrip("+")


def math_reward(completion: str, answer: str | tuple[str, ...]) -> float:
    numbers = _NUMBER.findall(completion)
    expected = (answer,) if isinstance(answer, str) else answer
    normalized_numbers = [_normalized_number(number) for number in numbers]
    matched = sum(_normalized_number(item) in normalized_numbers for item in expected)
    correctness = matched / len(expected) if expected else 0.0
    format_bonus = 0.05 if numbers else 0.0
    return correctness + format_bonus - min(len(completion), 100) * 0.001


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
        # A log-probability at index i predicts input_ids[i + 1].
        action_mask[row, max(prompt_length - 1, 0) : length - 1] = 1.0
    return input_ids, attention_mask, action_mask


@torch.no_grad()
def collect_rollouts(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[RLTask],
    options: GRPOOptions,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    rewards: list[float] = []
    prompt_budget = model.config.max_position_embeddings - options.max_new_tokens
    if prompt_budget < 2:
        raise ValueError("max_new_tokens leaves no room for a prompt")
    for task in tasks:
        prompt = tokenizer.format_messages([{"role": "user", "content": task.prompt}], generation_prompt=True)
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
    return input_ids, attention_mask, action_mask, torch.tensor(rewards, device=device)


def run_grpo(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[RLTask],
    output_dir: str | Path,
    options: GRPOOptions | None = None,
) -> dict[str, float | str]:
    options = options or GRPOOptions()
    if options.steps < 1 or options.group_size < 2 or not tasks:
        raise ValueError("GRPO requires tasks, positive steps, and group_size >= 2")
    seed_everything(options.seed)
    device = resolve_device(options.device)
    model.to(device)
    reference = deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)
    last: dict[str, float] = {}
    metrics_path = Path(output_dir) / "grpo_metrics.jsonl"
    task_cursor = 0
    for step in range(1, options.steps + 1):
        selected_tasks = [tasks[(task_cursor + index) % len(tasks)] for index in range(options.batch_size)]
        task_cursor = (task_cursor + options.batch_size) % len(tasks)
        model.eval()
        input_ids, attention_mask, action_mask, rewards = collect_rollouts(
            model, tokenizer, selected_tasks, options, device
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
            "reward_max": float(rewards.max()),
            **{name: float(value.detach()) for name, value in stats.items()},
        }
        if step == 1 or step % options.log_every == 0 or step == options.steps:
            append_metric(metrics_path, last)
            print(f"grpo step={step} loss={last['loss']:.4f} reward={last['reward_mean']:.4f}")
    checkpoint = save_checkpoint(Path(output_dir) / "rl.pt", model, stage="grpo", step=options.steps, metrics=last)
    return {**last, "checkpoint": str(checkpoint), "metrics": str(metrics_path), "device": str(device)}


def load_rl_tasks(data_path: str | Path, limit: int | None = None) -> list[RLTask]:
    tasks: list[RLTask] = []
    for row in load_jsonl_rows(data_path, limit):
        messages = row.get("conversations")
        if not isinstance(messages, list):
            continue
        users = [str(message.get("content") or "") for message in messages if message.get("role") == "user"]
        ground_truth = row.get("gt")
        answers = tuple(str(item) for item in ground_truth) if isinstance(ground_truth, list) else ()
        if users and answers:
            tasks.append(RLTask(users[-1], answers))
    return tasks


def run_grpo_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    options: GRPOOptions | None = None,
) -> dict[str, float | str]:
    options = options or GRPOOptions()
    return run_grpo(model, tokenizer, load_rl_tasks(data_path, options.data_limit), output_dir, options)
