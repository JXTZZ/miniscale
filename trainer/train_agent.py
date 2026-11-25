"""Train a policy to use the calculator with grouped rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from torch import Tensor
import torch

# Support running this file directly as well as with python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.rl_dataset import build_agent_corpus, CalculatorTask, DEFAULT_RL_DATA
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.cli import option_default, run_checkpoint_stage, training_parser
from trainer.policy_training import load_policy_data, run_policy
from trainer.rl_utils import add_rl_arguments, AgentRLOptions
from trainer.rollout_engine import AgentTrajectory, evaluate_agent, rollout_agent


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


def _collect_batch(model, tokenizer, tasks, options, device, autocast_dtype):
    trajectories = [
        rollout_agent(model, tokenizer, task, options, device, autocast_dtype=autocast_dtype)
        for task in tasks for _ in range(options.group_size)
    ]
    count = len(trajectories)
    metrics = {
        "success_rate": sum(item.exact for item in trajectories) / count,
        "tool_call_rate": sum(item.valid_calls > 0 for item in trajectories) / count,
        "invalid_call_rate": sum(item.invalid_calls > 0 for item in trajectories) / count,
        "mean_turns": sum(item.turns for item in trajectories) / count,
    }
    return _collate_trajectories(trajectories, tokenizer.pad_token_id, device), metrics


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
    return run_policy(
        model, tokenizer, tasks, output_dir, options or AgentRLOptions(),
        stage="agent_rl", collect_batch=_collect_batch, evaluate=evaluate_agent,
        validation_tasks=validation_tasks, data_identity=data_identity,
        initial_checkpoint_path=initial_checkpoint_path,
    )


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
    tasks, validation, identity = load_policy_data(build_agent_corpus, data_path, validation_path, options, "Agent-RL")
    return run_agent_grpo(
        model, tokenizer, tasks, output_dir, options,
        validation_tasks=validation, data_identity=identity,
        initial_checkpoint_path=initial_checkpoint_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = training_parser('Train tool-use with Agent GRPO.', 'agent-rl', DEFAULT_RL_DATA, AgentRLOptions)
    add_rl_arguments(parser, AgentRLOptions)
    parser.add_argument("--max-turns", type=int, default=option_default(AgentRLOptions, "max_turns"))
    return parser


def main(argv: list[str] | None = None) -> None:
    run_checkpoint_stage(build_parser(), AgentRLOptions, run_agent_grpo_jsonl, argv)


if __name__ == "__main__":
    main()
