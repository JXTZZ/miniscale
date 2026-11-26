"""Train with grouped rollouts and verifiable math rewards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Support running this file directly as well as with python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.rl_dataset import build_rl_corpus, DEFAULT_RL_DATA, RLTask
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.cli import run_checkpoint_stage, training_parser
from trainer.rl.training import load_policy_data, run_policy
from trainer.cli import add_rl_arguments
from trainer.rl.options import GRPOOptions
from trainer.rl.rollout import collect_rollouts, evaluate_grpo


def _collect_batch(model, tokenizer, tasks, options, device, autocast_dtype):
    batch = collect_rollouts(model, tokenizer, tasks, options, device, autocast_dtype=autocast_dtype)
    return batch, {"reward_max": float(batch[-1].max())}


def run_grpo(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    tasks: list[RLTask],
    output_dir: str | Path,
    options: GRPOOptions | None = None,
    *,
    validation_tasks: list[RLTask] | None = None,
    data_identity: dict[str, object] | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    return run_policy(
        model, tokenizer, tasks, output_dir, options or GRPOOptions(),
        stage="grpo", collect_batch=_collect_batch, evaluate=evaluate_grpo,
        validation_tasks=validation_tasks, data_identity=data_identity,
        initial_checkpoint_path=initial_checkpoint_path,
    )


def run_grpo_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    options: GRPOOptions | None = None,
    *,
    validation_path: str | Path | None = None,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    options = options or GRPOOptions()
    tasks, validation, identity = load_policy_data(build_rl_corpus, data_path, validation_path, options, "GRPO")
    return run_grpo(
        model, tokenizer, tasks, output_dir, options,
        validation_tasks=validation, data_identity=identity,
        initial_checkpoint_path=initial_checkpoint_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = training_parser('Run GRPO with verifiable math rewards.', 'grpo', DEFAULT_RL_DATA, GRPOOptions)
    add_rl_arguments(parser, GRPOOptions)
    parser.add_argument("--batch-rollouts", action="store_true", help="sample one prompt's candidates together")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_checkpoint_stage(build_parser(), GRPOOptions, run_grpo_jsonl, argv)


if __name__ == "__main__":
    main()
