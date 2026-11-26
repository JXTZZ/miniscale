"""Argument parsing shared by the flat training entry points."""

from __future__ import annotations

import argparse
from dataclasses import fields, MISSING
from functools import partial
import json
from pathlib import Path
from typing import TypeVar

from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer, Tokenizer
from trainer.checkpoint import load_checkpoint


def option_default(option_type: type, name: str) -> object:
    """Read a CLI default from the corresponding stage options."""
    field = next(field for field in fields(option_type) if field.name == name)
    if field.default is MISSING:
        raise ValueError(f"{option_type.__name__}.{name} has no default")
    return field.default


def training_parser(
    description: str,
    stage: str,
    data: Path,
    option_type: type,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=data)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / stage)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=option_default(option_type, "batch_size"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=option_default(option_type, "device"))
    parser.add_argument("--log-every", type=int, default=option_default(option_type, "log_every"))
    return parser


def add_optimizer_arguments(parser: argparse.ArgumentParser, option_type: type) -> None:
    default = partial(option_default, option_type)
    parser.add_argument("--learning-rate", type=float, default=default("learning_rate"))
    parser.add_argument("--min-learning-rate", type=float, default=default("min_learning_rate"))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay"))
    parser.add_argument("--adam-beta1", type=float, default=default("adam_beta1"))
    parser.add_argument("--adam-beta2", type=float, default=default("adam_beta2"))
    parser.add_argument("--adam-eps", type=float, default=default("adam_eps"))
    parser.add_argument("--grad-clip", type=float, default=default("grad_clip"))
    parser.add_argument("--warmup-steps", type=int, default=default("warmup_steps"))


def add_wandb_arguments(parser: argparse.ArgumentParser, option_type: type) -> None:
    default = partial(option_default, option_type)
    parser.add_argument(
        "--wandb", action="store_true", default=default("wandb_enabled"),
        help="log training metrics to Weights & Biases",
    )
    parser.add_argument("--wandb-project", default=default("wandb_project"))
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-run-id", help="explicit W&B id; stored in checkpoints for automatic resume")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default=default("wandb_mode"),
    )
    parser.add_argument(
        "--wandb-retry-every", type=int, default=default("wandb_retry_every_steps"),
        help="retry W&B connection and pending uploads every N training steps",
    )


_Options = TypeVar("_Options")


def training_options(option_type: type[_Options], arguments: argparse.Namespace) -> _Options:
    values = vars(arguments)
    # Preserve CLI flag names and Python option field names when they differ.
    aliases = {
        "gradient_accumulation_steps": "gradient_accumulation",
        "keep_last_checkpoints": "keep_last",
        "resume_from": "resume",
        "wandb_enabled": "wandb",
        "wandb_retry_every_steps": "wandb_retry_every",
    }
    options = {}
    for field in fields(option_type):
        argument = aliases.get(field.name, field.name)
        if argument in values:
            options[field.name] = values[argument]
    if "top_k" in options:
        options["top_k"] = options["top_k"] or None
    return option_type(**options)


def load_training_model(
    arguments: argparse.Namespace, tokenizer: Tokenizer
) -> MiniScaleForCausalLM:
    model = load_checkpoint(arguments.resume or arguments.checkpoint)
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary does not match tokenizer")
    return model


def add_lm_arguments(parser: argparse.ArgumentParser, option_type: type) -> None:
    default = partial(option_default, option_type)
    add_optimizer_arguments(parser, option_type)
    add_wandb_arguments(parser, option_type)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=default("precision"),
                        help="compute precision; bf16 requires a supported CUDA device")
    parser.add_argument("--validation-data", type=Path, help="optional dedicated validation JSONL")
    parser.add_argument(
        "--generation-every", type=int, default=default("generation_every"),
        help="generate evaluation samples every N updates (0 disables)",
    )
    for flag, field in (
        ("gradient-accumulation", "gradient_accumulation_steps"),
        ("keep-last", "keep_last_checkpoints"),
        ("validation-every", "validation_every"),
        ("validation-batches", "validation_batches"),
        ("save-every", "save_every"),
        ("generation-max-new-tokens", "generation_max_new_tokens"),
        ("seed", "seed"),
        ("num-workers", "num_workers"),
    ):
        parser.add_argument(f"--{flag}", type=int, default=default(field))
    parser.add_argument("--validation-fraction", type=float, default=default("validation_fraction"))


def add_checkpoint_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="initialize from the previous stage")
    source.add_argument("--resume", type=Path, help="resume the exact state of this stage")


def add_finetune_arguments(parser: argparse.ArgumentParser, option_type: type) -> None:
    add_lm_arguments(parser, option_type)
    add_checkpoint_source(parser)
    parser.add_argument("--max-length", type=int, help="defaults to the checkpoint context length")
    parser.add_argument("--min-context-tokens", type=int, default=option_default(option_type, "min_context_tokens"))
    parser.add_argument("--deduplicate-exact", action=argparse.BooleanOptionalAction,
                        default=option_default(option_type, "deduplicate_exact"))


def run_checkpoint_stage(parser, option_type, train, argv) -> None:
    arguments = parser.parse_args(argv)
    tokenizer = load_tokenizer(arguments.tokenizer)
    result = train(
        load_training_model(arguments, tokenizer), tokenizer, arguments.data, arguments.output,
        training_options(option_type, arguments), validation_path=arguments.validation_data,
        initial_checkpoint_path=arguments.checkpoint,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def add_rl_arguments(command: argparse.ArgumentParser, option_type: type) -> None:
    default = partial(option_default, option_type)
    add_checkpoint_source(command)
    command.add_argument("--group-size", type=int, default=default("group_size"))
    command.add_argument("--max-new-tokens", type=int, default=default("max_new_tokens"))
    command.add_argument("--policy-epochs", type=int, default=default("policy_epochs"))
    add_optimizer_arguments(command, option_type)
    command.add_argument("--clip-epsilon", type=float, default=default("clip_epsilon"))
    command.add_argument("--loss-type", choices=("grpo", "cispo"), default=default("loss_type"))
    command.add_argument("--cispo-max-ratio", type=float, default=default("cispo_max_ratio"))
    command.add_argument("--kv-cache", action="store_true", dest="use_cache")
    command.add_argument("--beta", type=float, default=default("beta"))
    command.add_argument("--temperature", type=float, default=default("temperature"))
    command.add_argument(
        "--top-k", type=int, default=default("top_k"), help="use 0 to disable top-k sampling"
    )
    command.add_argument(
        "--precision", choices=("fp32", "bf16"), default=default("precision")
    )
    command.add_argument(
        "--reference-device", choices=("same", "cpu"), default=default("reference_device"),
        help="use cpu to reduce GPU memory at the cost of slower reference scoring",
    )
    command.add_argument(
        "--validation-fraction", type=float, default=default("validation_fraction")
    )
    command.add_argument("--validation-data", type=Path)
    command.add_argument("--validation-every", type=int, default=default("validation_every"))
    command.add_argument("--validation-prompts", type=int, default=default("validation_prompts"))
    command.add_argument("--save-every", type=int, default=default("save_every"))
    command.add_argument("--keep-last", type=int, default=default("keep_last_checkpoints"))
    command.add_argument("--data-limit", type=int, default=default("data_limit"))
    command.add_argument("--seed", type=int, default=default("seed"))
    add_wandb_arguments(command, option_type)
