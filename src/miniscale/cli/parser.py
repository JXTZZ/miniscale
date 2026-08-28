"""Argument definitions for the MiniScale command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..training.configs.dpo import dpo_option_default
from ..training.configs.pretrain import pretrain_option_default
from ..training.configs.rl import agent_rl_option_default, grpo_option_default
from ..training.configs.sft import sft_option_default


DEFAULT_DATA = Path("data/raw/minimind")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="miniscale", description="MiniScale training stack")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="show the active Python/PyTorch/CUDA environment")
    pipeline = subcommands.add_parser("pipeline", help="run the end-to-end smoke pipeline")
    pipeline.add_argument("--output", type=Path, default=Path("artifacts/run"))
    pipeline.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    generate = subcommands.add_parser("generate", help="generate text from a training checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--tokenizer", type=Path, help="tokenizer directory or SentencePiece .model")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--system-prompt")
    generate.add_argument(
        "--calculator", action="store_true",
        help="execute calculator calls in a bounded multi-turn inference loop",
    )
    generate.add_argument("--max-turns", type=int, default=6)
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-k", type=int, default=50, help="use 0 to disable top-k sampling")
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    generate.add_argument("--raw-prompt", action="store_true", help="skip the chat template (useful for base models)")
    generate.add_argument("--raw", action="store_true", help="print only the generated response")
    evaluate = subcommands.add_parser(
        "evaluate", help="compare checkpoints on a fixed verifiable or tool-use validation suite"
    )
    evaluate.add_argument("--checkpoint", type=Path, action="append", required=True)
    evaluate.add_argument("--kind", choices=("grpo", "agent"), default="grpo")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA / "agent/agent_rl_math.jsonl")
    evaluate.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/minimind"))
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--validation-fraction", type=float, default=0.05)
    evaluate.add_argument("--prompts", type=int, default=100)
    evaluate.add_argument("--max-new-tokens", type=int, default=128)
    evaluate.add_argument("--max-turns", type=int, default=6)
    evaluate.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    inspect_tokenizer = subcommands.add_parser("tokenize", help="inspect token ids and pieces for text")
    inspect_tokenizer.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/minimind"))
    inspect_tokenizer.add_argument("--text", required=True)
    inspect_tokenizer.add_argument("--add-bos", action="store_true")
    inspect_tokenizer.add_argument("--add-eos", action="store_true")
    audit_data = subcommands.add_parser(
        "audit-pretrain-data", help="fully scan pretraining JSONL and emit a reproducible data report"
    )
    audit_data.add_argument("--data", type=Path, default=DEFAULT_DATA / "pretrain/pretrain_t2t_mini.jsonl")
    audit_data.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/minimind"))
    audit_data.add_argument("--output", type=Path)
    audit_data.add_argument(
        "--validation-fraction", type=float, default=pretrain_option_default("validation_fraction")
    )
    audit_data.add_argument(
        "--sequence-length", type=int, default=pretrain_option_default("sequence_length")
    )
    audit_data.add_argument("--tokenizer-batch-size", type=int, default=4096)
    audit_sft = subcommands.add_parser(
        "audit-sft-data", help="scan SFT JSONL and report structure, duplication, splits, and truncation"
    )
    audit_sft.add_argument("--data", type=Path, default=DEFAULT_DATA / "sft/sft_t2t_mini.jsonl")
    audit_sft.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/minimind"))
    audit_sft.add_argument("--output", type=Path)
    audit_sft.add_argument("--max-length", type=int, default=512)
    audit_sft.add_argument("--min-context-tokens", type=int, default=sft_option_default("min_context_tokens"))
    audit_sft.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default=sft_option_default("target_mode"),
    )
    audit_sft.add_argument("--validation-fraction", type=float, default=sft_option_default("validation_fraction"))
    audit_sft.add_argument("--sample-size", type=int, default=5000)
    audit_sft.add_argument("--seed", type=int, default=sft_option_default("seed"))
    audit_sft.add_argument(
        "--identity-pattern",
        action="append",
        default=[],
        help="case-insensitive identity/brand substring to count; repeat for multiple patterns",
    )
    audit_dpo = subcommands.add_parser(
        "audit-dpo-data", help="scan preference JSONL and report pair validity, splits, and truncation"
    )
    audit_dpo.add_argument("--data", type=Path, default=DEFAULT_DATA / "preference/dpo.jsonl")
    audit_dpo.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/minimind"))
    audit_dpo.add_argument("--output", type=Path)
    audit_dpo.add_argument("--max-length", type=int, default=512)
    audit_dpo.add_argument(
        "--min-context-tokens", type=int, default=dpo_option_default("min_context_tokens")
    )
    audit_dpo.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default="reasoning_and_response",
    )
    audit_dpo.add_argument(
        "--validation-fraction", type=float, default=dpo_option_default("validation_fraction")
    )
    audit_dpo.add_argument("--sample-size", type=int, default=2000)
    audit_dpo.add_argument("--seed", type=int, default=dpo_option_default("seed"))
    audit_grpo = subcommands.add_parser(
        "audit-grpo-data", help="scan verifiable RL data, stable splits, duplicates, and invalid rows"
    )
    audit_grpo.add_argument("--data", type=Path, default=DEFAULT_DATA / "agent/agent_rl_math.jsonl")
    audit_grpo.add_argument("--output", type=Path)
    audit_grpo.add_argument(
        "--validation-fraction", type=float, default=grpo_option_default("validation_fraction")
    )
    audit_grpo.add_argument("--sample-size", type=int, default=2000)
    audit_grpo.add_argument("--seed", type=int, default=grpo_option_default("seed"))
    audit_agent = subcommands.add_parser(
        "audit-agent-data", help="scan Agent-RL rows and verify executable tool capabilities"
    )
    audit_agent.add_argument("--data", type=Path, default=DEFAULT_DATA / "agent/agent_rl_math.jsonl")
    audit_agent.add_argument("--output", type=Path)
    audit_agent.add_argument(
        "--validation-fraction", type=float, default=agent_rl_option_default("validation_fraction")
    )
    audit_agent.add_argument("--seed", type=int, default=agent_rl_option_default("seed"))
    prepare_sft = subcommands.add_parser(
        "prepare-sft-data", help="write a deduplicated and optionally filtered derived SFT JSONL"
    )
    prepare_sft.add_argument("--data", type=Path, default=DEFAULT_DATA / "sft/sft_t2t_mini.jsonl")
    prepare_sft.add_argument("--output", type=Path, required=True)
    prepare_sft.add_argument("--manifest", type=Path)
    prepare_sft.add_argument(
        "--deduplicate-exact", action=argparse.BooleanOptionalAction, default=True
    )
    prepare_sft.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="case-insensitive identity/brand substring to exclude; repeat for multiple patterns",
    )
    tokenizer = subcommands.add_parser("train-tokenizer", help="train SentencePiece from pretraining JSONL")
    tokenizer.add_argument("--data", type=Path, default=DEFAULT_DATA / "pretrain/pretrain_t2t_mini.jsonl")
    tokenizer.add_argument("--output-prefix", type=Path, default=Path("data/tokenizer/miniscale"))
    tokenizer.add_argument("--vocab-size", type=int, default=8192)
    tokenizer.add_argument("--input-sentences", type=int, default=1_000_000)

    def training_parser(name: str, help_text: str, default_data: Path) -> argparse.ArgumentParser:
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("--data", type=Path, default=default_data)
        command.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/minimind"))
        command.add_argument("--output", type=Path, default=Path("artifacts") / name)
        command.add_argument("--steps", type=int, required=True)
        command.add_argument("--batch-size", type=int, default=1)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--log-every", type=int, default=10)
        return command

    pretrain = training_parser("pretrain", "pretrain the 64M base model", DEFAULT_DATA / "pretrain/pretrain_t2t_mini.jsonl")
    pretrain.set_defaults(
        batch_size=pretrain_option_default("batch_size"),
        log_every=pretrain_option_default("log_every"),
    )
    pretrain.add_argument("--sequence-length", type=int, default=pretrain_option_default("sequence_length"))
    pretrain.add_argument(
        "--gradient-accumulation", type=int,
        default=pretrain_option_default("gradient_accumulation_steps"),
    )
    pretrain.add_argument("--learning-rate", type=float, default=pretrain_option_default("learning_rate"))
    pretrain.add_argument("--min-learning-rate", type=float, default=pretrain_option_default("min_learning_rate"))
    pretrain.add_argument("--weight-decay", type=float, default=pretrain_option_default("weight_decay"))
    pretrain.add_argument("--adam-beta1", type=float, default=pretrain_option_default("adam_beta1"))
    pretrain.add_argument("--adam-beta2", type=float, default=pretrain_option_default("adam_beta2"))
    pretrain.add_argument("--adam-eps", type=float, default=pretrain_option_default("adam_eps"))
    pretrain.add_argument("--grad-clip", type=float, default=pretrain_option_default("grad_clip"))
    pretrain.add_argument("--seed", type=int, default=pretrain_option_default("seed"))
    pretrain.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default=pretrain_option_default("precision"),
        help="compute precision; bf16 requires a supported CUDA device",
    )
    pretrain.add_argument("--warmup-steps", type=int, default=pretrain_option_default("warmup_steps"))
    pretrain.add_argument("--validation-every", type=int, default=pretrain_option_default("validation_every"))
    pretrain.add_argument("--validation-batches", type=int, default=pretrain_option_default("validation_batches"))
    pretrain.add_argument(
        "--validation-fraction", type=float, default=pretrain_option_default("validation_fraction")
    )
    pretrain.add_argument("--validation-data", type=Path, help="optional dedicated validation JSONL")
    pretrain.add_argument(
        "--save-every", type=int, default=pretrain_option_default("save_every"),
        help="use 0 to disable periodic checkpoints",
    )
    pretrain.add_argument(
        "--keep-last", type=int, default=pretrain_option_default("keep_last_checkpoints"),
        help="number of periodic checkpoints to retain",
    )
    pretrain.add_argument(
        "--shuffle-buffer-size", type=int, default=pretrain_option_default("shuffle_buffer_size"),
        help="use 0 to preserve JSONL order",
    )
    pretrain.add_argument(
        "--generation-every", type=int, default=pretrain_option_default("generation_every"),
        help="use 0 to disable generation evaluation",
    )
    pretrain.add_argument(
        "--generation-max-new-tokens", type=int,
        default=pretrain_option_default("generation_max_new_tokens"),
    )
    pretrain.add_argument(
        "--wandb", action="store_true", default=pretrain_option_default("wandb_enabled"),
        help="log training metrics to Weights & Biases",
    )
    pretrain.add_argument("--wandb-project", default=pretrain_option_default("wandb_project"))
    pretrain.add_argument("--wandb-entity")
    pretrain.add_argument("--wandb-run-name")
    pretrain.add_argument("--wandb-run-id", help="explicit W&B id; stored in checkpoints for automatic resume")
    pretrain.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default=pretrain_option_default("wandb_mode"),
    )
    pretrain.add_argument(
        "--wandb-retry-every", type=int, default=pretrain_option_default("wandb_retry_every_steps"),
        help="retry W&B connection and pending uploads every N training steps",
    )
    pretrain.add_argument("--resume", type=Path, help="resume from a full training checkpoint")
    pretrain.add_argument(
        "--allow-legacy-resume", action="store_true",
        help="accept a pre-v2 checkpoint without strict data/tokenizer identity checks",
    )
    pretrain.add_argument("--num-workers", type=int, default=pretrain_option_default("num_workers"))
    sft = training_parser(
        "sft", "supervised fine-tuning from a pretrain checkpoint", DEFAULT_DATA / "sft/sft_t2t_mini.jsonl"
    )
    sft.set_defaults(
        batch_size=sft_option_default("batch_size"),
        log_every=sft_option_default("log_every"),
    )
    sft_source = sft.add_mutually_exclusive_group(required=True)
    sft_source.add_argument("--checkpoint", type=Path, help="initialize a new SFT run from model weights")
    sft_source.add_argument("--resume", type=Path, help="resume the exact state of a full SFT checkpoint")
    sft.add_argument("--max-length", type=int, help="defaults to the checkpoint context length")
    sft.add_argument("--min-context-tokens", type=int, default=sft_option_default("min_context_tokens"))
    sft.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default=sft_option_default("target_mode"),
    )
    sft.add_argument(
        "--gradient-accumulation", type=int, default=sft_option_default("gradient_accumulation_steps")
    )
    sft.add_argument("--learning-rate", type=float, default=sft_option_default("learning_rate"))
    sft.add_argument("--min-learning-rate", type=float, default=sft_option_default("min_learning_rate"))
    sft.add_argument("--weight-decay", type=float, default=sft_option_default("weight_decay"))
    sft.add_argument("--adam-beta1", type=float, default=sft_option_default("adam_beta1"))
    sft.add_argument("--adam-beta2", type=float, default=sft_option_default("adam_beta2"))
    sft.add_argument("--adam-eps", type=float, default=sft_option_default("adam_eps"))
    sft.add_argument("--grad-clip", type=float, default=sft_option_default("grad_clip"))
    sft.add_argument("--warmup-steps", type=int, default=sft_option_default("warmup_steps"))
    sft.add_argument(
        "--precision", choices=("fp32", "bf16"), default=sft_option_default("precision")
    )
    sft.add_argument("--validation-fraction", type=float, default=sft_option_default("validation_fraction"))
    sft.add_argument("--validation-data", type=Path, help="optional dedicated SFT validation JSONL")
    sft.add_argument("--validation-every", type=int, default=sft_option_default("validation_every"))
    sft.add_argument("--validation-batches", type=int, default=sft_option_default("validation_batches"))
    sft.add_argument("--save-every", type=int, default=sft_option_default("save_every"))
    sft.add_argument("--keep-last", type=int, default=sft_option_default("keep_last_checkpoints"))
    sft.add_argument("--generation-every", type=int, default=sft_option_default("generation_every"))
    sft.add_argument(
        "--generation-max-new-tokens", type=int, default=sft_option_default("generation_max_new_tokens")
    )
    sft.add_argument(
        "--deduplicate-exact",
        action=argparse.BooleanOptionalAction,
        default=sft_option_default("deduplicate_exact"),
    )
    sft.add_argument("--seed", type=int, default=sft_option_default("seed"))
    sft.add_argument("--num-workers", type=int, default=sft_option_default("num_workers"))
    sft.add_argument("--wandb", action="store_true", default=sft_option_default("wandb_enabled"))
    sft.add_argument("--wandb-project", default=sft_option_default("wandb_project"))
    sft.add_argument("--wandb-entity")
    sft.add_argument("--wandb-run-name")
    sft.add_argument("--wandb-run-id")
    sft.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=sft_option_default("wandb_mode"),
    )
    sft.add_argument("--wandb-retry-every", type=int, default=sft_option_default("wandb_retry_every_steps"))
    dpo = training_parser("dpo", "preference optimization from sft.pt", DEFAULT_DATA / "preference/dpo.jsonl")
    dpo.set_defaults(
        batch_size=dpo_option_default("batch_size"),
        log_every=dpo_option_default("log_every"),
    )
    dpo_source = dpo.add_mutually_exclusive_group(required=True)
    dpo_source.add_argument("--checkpoint", type=Path, help="initialize a new DPO run from SFT weights")
    dpo_source.add_argument("--resume", type=Path, help="resume the exact state of a full DPO checkpoint")
    dpo.add_argument("--max-length", type=int, help="defaults to the checkpoint context length")
    dpo.add_argument(
        "--gradient-accumulation", type=int,
        default=dpo_option_default("gradient_accumulation_steps"),
    )
    dpo.add_argument("--learning-rate", type=float, default=dpo_option_default("learning_rate"))
    dpo.add_argument("--min-learning-rate", type=float, default=dpo_option_default("min_learning_rate"))
    dpo.add_argument("--weight-decay", type=float, default=dpo_option_default("weight_decay"))
    dpo.add_argument("--adam-beta1", type=float, default=dpo_option_default("adam_beta1"))
    dpo.add_argument("--adam-beta2", type=float, default=dpo_option_default("adam_beta2"))
    dpo.add_argument("--adam-eps", type=float, default=dpo_option_default("adam_eps"))
    dpo.add_argument("--beta", type=float, default=dpo_option_default("beta"))
    dpo.add_argument("--grad-clip", type=float, default=dpo_option_default("grad_clip"))
    dpo.add_argument("--warmup-steps", type=int, default=dpo_option_default("warmup_steps"))
    dpo.add_argument(
        "--precision", choices=("fp32", "bf16"), default=dpo_option_default("precision")
    )
    dpo.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default=None,
        help="defaults to and must match the completion semantics stored by the parent SFT run",
    )
    dpo.add_argument(
        "--min-context-tokens", type=int, default=dpo_option_default("min_context_tokens")
    )
    dpo.add_argument(
        "--validation-fraction", type=float, default=dpo_option_default("validation_fraction")
    )
    dpo.add_argument("--validation-data", type=Path, help="optional dedicated DPO validation JSONL")
    dpo.add_argument("--validation-every", type=int, default=dpo_option_default("validation_every"))
    dpo.add_argument("--validation-batches", type=int, default=dpo_option_default("validation_batches"))
    dpo.add_argument("--save-every", type=int, default=dpo_option_default("save_every"))
    dpo.add_argument("--keep-last", type=int, default=dpo_option_default("keep_last_checkpoints"))
    dpo.add_argument("--generation-every", type=int, default=dpo_option_default("generation_every"))
    dpo.add_argument(
        "--generation-max-new-tokens", type=int,
        default=dpo_option_default("generation_max_new_tokens"),
    )
    dpo.add_argument(
        "--deduplicate-exact",
        action=argparse.BooleanOptionalAction,
        default=dpo_option_default("deduplicate_exact"),
    )
    dpo.add_argument("--seed", type=int, default=dpo_option_default("seed"))
    dpo.add_argument("--num-workers", type=int, default=dpo_option_default("num_workers"))
    dpo.add_argument("--wandb", action="store_true", default=dpo_option_default("wandb_enabled"))
    dpo.add_argument("--wandb-project", default=dpo_option_default("wandb_project"))
    dpo.add_argument("--wandb-entity")
    dpo.add_argument("--wandb-run-name")
    dpo.add_argument("--wandb-run-id")
    dpo.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default=dpo_option_default("wandb_mode"),
    )
    dpo.add_argument(
        "--wandb-retry-every", type=int, default=dpo_option_default("wandb_retry_every_steps")
    )

    def add_rl_arguments(command: argparse.ArgumentParser, *, agent_stage: bool = False) -> None:
        default = agent_rl_option_default if agent_stage else grpo_option_default
        command.set_defaults(batch_size=default("batch_size"), log_every=default("log_every"))
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--checkpoint", type=Path, help="initialize from the previous stage")
        source.add_argument("--resume", type=Path, help="resume an exact full-stage checkpoint")
        command.add_argument("--group-size", type=int, default=default("group_size"))
        command.add_argument("--max-new-tokens", type=int, default=default("max_new_tokens"))
        command.add_argument("--policy-epochs", type=int, default=default("policy_epochs"))
        command.add_argument("--learning-rate", type=float, default=default("learning_rate"))
        command.add_argument("--min-learning-rate", type=float, default=default("min_learning_rate"))
        command.add_argument("--weight-decay", type=float, default=default("weight_decay"))
        command.add_argument("--adam-beta1", type=float, default=default("adam_beta1"))
        command.add_argument("--adam-beta2", type=float, default=default("adam_beta2"))
        command.add_argument("--adam-eps", type=float, default=default("adam_eps"))
        command.add_argument("--warmup-steps", type=int, default=default("warmup_steps"))
        command.add_argument("--clip-epsilon", type=float, default=default("clip_epsilon"))
        command.add_argument("--beta", type=float, default=default("beta"))
        command.add_argument("--temperature", type=float, default=default("temperature"))
        command.add_argument(
            "--top-k", type=int, default=default("top_k"), help="use 0 to disable top-k sampling"
        )
        command.add_argument("--grad-clip", type=float, default=default("grad_clip"))
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
        command.add_argument("--wandb", action="store_true", default=default("wandb_enabled"))
        command.add_argument("--wandb-project", default=default("wandb_project"))
        command.add_argument("--wandb-entity")
        command.add_argument("--wandb-run-name")
        command.add_argument("--wandb-run-id")
        command.add_argument(
            "--wandb-mode", choices=("online", "offline", "disabled"),
            default=default("wandb_mode"),
        )
        command.add_argument(
            "--wandb-retry-every", type=int, default=default("wandb_retry_every_steps")
        )

    grpo = training_parser(
        "grpo", "online GRPO with verifiable math rewards", DEFAULT_DATA / "agent/agent_rl_math.jsonl"
    )
    add_rl_arguments(grpo)
    agent = training_parser("agent-rl", "tool-use Agent GRPO", DEFAULT_DATA / "agent/agent_rl_math.jsonl")
    add_rl_arguments(agent, agent_stage=True)
    agent.add_argument("--max-turns", type=int, default=agent_rl_option_default("max_turns"))
    return parser
