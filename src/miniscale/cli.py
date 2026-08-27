from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from .agent_env import CalculatorEnv
from .config import MiniScaleConfig
from .data_audit import audit_pretrain_jsonl, save_data_audit
from .inference import GenerationOptions, generate_from_checkpoint
from .model import MiniScaleForCausalLM
from .pipeline import run_training_pipeline
from .sft_data_audit import audit_sft_jsonl, save_sft_data_audit
from .sft_data_prepare import prepare_sft_jsonl
from .tokenizer import load_tokenizer, train_sentencepiece
from .training import (
    AgentRLOptions, DPOOptions, GRPOOptions, PretrainOptions, SFTOptions,
    run_agent_grpo_jsonl, run_dpo_jsonl, run_grpo_jsonl, run_pretrain_jsonl, run_sft_jsonl,
)
from .training.common import load_checkpoint, seed_everything
from .training.pretrain import pretrain_option_default
from .training.sft_config import sft_option_default


DEFAULT_DATA = Path("data/raw/minimind")


def environment_report() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


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
    system = generate.add_mutually_exclusive_group()
    system.add_argument("--system-prompt")
    system.add_argument("--calculator", action="store_true", help="inject the calculator tool schema")
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-k", type=int, default=50, help="use 0 to disable top-k sampling")
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    generate.add_argument("--raw-prompt", action="store_true", help="skip the chat template (useful for base models)")
    generate.add_argument("--raw", action="store_true", help="print only the generated response")
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
    dpo.add_argument("--checkpoint", type=Path, required=True)
    dpo.add_argument("--learning-rate", type=float, default=5e-6)
    dpo.add_argument("--beta", type=float, default=0.1)
    dpo.add_argument(
        "--target-mode",
        choices=("reasoning_and_response", "response_only"),
        default="reasoning_and_response",
        help="must match the completion semantics used by the parent SFT run",
    )
    dpo.add_argument("--min-context-tokens", type=int, default=32)
    grpo = training_parser(
        "grpo", "online GRPO with verifiable math rewards", DEFAULT_DATA / "agent/agent_rl_math.jsonl"
    )
    grpo.add_argument("--checkpoint", type=Path, required=True)
    grpo.add_argument("--group-size", type=int, default=4)
    grpo.add_argument("--max-new-tokens", type=int, default=128)
    grpo.add_argument("--data-limit", type=int, default=1000)
    agent = training_parser("agent-rl", "tool-use Agent GRPO", DEFAULT_DATA / "agent/agent_rl_math.jsonl")
    agent.add_argument("--checkpoint", type=Path, required=True)
    agent.add_argument("--group-size", type=int, default=4)
    agent.add_argument("--max-new-tokens", type=int, default=128)
    agent.add_argument("--max-turns", type=int, default=6)
    agent.add_argument("--data-limit", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "doctor":
        result = environment_report()
    elif arguments.command == "pipeline":
        result = run_training_pipeline(arguments.output, device=arguments.device)
    elif arguments.command == "generate":
        result = generate_from_checkpoint(
            arguments.checkpoint,
            arguments.prompt,
            GenerationOptions(
                max_new_tokens=arguments.max_new_tokens,
                temperature=arguments.temperature,
                top_k=arguments.top_k or None,
                device=arguments.device,
                system_prompt=CalculatorEnv.tool_prompt if arguments.calculator else arguments.system_prompt,
                tokenizer_path=arguments.tokenizer,
                raw_prompt=arguments.raw_prompt,
            ),
        )
        if arguments.raw:
            print(result["response"])
            return
    elif arguments.command == "tokenize":
        tokenizer = load_tokenizer(arguments.tokenizer)
        token_ids = tokenizer.encode(arguments.text, bos=arguments.add_bos, eos=arguments.add_eos)
        decoded = tokenizer.decode(token_ids)
        result = {
            "tokenizer": str(arguments.tokenizer),
            "vocab_size": tokenizer.vocab_size,
            "text": arguments.text,
            "token_ids": token_ids,
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "token_count": len(token_ids),
            "character_count": len(arguments.text),
            "characters_per_token": len(arguments.text) / max(len(token_ids), 1),
            "decoded": decoded,
            "round_trip": decoded == arguments.text,
        }
    elif arguments.command == "audit-pretrain-data":
        tokenizer = load_tokenizer(arguments.tokenizer)
        result = audit_pretrain_jsonl(
            arguments.data,
            tokenizer,
            validation_fraction=arguments.validation_fraction,
            sequence_length=arguments.sequence_length,
            tokenizer_batch_size=arguments.tokenizer_batch_size,
        )
        if arguments.output is not None:
            save_data_audit(result, arguments.output)
            result["report"] = str(arguments.output)
    elif arguments.command == "audit-sft-data":
        tokenizer = load_tokenizer(arguments.tokenizer)
        result = audit_sft_jsonl(
            arguments.data,
            tokenizer,
            max_length=arguments.max_length,
            min_context_tokens=arguments.min_context_tokens,
            target_mode=arguments.target_mode,
            validation_fraction=arguments.validation_fraction,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
            identity_patterns=arguments.identity_pattern,
        )
        if arguments.output is not None:
            save_sft_data_audit(result, arguments.output)
            result["report"] = str(arguments.output)
    elif arguments.command == "prepare-sft-data":
        result = prepare_sft_jsonl(
            arguments.data,
            arguments.output,
            manifest_path=arguments.manifest,
            deduplicate_exact=arguments.deduplicate_exact,
            exclude_patterns=arguments.exclude_pattern,
        )
    elif arguments.command == "train-tokenizer":
        result = {"tokenizer": str(train_sentencepiece(
            arguments.data, arguments.output_prefix, vocab_size=arguments.vocab_size,
            input_sentence_size=arguments.input_sentences,
        ))}
    else:
        tokenizer = load_tokenizer(arguments.tokenizer)
        if arguments.command == "pretrain":
            # The training function receives an already-created model. Seed
            # before construction so --seed controls initial weights as well
            # as data order and later stochastic operations.
            seed_everything(arguments.seed)
            model = MiniScaleForCausalLM(MiniScaleConfig.small_64m(
                tokenizer.vocab_size,
                arguments.sequence_length,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ))
            result = run_pretrain_jsonl(model, tokenizer, arguments.data, arguments.output, PretrainOptions(
                steps=arguments.steps, batch_size=arguments.batch_size, sequence_length=arguments.sequence_length,
                learning_rate=arguments.learning_rate, min_learning_rate=arguments.min_learning_rate,
                weight_decay=arguments.weight_decay,
                adam_beta1=arguments.adam_beta1, adam_beta2=arguments.adam_beta2, adam_eps=arguments.adam_eps,
                grad_clip=arguments.grad_clip, seed=arguments.seed, precision=arguments.precision,
                warmup_steps=arguments.warmup_steps,
                gradient_accumulation_steps=arguments.gradient_accumulation,
                log_every=arguments.log_every, validation_every=arguments.validation_every,
                validation_batches=arguments.validation_batches, validation_fraction=arguments.validation_fraction,
                save_every=arguments.save_every,
                keep_last_checkpoints=arguments.keep_last,
                generation_every=arguments.generation_every,
                generation_max_new_tokens=arguments.generation_max_new_tokens,
                shuffle_buffer_size=arguments.shuffle_buffer_size,
                wandb_enabled=arguments.wandb, wandb_project=arguments.wandb_project,
                wandb_entity=arguments.wandb_entity, wandb_run_name=arguments.wandb_run_name,
                wandb_run_id=arguments.wandb_run_id, wandb_mode=arguments.wandb_mode,
                wandb_retry_every_steps=arguments.wandb_retry_every,
                resume_from=arguments.resume, allow_legacy_resume=arguments.allow_legacy_resume,
                num_workers=arguments.num_workers, device=arguments.device,
            ), validation_path=arguments.validation_data)
        else:
            checkpoint_source = (
                arguments.resume if arguments.command == "sft" and arguments.resume is not None
                else arguments.checkpoint
            )
            model = load_checkpoint(checkpoint_source)
            if model.config.vocab_size != tokenizer.vocab_size:
                raise ValueError("checkpoint vocabulary does not match tokenizer")
            if arguments.command == "sft":
                result = run_sft_jsonl(model, tokenizer, arguments.data, arguments.output, SFTOptions(
                    steps=arguments.steps,
                    batch_size=arguments.batch_size,
                    max_length=arguments.max_length,
                    min_context_tokens=arguments.min_context_tokens,
                    target_mode=arguments.target_mode,
                    learning_rate=arguments.learning_rate,
                    min_learning_rate=arguments.min_learning_rate,
                    weight_decay=arguments.weight_decay,
                    adam_beta1=arguments.adam_beta1,
                    adam_beta2=arguments.adam_beta2,
                    adam_eps=arguments.adam_eps,
                    grad_clip=arguments.grad_clip,
                    warmup_steps=arguments.warmup_steps,
                    precision=arguments.precision,
                    gradient_accumulation_steps=arguments.gradient_accumulation,
                    validation_fraction=arguments.validation_fraction,
                    validation_every=arguments.validation_every,
                    validation_batches=arguments.validation_batches,
                    save_every=arguments.save_every,
                    keep_last_checkpoints=arguments.keep_last,
                    generation_every=arguments.generation_every,
                    generation_max_new_tokens=arguments.generation_max_new_tokens,
                    deduplicate_exact=arguments.deduplicate_exact,
                    log_every=arguments.log_every,
                    num_workers=arguments.num_workers,
                    seed=arguments.seed,
                    device=arguments.device,
                    wandb_enabled=arguments.wandb,
                    wandb_project=arguments.wandb_project,
                    wandb_entity=arguments.wandb_entity,
                    wandb_run_name=arguments.wandb_run_name,
                    wandb_run_id=arguments.wandb_run_id,
                    wandb_mode=arguments.wandb_mode,
                    wandb_retry_every_steps=arguments.wandb_retry_every,
                    resume_from=arguments.resume,
                ), validation_path=arguments.validation_data, initial_checkpoint_path=arguments.checkpoint)
            elif arguments.command == "dpo":
                result = run_dpo_jsonl(model, tokenizer, arguments.data, arguments.output, DPOOptions(
                    steps=arguments.steps, batch_size=arguments.batch_size, learning_rate=arguments.learning_rate,
                    beta=arguments.beta, target_mode=arguments.target_mode,
                    min_context_tokens=arguments.min_context_tokens,
                    log_every=arguments.log_every, device=arguments.device,
                ))
            elif arguments.command == "grpo":
                result = run_grpo_jsonl(model, tokenizer, arguments.data, arguments.output, GRPOOptions(
                    steps=arguments.steps, batch_size=arguments.batch_size, group_size=arguments.group_size,
                    max_new_tokens=arguments.max_new_tokens,
                    data_limit=arguments.data_limit, log_every=arguments.log_every, device=arguments.device,
                ))
            else:
                result = run_agent_grpo_jsonl(model, tokenizer, arguments.data, arguments.output, AgentRLOptions(
                    steps=arguments.steps, batch_size=arguments.batch_size, group_size=arguments.group_size,
                    max_new_tokens=arguments.max_new_tokens,
                    max_turns=arguments.max_turns, data_limit=arguments.data_limit,
                    log_every=arguments.log_every, device=arguments.device,
                ))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
