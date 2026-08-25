from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from .agent_env import CalculatorEnv
from .config import MiniScaleConfig
from .inference import GenerationOptions, generate_from_checkpoint
from .model import MiniScaleForCausalLM
from .pipeline import run_training_pipeline
from .tokenizer import load_tokenizer, train_sentencepiece
from .training import (
    AgentRLOptions, DPOOptions, GRPOOptions, PretrainOptions, SFTOptions,
    run_agent_grpo_jsonl, run_dpo_jsonl, run_grpo_jsonl, run_pretrain_jsonl, run_sft_jsonl,
)
from .training.common import load_checkpoint


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
    pretrain.add_argument("--sequence-length", type=int, default=768)
    pretrain.add_argument("--gradient-accumulation", type=int, default=16)
    pretrain.add_argument("--learning-rate", type=float, default=3e-4)
    pretrain.add_argument("--min-learning-rate", type=float, default=3e-5)
    pretrain.add_argument("--warmup-steps", type=int, default=200)
    pretrain.add_argument("--validation-every", type=int, default=200)
    pretrain.add_argument("--validation-batches", type=int, default=20)
    pretrain.add_argument("--validation-data", type=Path, help="optional dedicated validation JSONL")
    pretrain.add_argument("--save-every", type=int, default=500, help="use 0 to disable periodic checkpoints")
    pretrain.add_argument("--keep-last", type=int, default=3, help="number of periodic checkpoints to retain")
    pretrain.add_argument("--shuffle-buffer-size", type=int, default=8192, help="use 0 to preserve JSONL order")
    pretrain.add_argument("--generation-every", type=int, default=1000, help="use 0 to disable generation evaluation")
    pretrain.add_argument("--generation-max-new-tokens", type=int, default=64)
    pretrain.add_argument("--wandb", action="store_true", help="log training metrics to Weights & Biases")
    pretrain.add_argument("--wandb-project", default="MiniScale")
    pretrain.add_argument("--wandb-entity")
    pretrain.add_argument("--wandb-run-name")
    pretrain.add_argument("--wandb-run-id", help="explicit W&B id; stored in checkpoints for automatic resume")
    pretrain.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    pretrain.add_argument("--resume", type=Path, help="resume from a full training checkpoint")
    pretrain.add_argument("--num-workers", type=int, default=0)
    sft = training_parser("sft", "supervised fine-tuning from a pretrain checkpoint", DEFAULT_DATA / "sft/sft_t2t_mini.jsonl")
    sft.add_argument("--checkpoint", type=Path, required=True)
    sft.add_argument("--gradient-accumulation", type=int, default=16)
    sft.add_argument("--learning-rate", type=float, default=2e-5)
    dpo = training_parser("dpo", "preference optimization from sft.pt", DEFAULT_DATA / "preference/dpo.jsonl")
    dpo.add_argument("--checkpoint", type=Path, required=True)
    dpo.add_argument("--learning-rate", type=float, default=5e-6)
    dpo.add_argument("--beta", type=float, default=0.1)
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
    elif arguments.command == "train-tokenizer":
        result = {"tokenizer": str(train_sentencepiece(
            arguments.data, arguments.output_prefix, vocab_size=arguments.vocab_size,
            input_sentence_size=arguments.input_sentences,
        ))}
    else:
        tokenizer = load_tokenizer(arguments.tokenizer)
        if arguments.command == "pretrain":
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
                warmup_steps=arguments.warmup_steps,
                gradient_accumulation_steps=arguments.gradient_accumulation,
                log_every=arguments.log_every, validation_every=arguments.validation_every,
                validation_batches=arguments.validation_batches, save_every=arguments.save_every,
                keep_last_checkpoints=arguments.keep_last, resume_from=arguments.resume,
                generation_every=arguments.generation_every,
                generation_max_new_tokens=arguments.generation_max_new_tokens,
                shuffle_buffer_size=arguments.shuffle_buffer_size,
                wandb_enabled=arguments.wandb, wandb_project=arguments.wandb_project,
                wandb_entity=arguments.wandb_entity, wandb_run_name=arguments.wandb_run_name,
                wandb_run_id=arguments.wandb_run_id, wandb_mode=arguments.wandb_mode,
                num_workers=arguments.num_workers, device=arguments.device,
            ), validation_path=arguments.validation_data)
        else:
            model = load_checkpoint(arguments.checkpoint)
            if model.config.vocab_size != tokenizer.vocab_size:
                raise ValueError("checkpoint vocabulary does not match tokenizer")
            if arguments.command == "sft":
                result = run_sft_jsonl(model, tokenizer, arguments.data, arguments.output, SFTOptions(
                    steps=arguments.steps, batch_size=arguments.batch_size, learning_rate=arguments.learning_rate,
                    gradient_accumulation_steps=arguments.gradient_accumulation, log_every=arguments.log_every,
                    device=arguments.device,
                ))
            elif arguments.command == "dpo":
                result = run_dpo_jsonl(model, tokenizer, arguments.data, arguments.output, DPOOptions(
                    steps=arguments.steps, batch_size=arguments.batch_size, learning_rate=arguments.learning_rate,
                    beta=arguments.beta, log_every=arguments.log_every, device=arguments.device,
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
