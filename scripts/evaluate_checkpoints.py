"""Compare checkpoints on fixed generation and reward suites."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

# Direct file execution starts with this subdirectory on sys.path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import atomic_write_json, path_identity, tokenizer_identity
from dataset.rl_dataset import DEFAULT_RL_DATA, build_agent_corpus, build_rl_corpus
from model.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer
from trainer.checkpoint import load_checkpoint
from trainer.eval_utils import (
    DEFAULT_SFT_GENERATION_SUITE,
    evaluate_sft_generation_quality,
    load_generation_suite,
)
from trainer.rl_utils import AgentRLOptions, GRPOOptions
from trainer.train_agent import evaluate_agent
from trainer.train_grpo import evaluate_grpo
from trainer.trainer_utils import resolve_autocast_dtype, resolve_device


def evaluate_rl_checkpoints(
    checkpoints: list[str | Path],
    data_path: str | Path,
    tokenizer_path: str | Path,
    *,
    kind: str = "grpo",
    validation_fraction: float = 0.05,
    prompts: int = 100,
    max_new_tokens: int = 128,
    max_turns: int = 6,
    precision: str = "fp32",
    seed: int = 42,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Compare arbitrary stage checkpoints on one immutable RL validation suite."""

    if kind not in {"grpo", "agent"}:
        raise ValueError("kind must be 'grpo' or 'agent'")
    if not checkpoints or prompts < 1:
        raise ValueError("at least one checkpoint and one validation prompt are required")
    resolved_device = resolve_device(device)
    autocast_dtype = resolve_autocast_dtype(precision, resolved_device)
    tokenizer = load_tokenizer(tokenizer_path)
    if kind == "grpo":
        corpus = build_rl_corpus(data_path, validation_fraction=validation_fraction, seed=seed)
        tasks = list(corpus.validation)
        data_identity: dict[str, object] = {
            "identity": corpus.identity,
            "stats": asdict(corpus.stats),
        }
    else:
        corpus = build_agent_corpus(data_path, validation_fraction=validation_fraction, seed=seed)
        tasks = list(corpus.validation)
        data_identity = {"identity": corpus.identity, "stats": asdict(corpus.stats)}
    if not tasks:
        raise ValueError("validation split is empty; increase validation_fraction or use another dataset")
    if len(tasks) > prompts:
        tasks = random.Random(seed + 1).sample(tasks, prompts)

    results: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        model = load_checkpoint(checkpoint_path, resolved_device)
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"checkpoint vocabulary does not match tokenizer: {checkpoint_path}")
        if kind == "grpo":
            metrics = evaluate_grpo(
                model,
                tokenizer,
                tasks,
                GRPOOptions(max_new_tokens=max_new_tokens, device=str(resolved_device)),
                resolved_device,
                autocast_dtype=autocast_dtype,
            )
        else:
            metrics = evaluate_agent(
                model,
                tokenizer,
                tasks,
                AgentRLOptions(
                    max_new_tokens=max_new_tokens,
                    max_turns=max_turns,
                    device=str(resolved_device),
                ),
                resolved_device,
                autocast_dtype=autocast_dtype,
            )
        results.append({
            "checkpoint": str(checkpoint_path.resolve()),
            "identity": path_identity(checkpoint_path),
            "stage": checkpoint_path.parent.name,
            "metrics": metrics,
        })
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": f"{kind}_checkpoint_comparison",
        "configuration": {
            "validation_fraction": validation_fraction,
            "prompts": len(tasks),
            "max_new_tokens": max_new_tokens,
            "max_turns": max_turns if kind == "agent" else None,
            "precision": precision,
            "seed": seed,
            "device": str(resolved_device),
        },
        "data": {"path": str(Path(data_path).resolve()), **data_identity},
        "tokenizer": tokenizer_identity(tokenizer),
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
        report["report"] = str(output_path)
    return report


def evaluate_sft_checkpoints(
    checkpoints: list[str | Path],
    suite_path: str | Path,
    tokenizer_path: str | Path,
    *,
    max_new_tokens: int = 160,
    precision: str = "fp32",
    seed: int = 42,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Compare SFT checkpoints under raw and deployment-oriented decoding."""

    if not checkpoints:
        raise ValueError("at least one SFT checkpoint is required")
    resolved_device = resolve_device(device)
    autocast_dtype = resolve_autocast_dtype(precision, resolved_device)
    tokenizer = load_tokenizer(tokenizer_path)
    probes = load_generation_suite(suite_path)
    profiles = {
        "raw_greedy": {
            "temperature": 0.0,
            "top_k": None,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
        },
        "serving": {
            "temperature": 0.6,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 4,
        },
    }
    results: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        model = load_checkpoint(checkpoint_path, resolved_device).eval()
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"checkpoint vocabulary does not match tokenizer: {checkpoint_path}")
        profile_results: dict[str, object] = {}
        for name, profile in profiles.items():
            samples, summary = evaluate_sft_generation_quality(
                model,
                tokenizer,
                probes,
                device=resolved_device,
                max_new_tokens=max_new_tokens,
                autocast_dtype=autocast_dtype,
                seed=seed,
                **profile,
            )
            profile_results[name] = {
                "decoding": profile,
                "summary": summary,
                "samples": samples,
            }
        results.append({
            "checkpoint": str(checkpoint_path.resolve()),
            "identity": path_identity(checkpoint_path),
            "profiles": profile_results,
        })
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "sft_checkpoint_quality_comparison",
        "configuration": {
            "suite": path_identity(suite_path),
            "max_new_tokens": max_new_tokens,
            "precision": precision,
            "seed": seed,
            "device": str(resolved_device),
        },
        "tokenizer": tokenizer_identity(tokenizer),
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
        report["report"] = str(output_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare MiniScale checkpoints offline.')
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser('rl', help='compare checkpoints on a fixed verifiable or tool-use validation suite')
    evaluate.add_argument("--checkpoint", type=Path, action="append", required=True)
    evaluate.add_argument("--kind", choices=("grpo", "agent"), default="grpo")
    evaluate.add_argument("--data", type=Path, default=DEFAULT_RL_DATA)
    evaluate.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--validation-fraction", type=float, default=0.05)
    evaluate.add_argument("--prompts", type=int, default=100)
    evaluate.add_argument("--max-new-tokens", type=int, default=128)
    evaluate.add_argument("--max-turns", type=int, default=6)
    evaluate.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate_sft = commands.add_parser('sft', help='compare SFT checkpoints on raw and guarded generation quality')
    evaluate_sft.add_argument("--checkpoint", type=Path, action="append", required=True)
    evaluate_sft.add_argument(
        "--suite", type=Path, default=DEFAULT_SFT_GENERATION_SUITE
    )
    evaluate_sft.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    evaluate_sft.add_argument("--output", type=Path)
    evaluate_sft.add_argument("--max-new-tokens", type=int, default=160)
    evaluate_sft.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    evaluate_sft.add_argument("--seed", type=int, default=42)
    evaluate_sft.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "rl":
        result = evaluate_rl_checkpoints(
            arguments.checkpoint,
            arguments.data,
            arguments.tokenizer,
            kind=arguments.kind,
            validation_fraction=arguments.validation_fraction,
            prompts=arguments.prompts,
            max_new_tokens=arguments.max_new_tokens,
            max_turns=arguments.max_turns,
            precision=arguments.precision,
            seed=arguments.seed,
            device=arguments.device,
            output_path=arguments.output,
        )
    elif arguments.command == "sft":
        result = evaluate_sft_checkpoints(
            arguments.checkpoint,
            arguments.suite,
            arguments.tokenizer,
            max_new_tokens=arguments.max_new_tokens,
            precision=arguments.precision,
            seed=arguments.seed,
            device=arguments.device,
            output_path=arguments.output,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
