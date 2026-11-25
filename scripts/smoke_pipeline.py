from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch

# Support running this file directly as well as with python -m.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.file_utils import atomic_write_json, atomic_write_text
from dataset.rl_dataset import CalculatorTask, RLTask
from model.model_miniscale import MiniScaleConfig, MiniScaleForCausalLM
from model.tokenizer import ByteTokenizer
from scripts.inference import generate_from_checkpoint, GenerationOptions
from trainer.agent_env import CalculatorEnv
from trainer.checkpoint import load_checkpoint, save_checkpoint
from trainer.rl_utils import AgentRLOptions, GRPOOptions
from trainer.train_agent import run_agent_grpo
from trainer.train_grpo import run_grpo
from trainer.train_full_sft import run_sft_jsonl, SFTOptions
from trainer.train_pretrain import PretrainOptions, run_pretrain_jsonl


PRETRAIN_TEXTS = [
    "Arithmetic maps expressions to values. Two plus three equals five.",
    "An assistant should answer clearly and call tools when computation is requested.",
    "A calculator evaluates numeric expressions such as 3*4 and returns 12.",
]


SFT_CONVERSATIONS = [
    [
        {"role": "user", "content": "Return only the result: 2+3"},
        {"role": "assistant", "content": "5"},
    ],
    [
        {"role": "system", "content": CalculatorEnv.tool_prompt},
        {"role": "user", "content": "Use the calculator for 3*4."},
        {
            "role": "assistant",
            "content": '<tool_call>{"name":"calculator","arguments":{"expression":"3*4"}}</tool_call>',
        },
        {"role": "tool", "content": "12"},
        {"role": "assistant", "content": "12"},
    ],
]


RL_TASKS = [RLTask("Return only the result: 2+3", "5"), RLTask("Return only the result: 3*4", "12")]


AGENT_TASKS = [CalculatorTask("Use the calculator for 3*4.", "3*4", "12")]


def _timed_stage(function, *args, **kwargs) -> dict[str, float | str]:
    started = time.perf_counter()
    metrics = function(*args, **kwargs)
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return metrics


def run_training_pipeline(output_dir: str | Path, *, device: str = "auto") -> dict[str, object]:
    """Run a deliberately tiny integration pipeline, not a quality training run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    config = MiniScaleConfig.smoke()
    tokenizer = ByteTokenizer()
    model = MiniScaleForCausalLM(config)
    text_path = atomic_write_text(
        output / "pretrain.jsonl",
        "".join(json.dumps({"text": text}) + "\n" for text in PRETRAIN_TEXTS),
    )
    sft_path = atomic_write_text(
        output / "sft.jsonl",
        "".join(json.dumps({"conversations": messages}) + "\n" for messages in SFT_CONVERSATIONS),
    )
    quick_run = dict(
        device=device, gradient_accumulation_steps=1, validation_fraction=0,
        save_every=0, generation_every=0, warmup_steps=0,
    )
    pretrain = PretrainOptions(steps=2, batch_size=2, sequence_length=64, shuffle_buffer_size=0, **quick_run)
    sft = SFTOptions(steps=200, batch_size=2, learning_rate=3e-3, **quick_run)
    stages = {
        "pretrain": _timed_stage(
            run_pretrain_jsonl, model, tokenizer, text_path, output / "pretrain", pretrain,
        ),
    }
    pretrain_checkpoint = save_checkpoint(
        output / "pretrain.pt", model, stage="pretrain", step=pretrain.steps,
        metrics={"loss": float(stages["pretrain"]["loss"])},
    )
    stages["pretrain"]["checkpoint"] = str(pretrain_checkpoint)
    model = load_checkpoint(pretrain_checkpoint)
    stages["sft"] = _timed_stage(
        run_sft_jsonl, model, tokenizer, sft_path, output / "sft", sft,
        initial_checkpoint_path=pretrain_checkpoint,
    )
    sft_checkpoint = save_checkpoint(
        output / "sft.pt", model, stage="sft", step=sft.steps,
        metrics={"loss": float(stages["sft"]["loss"])},
    )
    stages["sft"]["checkpoint"] = str(sft_checkpoint)
    stages["rl"] = _timed_stage(
        run_grpo,
        model,
        tokenizer,
        RL_TASKS,
        output / "grpo",
        GRPOOptions(steps=1, group_size=2, max_new_tokens=16, device=device),
    )
    save_checkpoint(
        output / "rl.pt", model, stage="grpo", step=1,
        metrics={"reward_mean": float(stages["rl"]["reward_mean"])},
    )
    stages["agent_rl"] = _timed_stage(
        run_agent_grpo,
        model,
        tokenizer,
        AGENT_TASKS,
        output / "agent_rl",
        AgentRLOptions(steps=1, group_size=2, max_turns=2, max_new_tokens=100, device=device),
    )
    agent_checkpoint = save_checkpoint(
        output / "agent_rl.pt", model, stage="agent_rl", step=1,
        metrics={"reward_mean": float(stages["agent_rl"]["reward_mean"])},
    )
    generated = generate_from_checkpoint(
        agent_checkpoint, RL_TASKS[0].prompt,
        GenerationOptions(device=device, max_new_tokens=8, temperature=0),
    )
    manifest: dict[str, object] = {
        "kind": "integration-smoke-test",
        "config": asdict(config),
        "stages": stages,
        "sample": generated["response"],
    }
    manifest_path = atomic_write_json(output / "pipeline_manifest.json", manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='run the end-to-end smoke pipeline')
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    result = run_training_pipeline(arguments.output, device=arguments.device)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
