from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch

# Direct file execution starts with this subdirectory on sys.path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.rl_dataset import CalculatorTask, RLTask
from model.model_miniscale import MiniScaleConfig, MiniScaleForCausalLM
from model.tokenizer import ByteTokenizer
from trainer.agent_env import CalculatorEnv
from trainer.checkpoint import save_checkpoint
from trainer.rl_utils import AgentRLOptions, GRPOOptions
from trainer.train_agent import run_agent_grpo
from trainer.train_full_sft import SmokeSFTOptions, run_sft
from trainer.train_grpo import run_grpo
from trainer.train_pretrain import SmokePretrainOptions, run_pretrain


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
    stages: dict[str, dict[str, float | str]] = {}
    stages["pretrain"] = _timed_stage(
        run_pretrain,
        model,
        tokenizer,
        PRETRAIN_TEXTS,
        output,
        SmokePretrainOptions(device=device),
    )
    stages["sft"] = _timed_stage(
        run_sft,
        model,
        tokenizer,
        SFT_CONVERSATIONS,
        output,
        # Enough steps to deliberately overfit the two-example integration set;
        # this proves stage hand-off, not generalization.
        SmokeSFTOptions(steps=200, batch_size=2, learning_rate=3e-3, device=device),
    )
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
    save_checkpoint(
        output / "agent_rl.pt", model, stage="agent_rl", step=1,
        metrics={"reward_mean": float(stages["agent_rl"]["reward_mean"])},
    )
    prompt_ids = tokenizer.encode("<|user|>\n2+3?<|end|>\n<|assistant|>\n", bos=True)
    prompt = torch.tensor([prompt_ids], device=next(model.parameters()).device)
    model.eval()
    generated = model.generate(prompt, max_new_tokens=8, temperature=0)
    manifest: dict[str, object] = {
        "kind": "integration-smoke-test",
        "config": asdict(config),
        "stages": stages,
        "sample": tokenizer.decode(generated[0, len(prompt_ids) :].tolist()),
    }
    manifest_path = output / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
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
