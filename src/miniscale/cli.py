from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import torch

from .agent_env import CalculatorEnv
from .inference import GenerationOptions, generate_from_checkpoint
from .pipeline import run_training_pipeline


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
    generate.add_argument("--prompt", required=True)
    system = generate.add_mutually_exclusive_group()
    system.add_argument("--system-prompt")
    system.add_argument("--calculator", action="store_true", help="inject the calculator tool schema")
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.7)
    generate.add_argument("--top-k", type=int, default=50, help="use 0 to disable top-k sampling")
    generate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    generate.add_argument("--raw", action="store_true", help="print only the generated response")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "doctor":
        result = environment_report()
    elif arguments.command == "pipeline":
        result = run_training_pipeline(arguments.output, device=arguments.device)
    else:
        result = generate_from_checkpoint(
            arguments.checkpoint,
            arguments.prompt,
            GenerationOptions(
                max_new_tokens=arguments.max_new_tokens,
                temperature=arguments.temperature,
                top_k=arguments.top_k or None,
                device=arguments.device,
                system_prompt=CalculatorEnv.tool_prompt if arguments.calculator else arguments.system_prompt,
            ),
        )
        if arguments.raw:
            print(result["response"])
            return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
