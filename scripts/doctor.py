"""Report the active Python, PyTorch, and CUDA environment."""

from __future__ import annotations

import argparse
import json
import platform

import torch


def environment_report() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    print(json.dumps(environment_report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
