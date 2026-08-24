from .config import MiniScaleConfig
from .model import MiniScaleForCausalLM
from .tokenizer import ByteTokenizer

__all__ = ["ByteTokenizer", "MiniScaleConfig", "MiniScaleForCausalLM"]


def main() -> None:
    from .cli import main as cli_main

    cli_main()
