from .config import MiniScaleConfig
from .model import MiniScaleForCausalLM
from .tokenizer import ByteTokenizer, SentencePieceTokenizer

__all__ = ["ByteTokenizer", "MiniScaleConfig", "MiniScaleForCausalLM", "SentencePieceTokenizer"]


def main() -> None:
    from .cli import main as cli_main

    cli_main()
