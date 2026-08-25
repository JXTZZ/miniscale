from .config import MiniScaleConfig
from .model import MiniScaleForCausalLM
from .tokenizer import ByteTokenizer, HuggingFaceTokenizer, SentencePieceTokenizer

__all__ = [
    "ByteTokenizer",
    "HuggingFaceTokenizer",
    "MiniScaleConfig",
    "MiniScaleForCausalLM",
    "SentencePieceTokenizer",
]


def main() -> None:
    from .cli import main as cli_main

    cli_main()
