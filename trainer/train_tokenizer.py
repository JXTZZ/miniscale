from __future__ import annotations

import argparse
from collections.abc import Iterator
import json
from pathlib import Path
import sys

# Direct file execution starts with this subdirectory on sys.path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.lm_dataset import DEFAULT_PRETRAIN_DATA


DEFAULT_VOCAB_SIZE = 8192
DEFAULT_INPUT_SENTENCES = 1_000_000


def iter_tokenizer_texts(jsonl_path: str | Path, limit: int | None = None) -> Iterator[str]:
    with Path(jsonl_path).open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if limit is not None and index >= limit:
                break
            text = json.loads(line).get("text")
            if isinstance(text, str) and text.strip():
                yield text


def train_sentencepiece(
    jsonl_path: str | Path,
    output_prefix: str | Path,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    input_sentence_size: int = DEFAULT_INPUT_SENTENCES,
) -> Path:
    import sentencepiece as spm

    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter_tokenizer_texts(jsonl_path, input_sentence_size),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        normalization_rule_name="identity",
        character_coverage=0.9995,
        byte_fallback=True,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        user_defined_symbols=["<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|end|>"],
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return prefix.with_suffix(".model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SentencePiece from pretraining JSONL.")
    parser.add_argument("--data", type=Path, default=DEFAULT_PRETRAIN_DATA)
    parser.add_argument("--output-prefix", type=Path, default=Path("artifacts/tokenizer/miniscale"))
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--input-sentences", type=int, default=DEFAULT_INPUT_SENTENCES)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    result = {"tokenizer": str(train_sentencepiece(
        arguments.data, arguments.output_prefix, vocab_size=arguments.vocab_size,
        input_sentence_size=arguments.input_sentences,
    ))}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
