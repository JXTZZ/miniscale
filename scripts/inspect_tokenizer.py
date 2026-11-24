from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Direct file execution starts with this subdirectory on sys.path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.tokenizer import DEFAULT_TOKENIZER_PATH, load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='inspect token IDs and pieces for text')
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--text", required=True)
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--add-eos", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
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
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
