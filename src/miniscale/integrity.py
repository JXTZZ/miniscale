from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .tokenizer import Tokenizer


def _update_hash_from_file(digest: Any, path: Path) -> int:
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size


def path_identity(path: str | Path) -> dict[str, object]:
    """Return a SHA-256 content identity that is stable when inputs move."""

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"input does not exist: {target}")
    digest = hashlib.sha256()
    if target.is_file():
        expected_size = target.stat().st_size
        size = _update_hash_from_file(digest, target)
        if size != expected_size:
            raise RuntimeError(f"input changed while hashing: {target}")
        return {"kind": "file", "sha256": digest.hexdigest(), "size_bytes": size}
    if not target.is_dir():
        raise ValueError(f"input must be a file or directory: {target}")

    files_in_identity = 0
    total_size = 0
    for child in sorted(candidate for candidate in target.rglob("*") if candidate.is_file()):
        relative = child.relative_to(target).as_posix().encode()
        child_size = child.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(child_size.to_bytes(8, "big"))
        hashed_size = _update_hash_from_file(digest, child)
        if hashed_size != child_size:
            raise RuntimeError(f"input changed while hashing: {child}")
        total_size += hashed_size
        files_in_identity += 1
    return {
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "size_bytes": total_size,
        "files": files_in_identity,
    }


def tokenizer_identity(tokenizer: Tokenizer) -> dict[str, object]:
    model_path = getattr(tokenizer, "model_path", None)
    identity: dict[str, object] = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    if model_path is not None:
        identity["files"] = path_identity(model_path)
    return identity
