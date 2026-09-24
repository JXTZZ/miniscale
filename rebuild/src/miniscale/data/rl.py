from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

from ..integrity import atomic_write_json, path_identity
from .metrics import integer_percentiles


RL_DATA_FORMAT_VERSION = 1
RL_SPLIT_VERSION = "prompt_sha256_v1"


@dataclass(frozen=True, slots=True)
class RLTask:
    prompt: str
    answer: str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RLCorpusStats:
    rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_rows: int
    train_tasks: int
    validation_tasks: int


@dataclass(frozen=True, slots=True)
class RLCorpus:
    train: tuple[RLTask, ...]
    validation: tuple[RLTask, ...]
    stats: RLCorpusStats
    identity: dict[str, object]
    invalid_reasons: dict[str, int]


def parse_rl_task(row: object, *, location: str = "row") -> RLTask:
    if not isinstance(row, dict):
        raise ValueError(f"{location}: expected a JSON object")
    messages = row.get("conversations")
    if not isinstance(messages, list):
        raise ValueError(f"{location}: conversations must be a list")
    users = [
        str(message.get("content") or "").strip()
        for message in messages
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not users or not users[-1]:
        raise ValueError(f"{location}: missing a non-empty user prompt")
    ground_truth = row.get("gt")
    if not isinstance(ground_truth, list):
        raise ValueError(f"{location}: gt must be a list")
    answers = tuple(str(item).strip() for item in ground_truth if str(item).strip())
    if not answers:
        raise ValueError(f"{location}: gt contains no answers")
    return RLTask(users[-1], answers)


def prompt_in_validation(prompt: str, fraction: float) -> bool:
    if not 0 <= fraction < 1:
        raise ValueError("validation fraction must be in [0, 1)")
    if fraction == 0:
        return False
    digest = hashlib.sha256(prompt.strip().encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < fraction


def _invalid_reason(error: ValueError) -> str:
    message = str(error)
    for reason, marker in {
        "not_an_object": "expected a JSON object",
        "invalid_conversations": "conversations must be a list",
        "missing_prompt": "missing a non-empty user prompt",
        "invalid_gt": "gt must be a list",
        "empty_gt": "gt contains no answers",
    }.items():
        if marker in message:
            return reason
    return "other"


def build_rl_corpus(
    path: str | Path,
    *,
    validation_fraction: float = 0.05,
    train_limit: int | None = None,
    seed: int = 42,
) -> RLCorpus:
    """Fully scan, deduplicate, split, then globally sample verifiable tasks."""

    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be positive when set")
    source_path = Path(path)
    train: list[RLTask] = []
    validation: list[RLTask] = []
    invalid: Counter[str] = Counter()
    seen: set[tuple[str, tuple[str, ...]]] = set()
    rows = duplicate_rows = 0
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
                task = parse_rl_task(row, location=f"{source_path}:{line_number}")
            except json.JSONDecodeError:
                invalid["invalid_json"] += 1
                continue
            except ValueError as error:
                invalid[_invalid_reason(error)] += 1
                continue
            answers = (task.answer,) if isinstance(task.answer, str) else task.answer
            key = (task.prompt.strip(), tuple(answers))
            if key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
            destination = validation if prompt_in_validation(task.prompt, validation_fraction) else train
            destination.append(task)
    if train_limit is not None and len(train) > train_limit:
        train = random.Random(seed).sample(train, train_limit)
    stats = RLCorpusStats(
        rows=rows,
        valid_rows=len(seen),
        invalid_rows=sum(invalid.values()),
        duplicate_rows=duplicate_rows,
        train_tasks=len(train),
        validation_tasks=len(validation),
    )
    identity = {
        "source": path_identity(source_path),
        "format_version": RL_DATA_FORMAT_VERSION,
        "split_version": RL_SPLIT_VERSION,
        "validation_fraction": validation_fraction,
        "train_limit": train_limit,
        "sample_seed": seed,
    }
    return RLCorpus(tuple(train), tuple(validation), stats, identity, dict(invalid))


def load_rl_tasks(path: str | Path, limit: int | None = None) -> list[RLTask]:
    """Compatibility loader with deterministic global sampling and no validation split."""

    return list(build_rl_corpus(path, validation_fraction=0, train_limit=limit).train)


def audit_rl_jsonl(
    path: str | Path,
    *,
    validation_fraction: float = 0.05,
    sample_size: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    corpus = build_rl_corpus(path, validation_fraction=validation_fraction, seed=seed)
    all_tasks = list(corpus.train) + list(corpus.validation)
    selected = random.Random(seed + 1).sample(all_tasks, min(sample_size, len(all_tasks)))
    prompt_lengths = [len(task.prompt) for task in selected]
    prompt_percentiles = integer_percentiles(prompt_lengths, (50, 90, 99))

    return {
        "schema_version": 1,
        "kind": "grpo_data_audit",
        "data": {"path": str(Path(path).resolve()), "identity": corpus.identity},
        "configuration": {
            "validation_fraction": validation_fraction,
            "sample_size": sample_size,
            "seed": seed,
        },
        "structure": {**asdict(corpus.stats), "invalid_reasons": corpus.invalid_reasons},
        "sample": {
            "tasks": len(selected),
            "prompt_characters": {
                "p50": prompt_percentiles.get("p50", 0),
                "p90": prompt_percentiles.get("p90", 0),
                "p99": prompt_percentiles.get("p99", 0),
                "max": prompt_percentiles.get("max", 0),
            },
        },
    }


def save_rl_data_audit(report: dict[str, object], path: str | Path) -> Path:
    return atomic_write_json(path, report)
