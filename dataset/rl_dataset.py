"""Verifiable math and calculator tasks with deterministic dataset splits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from dataset.file_utils import JsonlScan


DEFAULT_RL_DATA = Path("dataset/agent_rl_math.jsonl")


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


def _math_row(row: object, location: str) -> tuple[list[object], str, tuple[str, ...]]:
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
    return messages, users[-1], answers


def parse_rl_task(row: object, *, location: str = "row") -> RLTask:
    _, prompt, answers = _math_row(row, location)
    return RLTask(prompt, answers)


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
    scan = JsonlScan(source_path)
    for line_number, _, raw_line in scan:
        line = raw_line.decode("utf-8")
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
        "source": scan.identity,
        "format_version": RL_DATA_FORMAT_VERSION,
        "split_version": RL_SPLIT_VERSION,
        "validation_fraction": validation_fraction,
        "train_limit": train_limit,
        "sample_seed": seed,
    }
    return RLCorpus(tuple(train), tuple(validation), stats, identity, dict(invalid))


@dataclass(frozen=True, slots=True)
class CalculatorTask:
    question: str
    expression: str
    answer: str | tuple[str, ...]
    system_prompt: str | None = None
    tools: object | None = None


CALCULATOR_NAMES = frozenset({"calculator", "calculate_math"})


def filter_calculator_tools(tools: object | None) -> list[dict[str, object]] | None:
    """Return only schemas executable by CalculatorEnv.

    MiniMind math rows also advertise weather. Exposing that schema while the
    environment cannot execute it teaches an impossible action, so it is
    deliberately removed at the data boundary.
    """

    if tools is None:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as error:
            raise ValueError("tools must contain valid JSON") from error
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    supported: list[dict[str, object]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        name = function.get("name") if isinstance(function, dict) else None
        if name in CALCULATOR_NAMES:
            supported.append(tool)
    return supported or None


@dataclass(frozen=True, slots=True)
class AgentCorpusStats:
    rows: int
    valid_rows: int
    invalid_rows: int
    duplicate_rows: int
    train_tasks: int
    validation_tasks: int
    stripped_unsupported_tool_schemas: int


@dataclass(frozen=True, slots=True)
class AgentCorpus:
    train: tuple[CalculatorTask, ...]
    validation: tuple[CalculatorTask, ...]
    stats: AgentCorpusStats
    identity: dict[str, object]
    invalid_reasons: dict[str, int]
    unsupported_tools: dict[str, int]


def _decode_tools(value: object | None) -> object | None:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("tools must contain valid JSON") from error
    return value


def _tool_names(tools: object | None) -> list[str]:
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function", tool)
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.append(name)
    return names


def parse_agent_task(row: object, *, location: str = "row") -> tuple[CalculatorTask, list[str]]:
    messages, prompt, answers = _math_row(row, location)
    systems = [
        message for message in messages
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    system_prompt = None
    raw_tools = None
    if systems:
        system_prompt = str(systems[-1].get("content") or "").strip() or None
        raw_tools = _decode_tools(systems[-1].get("tools"))
    unsupported = [name for name in _tool_names(raw_tools) if name not in CALCULATOR_NAMES]
    supported_tools = filter_calculator_tools(raw_tools)
    if raw_tools is not None and supported_tools is None:
        raise ValueError(f"{location}: no supported calculator tool schema")
    return CalculatorTask(prompt, "", answers, system_prompt, supported_tools), unsupported


def build_agent_corpus(
    path: str | Path,
    *,
    validation_fraction: float = 0.05,
    train_limit: int | None = None,
    seed: int = 42,
) -> AgentCorpus:
    if train_limit is not None and train_limit < 1:
        raise ValueError("train_limit must be positive when set")
    source_path = Path(path)
    train: list[CalculatorTask] = []
    validation: list[CalculatorTask] = []
    invalid: Counter[str] = Counter()
    unsupported_tools: Counter[str] = Counter()
    seen: set[tuple[str, tuple[str, ...]]] = set()
    rows = duplicates = stripped = 0
    scan = JsonlScan(source_path)
    for line_number, _, raw_line in scan:
        line = raw_line.decode("utf-8")
        if not line.strip():
            continue
        rows += 1
        try:
            task, unsupported = parse_agent_task(
                json.loads(line), location=f"{source_path}:{line_number}"
            )
        except json.JSONDecodeError:
            invalid["invalid_json"] += 1
            continue
        except ValueError as error:
            message = str(error)
            reason = (
                "unsupported_tools" if "no supported" in message
                else "empty_gt" if "gt contains no answers" in message
                else "invalid_tools" if "tools" in message
                else "invalid_structure"
            )
            invalid[reason] += 1
            continue
        unsupported_tools.update(unsupported)
        stripped += len(unsupported)
        answers = (task.answer,) if isinstance(task.answer, str) else task.answer
        key = (task.question.strip(), tuple(answers))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        destination = validation if prompt_in_validation(task.question, validation_fraction) else train
        destination.append(task)
    if train_limit is not None and len(train) > train_limit:
        train = random.Random(seed).sample(train, train_limit)
    stats = AgentCorpusStats(
        rows=rows,
        valid_rows=len(seen),
        invalid_rows=sum(invalid.values()),
        duplicate_rows=duplicates,
        train_tasks=len(train),
        validation_tasks=len(validation),
        stripped_unsupported_tool_schemas=stripped,
    )
    identity = {
        "source": scan.identity,
        "format_version": 1,
        "split_version": "prompt_sha256_v1",
        "tool_registry": "calculator_v1",
        "validation_fraction": validation_fraction,
        "train_limit": train_limit,
        "sample_seed": seed,
    }
    return AgentCorpus(
        tuple(train), tuple(validation), stats, identity, dict(invalid), dict(unsupported_tools)
    )
