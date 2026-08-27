from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

from .agent_env import CALCULATOR_NAMES, CalculatorTask, filter_calculator_tools
from .integrity import atomic_write_json, path_identity
from .rl_data import prompt_in_validation


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
    return CalculatorTask(users[-1], "", answers, system_prompt, supported_tools), unsupported


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
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
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
        "source": path_identity(source_path),
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


def load_agent_tasks(path: str | Path, limit: int | None = None) -> list[CalculatorTask]:
    return list(build_agent_corpus(path, validation_fraction=0, train_limit=limit).train)


def audit_agent_jsonl(
    path: str | Path,
    *,
    validation_fraction: float = 0.05,
    seed: int = 42,
) -> dict[str, object]:
    corpus = build_agent_corpus(path, validation_fraction=validation_fraction, seed=seed)
    return {
        "schema_version": 1,
        "kind": "agent_rl_data_audit",
        "data": {"path": str(Path(path).resolve()), "identity": corpus.identity},
        "configuration": {"validation_fraction": validation_fraction, "seed": seed},
        "structure": {
            **asdict(corpus.stats),
            "invalid_reasons": corpus.invalid_reasons,
            "unsupported_tools": corpus.unsupported_tools,
        },
    }


def save_agent_data_audit(report: dict[str, object], path: str | Path) -> Path:
    return atomic_write_json(path, report)
