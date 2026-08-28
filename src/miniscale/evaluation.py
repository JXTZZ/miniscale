from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

from .data.agent import build_agent_corpus
from .data.rl import build_rl_corpus
from .integrity import atomic_write_json, path_identity, tokenizer_identity
from .tokenizer import load_tokenizer
from .training.configs.rl import AgentRLOptions, GRPOOptions
from .training.core.checkpoint import load_checkpoint
from .training.core.runtime import resolve_autocast_dtype, resolve_device
from .training.stages.agent_rl import evaluate_agent
from .training.stages.grpo import evaluate_grpo


def evaluate_rl_checkpoints(
    checkpoints: list[str | Path],
    data_path: str | Path,
    tokenizer_path: str | Path,
    *,
    kind: str = "grpo",
    validation_fraction: float = 0.05,
    prompts: int = 100,
    max_new_tokens: int = 128,
    max_turns: int = 6,
    precision: str = "fp32",
    seed: int = 42,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Compare arbitrary stage checkpoints on one immutable RL validation suite."""

    if kind not in {"grpo", "agent"}:
        raise ValueError("kind must be 'grpo' or 'agent'")
    if not checkpoints or prompts < 1:
        raise ValueError("at least one checkpoint and one validation prompt are required")
    resolved_device = resolve_device(device)
    autocast_dtype = resolve_autocast_dtype(precision, resolved_device)
    tokenizer = load_tokenizer(tokenizer_path)
    if kind == "grpo":
        corpus = build_rl_corpus(data_path, validation_fraction=validation_fraction, seed=seed)
        tasks = list(corpus.validation)
        data_identity: dict[str, object] = {
            "identity": corpus.identity,
            "stats": asdict(corpus.stats),
        }
    else:
        corpus = build_agent_corpus(data_path, validation_fraction=validation_fraction, seed=seed)
        tasks = list(corpus.validation)
        data_identity = {"identity": corpus.identity, "stats": asdict(corpus.stats)}
    if not tasks:
        raise ValueError("validation split is empty; increase validation_fraction or use another dataset")
    if len(tasks) > prompts:
        tasks = random.Random(seed + 1).sample(tasks, prompts)

    results: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        model = load_checkpoint(checkpoint_path, resolved_device)
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError(f"checkpoint vocabulary does not match tokenizer: {checkpoint_path}")
        if kind == "grpo":
            metrics = evaluate_grpo(
                model,
                tokenizer,
                tasks,
                GRPOOptions(max_new_tokens=max_new_tokens, device=str(resolved_device)),
                resolved_device,
                autocast_dtype=autocast_dtype,
            )
        else:
            metrics = evaluate_agent(
                model,
                tokenizer,
                tasks,
                AgentRLOptions(
                    max_new_tokens=max_new_tokens,
                    max_turns=max_turns,
                    device=str(resolved_device),
                ),
                resolved_device,
                autocast_dtype=autocast_dtype,
            )
        results.append({
            "checkpoint": str(checkpoint_path.resolve()),
            "identity": path_identity(checkpoint_path),
            "stage": checkpoint_path.parent.name,
            "metrics": metrics,
        })
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": f"{kind}_checkpoint_comparison",
        "configuration": {
            "validation_fraction": validation_fraction,
            "prompts": len(tasks),
            "max_new_tokens": max_new_tokens,
            "max_turns": max_turns if kind == "agent" else None,
            "precision": precision,
            "seed": seed,
            "device": str(resolved_device),
        },
        "data": {"path": str(Path(data_path).resolve()), **data_identity},
        "tokenizer": tokenizer_identity(tokenizer),
        "results": results,
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
        report["report"] = str(output_path)
    return report
