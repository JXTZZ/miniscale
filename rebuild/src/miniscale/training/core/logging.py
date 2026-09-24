"""Compact human-readable formatting for full-fidelity training metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _number(metric: Mapping[str, object], name: str) -> float | None:
    value = metric.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _rate(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.1f}"


def _append_number(
    fields: list[str],
    metric: Mapping[str, object],
    name: str,
    label: str,
    format_spec: str,
) -> None:
    value = _number(metric, name)
    if value is not None:
        fields.append(f"{label} {value:{format_spec}}")


def format_training_metric(metric: Mapping[str, object]) -> str:
    """Return a compact terminal line without changing the persisted metric.

    Every stage shares the same loss/LR/gradient/throughput/memory spine.
    Preference and RL stages append only their most useful live signals.
    Unknown or unavailable values are omitted instead of rendering ``None``.
    """

    stage = str(metric.get("stage", "train"))
    step = metric.get("step", "?")
    fields = [f"[{stage}] step {step}"]

    loss = _number(metric, "train_loss")
    if loss is None:
        loss = _number(metric, "loss")
    if loss is not None:
        fields.append(f"loss {loss:.3f}")

    learning_rate = _number(metric, "learning_rate")
    if learning_rate is not None:
        fields.append(f"lr {_scientific(learning_rate)}")

    grad_norm = _number(metric, "grad_norm")
    if grad_norm is not None:
        clipped = "*" if metric.get("grad_was_clipped") is True else ""
        fields.append(f"grad {grad_norm:.2f}{clipped}")

    tokens_per_second = _number(metric, "tokens_per_second")
    if tokens_per_second is not None:
        fields.append(f"tok/s {_rate(tokens_per_second)}")

    memory = _number(metric, "cuda_peak_memory_mb")
    if memory is not None:
        fields.append(f"mem {memory:.0f}MB")

    if stage == "dpo":
        _append_number(fields, metric, "preference_accuracy", "acc", ".1%")
    elif stage == "grpo":
        _append_number(fields, metric, "reward_mean", "reward", ".3f")
        _append_number(fields, metric, "kl", "kl", ".3f")
    elif stage == "agent_rl":
        _append_number(fields, metric, "reward_mean", "reward", ".3f")
        _append_number(fields, metric, "success_rate", "success", ".1%")

    if tokens_per_second is None:
        rollouts_per_second = _number(metric, "rollouts_per_second")
        if rollouts_per_second is not None:
            fields.append(f"roll/s {_rate(rollouts_per_second)}")

    validation_loss = _number(metric, "validation_loss")
    if validation_loss is not None and math.isfinite(validation_loss):
        fields.append(f"val {validation_loss:.3f}")
    elif stage in {"grpo", "agent_rl"}:
        _append_number(fields, metric, "validation_reward", "val_reward", ".3f")

    return " | ".join(fields)


__all__ = ["format_training_metric"]
