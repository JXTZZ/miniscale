"""Training runtime, optimization, and local metrics."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import MISSING, fields
from functools import partial
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import TypeVar

import numpy as np
from torch import Tensor
import torch

from dataset.file_utils import atomic_output_path, atomic_write_text
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import DEFAULT_TOKENIZER_PATH, Tokenizer
from trainer.checkpoint import load_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    """Seed Python/NumPy from the deterministic seed assigned by DataLoader."""

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision != "bf16":
        raise ValueError("precision must be 'fp32' or 'bf16'")
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 precision requires a CUDA device with BF16 support")
    return torch.bfloat16


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def warmup_cosine_multiplier(
    step_index: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step_index < warmup_steps:
        return (step_index + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return 1.0
    progress = (
        step_index / max(total_steps - 1, 1)
        if warmup_steps == 0
        else (step_index - warmup_steps + 1) / decay_steps
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    peak_lr = float(optimizer.defaults["lr"])
    if peak_lr <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= min_learning_rate <= peak_lr:
        raise ValueError("min_learning_rate must be between zero and learning_rate")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    effective_warmup = min(warmup_steps, total_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step_index: warmup_cosine_multiplier(
            step_index,
            total_steps=total_steps,
            warmup_steps=effective_warmup,
            min_lr_ratio=min_learning_rate / peak_lr,
        ),
    )


def build_adamw_optimizer(
    model: MiniScaleForCausalLM,
    *,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> torch.optim.AdamW:
    """Build explicit AdamW groups without decaying biases, norms, or embeddings."""

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2 and name != "embedding.weight":
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    if not decay or not no_decay:
        raise ValueError("optimizer requires non-empty decay and no-decay parameter groups")
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay, "group_name": "decay"},
            {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
        ],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
    )


def infinite_batches(loader: Iterable[dict[str, Tensor]]) -> Iterable[dict[str, Tensor]]:
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("data loader yielded no batches; check data, filtering, and batch size")


@torch.no_grad()
def evaluate_lm(
    model: MiniScaleForCausalLM,
    loader: Iterable[dict[str, Tensor]],
    device: torch.device,
    batches: int,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    was_training = model.training
    model.eval()
    loss_times_targets = 0.0
    total_targets = 0
    try:
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            device_batch = {name: value.to(device) for name, value in batch.items()}
            autocast = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            with autocast:
                output = model(**device_batch)
            if output.loss is not None:
                if not bool(torch.isfinite(output.loss)):
                    raise FloatingPointError(
                        f"non-finite validation loss at batch {index}: {float(output.loss)}"
                    )
                target_count = int((device_batch["labels"][:, 1:] != -100).sum())
                if target_count:
                    loss_times_targets += float(output.loss) * target_count
                    total_targets += target_count
    finally:
        model.train(was_training)
    return loss_times_targets / total_targets if total_targets else float("nan")


def append_metric(path: str | Path, metric: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, ensure_ascii=False) + "\n")


def truncate_metrics_after(path: str | Path, step: int) -> None:
    target = Path(path)
    if not target.exists():
        return
    retained: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            metric = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(metric.get("step", -1)) <= step:
            retained.append(json.dumps(metric, ensure_ascii=False))
    atomic_write_text(target, "".join(f"{line}\n" for line in retained))


def prune_periodic_checkpoints(path: str | Path, keep_last: int) -> None:
    checkpoint_dir = Path(path)
    for checkpoint in sorted(checkpoint_dir.glob("step_*.pt"))[:-keep_last]:
        checkpoint.unlink()


def mirror_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Atomically expose a second checkpoint name without duplicating disk use when possible."""

    source_path = Path(source)
    target = Path(destination)
    with atomic_output_path(target) as temporary:
        temporary.unlink()
        try:
            os.link(source_path, temporary)
        except OSError:
            shutil.copyfile(source_path, temporary)
    return target


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


def option_default(option_type: type, name: str) -> object:
    """Read a CLI default from the corresponding stage options."""
    field = next(field for field in fields(option_type) if field.name == name)
    if field.default is MISSING:
        raise ValueError(f"{option_type.__name__}.{name} has no default")
    return field.default


def training_parser(
    description: str,
    stage: str,
    data: Path,
    option_type: type,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=data)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, default=Path("artifacts") / stage)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=option_default(option_type, "batch_size"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=option_default(option_type, "device"))
    parser.add_argument("--log-every", type=int, default=option_default(option_type, "log_every"))
    return parser


def add_optimizer_arguments(parser: argparse.ArgumentParser, option_type: type) -> None:
    default = partial(option_default, option_type)
    parser.add_argument("--learning-rate", type=float, default=default("learning_rate"))
    parser.add_argument("--min-learning-rate", type=float, default=default("min_learning_rate"))
    parser.add_argument("--weight-decay", type=float, default=default("weight_decay"))
    parser.add_argument("--adam-beta1", type=float, default=default("adam_beta1"))
    parser.add_argument("--adam-beta2", type=float, default=default("adam_beta2"))
    parser.add_argument("--adam-eps", type=float, default=default("adam_eps"))
    parser.add_argument("--grad-clip", type=float, default=default("grad_clip"))
    parser.add_argument("--warmup-steps", type=int, default=default("warmup_steps"))


def add_wandb_arguments(parser: argparse.ArgumentParser, option_type: type) -> None:
    default = partial(option_default, option_type)
    parser.add_argument(
        "--wandb", action="store_true", default=default("wandb_enabled"),
        help="log training metrics to Weights & Biases",
    )
    parser.add_argument("--wandb-project", default=default("wandb_project"))
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-run-id", help="explicit W&B id; stored in checkpoints for automatic resume")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default=default("wandb_mode"),
    )
    parser.add_argument(
        "--wandb-retry-every", type=int, default=default("wandb_retry_every_steps"),
        help="retry W&B connection and pending uploads every N training steps",
    )


_Options = TypeVar("_Options")


def training_options(option_type: type[_Options], arguments: argparse.Namespace) -> _Options:
    values = vars(arguments)
    # Preserve CLI flag names and Python option field names when they differ.
    aliases = {
        "gradient_accumulation_steps": "gradient_accumulation",
        "keep_last_checkpoints": "keep_last",
        "resume_from": "resume",
        "wandb_enabled": "wandb",
        "wandb_retry_every_steps": "wandb_retry_every",
    }
    options = {}
    for field in fields(option_type):
        argument = aliases.get(field.name, field.name)
        if argument in values:
            options[field.name] = values[argument]
    if "top_k" in options:
        options["top_k"] = options["top_k"] or None
    return option_type(**options)


def load_training_model(
    arguments: argparse.Namespace, tokenizer: Tokenizer
) -> MiniScaleForCausalLM:
    model = load_checkpoint(arguments.resume or arguments.checkpoint)
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary does not match tokenizer")
    return model



def train_smoke_batches(
    model: MiniScaleForCausalLM,
    loader: Iterable[dict[str, Tensor]],
    device: torch.device,
    *,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    missing_loss: str,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    batches = iter(infinite_batches(loader))
    losses: list[float] = []
    for _ in range(steps):
        batch = {name: value.to(device) for name, value in next(batches).items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if output.loss is None:
            raise RuntimeError(missing_loss)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(output.loss.detach()))
    return {"loss": losses[-1], "mean_loss": sum(losses) / len(losses)}
