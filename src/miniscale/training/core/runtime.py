from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
import math
import random

import numpy as np
import torch
from torch import Tensor

from miniscale.model import MiniScaleForCausalLM


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
