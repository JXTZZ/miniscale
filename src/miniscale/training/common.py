from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
import json
import math
import os
import random
import shutil

import numpy as np
import torch
from torch import Tensor

from miniscale.config import MiniScaleConfig
from miniscale.integrity import atomic_write_json
from miniscale.model import MiniScaleForCausalLM


TRAINING_CHECKPOINT_FORMAT_VERSION = 2


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


def save_checkpoint(
    path: str | Path,
    model: MiniScaleForCausalLM,
    *,
    stage: str,
    step: int,
    metrics: dict[str, float],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    torch.save(
        {
            "stage": stage,
            "step": step,
            "metrics": metrics,
            "config": model.config,
            "model": model.state_dict(),
        },
        temporary,
    )
    temporary.replace(target)
    return target


def save_training_checkpoint(
    path: str | Path,
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    stage: str,
    step: int,
    metrics: dict[str, float],
    training_state: dict[str, object],
) -> Path:
    """Atomically save everything required to continue a training run."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    payload: dict[str, object] = {
        "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "step": step,
        "metrics": metrics,
        "config": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_state": training_state,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    # Duplicate these progress fields at the top level for easy inspection;
    # training_state remains the canonical object used when resuming.
    for name in ("tokens_seen", "best_val_loss"):
        if name in training_state:
            payload[name] = training_state[name]
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def read_training_checkpoint(
    path: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    """Read and structurally validate a resumable checkpoint without mutation."""

    payload = torch.load(path, map_location=device, weights_only=False)
    required = {"model", "optimizer", "scheduler", "training_state", "step"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint cannot resume training; missing: {', '.join(sorted(missing))}")
    version = payload.get("format_version", 1)
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"invalid checkpoint format_version: {version!r}")
    if version > TRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {version} is newer than supported format "
            f"{TRAINING_CHECKPOINT_FORMAT_VERSION}"
        )
    return payload


def restore_training_checkpoint(
    payload: dict[str, object],
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    restore_rng: bool = True,
) -> None:
    """Restore a payload after the caller has validated run compatibility."""

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        restore_rng_state(payload.get("rng_state"))


def load_training_checkpoint(
    path: str | Path,
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str | torch.device,
) -> dict[str, object]:
    """Compatibility wrapper for callers that do not need pre-restore checks."""

    payload = read_training_checkpoint(path, device)
    restore_training_checkpoint(payload, model, optimizer, scheduler)
    return payload


def restore_rng_state(rng_state: object) -> None:
    """Restore Python, NumPy, CPU and CUDA RNG state captured in a checkpoint."""

    if isinstance(rng_state, dict):
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("torch") is not None:
            torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            # torch.load(..., map_location="cuda") also moves serialized RNG
            # tensors to CUDA, while set_rng_state_all requires CPU ByteTensors.
            cuda_states = [state.cpu() for state in rng_state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> MiniScaleForCausalLM:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = payload["config"]
    if isinstance(config, dict):
        config = MiniScaleConfig(**config)
    model = MiniScaleForCausalLM(config)
    model.load_state_dict(payload["model"])
    return model.to(device)


def append_metric(path: str | Path, metric: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, ensure_ascii=False) + "\n")


def signature_differences(
    saved: object,
    current: object,
    prefix: str = "",
) -> dict[str, tuple[object, object]]:
    if isinstance(saved, dict) and isinstance(current, dict):
        differences: dict[str, tuple[object, object]] = {}
        for name in sorted(set(saved) | set(current)):
            path = f"{prefix}.{name}" if prefix else str(name)
            if name not in saved:
                differences[path] = ("<missing>", current[name])
            elif name not in current:
                differences[path] = (saved[name], "<missing>")
            else:
                differences.update(signature_differences(saved[name], current[name], path))
        return differences
    return {} if saved == current else {prefix: (saved, current)}


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
    target.write_text("".join(f"{line}\n" for line in retained), encoding="utf-8")


def prune_periodic_checkpoints(path: str | Path, keep_last: int) -> None:
    checkpoint_dir = Path(path)
    for checkpoint in sorted(checkpoint_dir.glob("step_*.pt"))[:-keep_last]:
        checkpoint.unlink()


def mirror_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Atomically expose a second checkpoint name without duplicating disk use when possible."""

    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source_path, temporary)
    except OSError:
        shutil.copyfile(source_path, temporary)
    temporary.replace(target)
    return target


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
                    raise FloatingPointError(f"non-finite validation loss at batch {index}: {float(output.loss)}")
                target_count = int((device_batch["labels"][:, 1:] != -100).sum())
                if target_count:
                    loss_times_targets += float(output.loss) * target_count
                    total_targets += target_count
    finally:
        model.train(was_training)
    return loss_times_targets / total_targets if total_targets else float("nan")
