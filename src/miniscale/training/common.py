from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import Tensor

from miniscale.config import MiniScaleConfig
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
    torch.save(
        {
            "stage": stage,
            "step": step,
            "metrics": metrics,
            "config": model.config,
            "model": model.state_dict(),
        },
        target,
    )
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
