from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch

from miniscale.config import MiniScaleConfig
from miniscale.integrity import atomic_output_path
from miniscale.model import MiniScaleForCausalLM


TRAINING_CHECKPOINT_FORMAT_VERSION = 2


def save_checkpoint(
    path: str | Path,
    model: MiniScaleForCausalLM,
    *,
    stage: str,
    step: int,
    metrics: dict[str, float],
) -> Path:
    target = Path(path)
    with atomic_output_path(target) as temporary:
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
    for name in ("tokens_seen", "best_val_loss"):
        if name in training_state:
            payload[name] = training_state[name]
    with atomic_output_path(target) as temporary:
        torch.save(payload, temporary)
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
