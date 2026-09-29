from __future__ import annotations

from pathlib import Path
import random
import warnings

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
        "rng_state": capture_rng_state(),
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
    """Validate without mutation; tensors stay on CPU until restoration.

    ``device`` is retained for compatibility with existing callers.
    """

    # Keep serialized optimizer tensors off the GPU. load_state_dict moves
    # only the live state to each parameter's device after identity validation.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("training checkpoint must be a mapping")
    required = {"config", "model", "optimizer", "scheduler", "training_state", "step"}
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
    for name in ("model", "optimizer", "scheduler", "training_state"):
        if not isinstance(payload[name], dict):
            raise ValueError(f"checkpoint {name} must be a mapping")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if version >= 2:
        rng = payload.get("rng_state")
        if not isinstance(rng, dict) or not {"python", "numpy", "torch", "cuda"} <= rng.keys():
            raise ValueError("checkpoint is missing complete rng_state for exact resume")
        if not isinstance(rng["torch"], torch.Tensor) or rng["torch"].dtype != torch.uint8:
            raise ValueError("checkpoint torch RNG must be a byte tensor")
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


def capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


class UpdateBoundary:
    """Track whether an interrupted update can produce an exact checkpoint.

    Before optimizer mutation, discard partial gradients and rewind RNG to the
    update start. During optimizer mutation (or between RL policy epochs), the
    last periodic checkpoint is the recovery point; copying AdamW every step
    would be too expensive for this single-device trainer.
    """

    def __init__(self) -> None:
        self.rng_state: dict[str, object] | None = None
        self.mutating = False

    def begin(self) -> None:
        self.rng_state = capture_rng_state()
        self.mutating = False

    def mark_mutating(self) -> None:
        self.mutating = True

    def commit(self) -> None:
        self.rng_state = None
        self.mutating = False

    def prepare_emergency(self) -> bool:
        if self.mutating:
            warnings.warn(
                "update interrupted after optimizer mutation began, before update bookkeeping completed; "
                "emergency checkpoint was not saved. "
                "Resume from the last complete periodic/best checkpoint.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        if self.rng_state is not None:
            restore_rng_state(self.rng_state)
        return True


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
    # A full training payload includes AdamW state that inference never uses.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    if isinstance(config, dict):
        config = MiniScaleConfig(**config)
    # Loading weights must not advance the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        model = MiniScaleForCausalLM(config)
    model.load_state_dict(payload["model"])
    del payload
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
