from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import json
import random

import torch
from torch import Tensor

from miniscale.config import MiniScaleConfig
from miniscale.model import MiniScaleForCausalLM


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def infinite_batches(loader: Iterable[dict[str, Tensor]]) -> Iterable[dict[str, Tensor]]:
    while True:
        yield from loader


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


def load_training_checkpoint(
    path: str | Path,
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str | torch.device,
) -> dict[str, object]:
    payload = torch.load(path, map_location=device, weights_only=False)
    required = {"model", "optimizer", "scheduler", "training_state", "step"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint cannot resume training; missing: {', '.join(sorted(missing))}")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload.get("rng_state"))
    return payload


def restore_rng_state(rng_state: object) -> None:
    """Restore Python, CPU and CUDA RNG state captured in a checkpoint."""

    if isinstance(rng_state, dict):
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
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
def evaluate_lm(model: MiniScaleForCausalLM, loader: Iterable[dict[str, Tensor]], device: torch.device, batches: int) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        output = model(**{name: value.to(device) for name, value in batch.items()})
        if output.loss is not None:
            losses.append(float(output.loss))
    model.train(was_training)
    return sum(losses) / len(losses) if losses else float("nan")
