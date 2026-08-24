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
