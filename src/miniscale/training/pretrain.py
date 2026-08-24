from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from miniscale.data import PretrainDataset, collate_lm_batch
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer
from .common import infinite_batches, resolve_device, save_checkpoint, seed_everything


@dataclass(slots=True)
class PretrainOptions:
    steps: int = 100
    batch_size: int = 8
    sequence_length: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"


def run_pretrain(
    model: MiniScaleForCausalLM,
    tokenizer: ByteTokenizer,
    texts: list[str],
    output_dir: str | Path,
    options: PretrainOptions | None = None,
) -> dict[str, float | str]:
    options = options or PretrainOptions()
    if options.steps < 1:
        raise ValueError("steps must be positive")
    seed_everything(options.seed)
    device = resolve_device(options.device)
    model.to(device).train()
    dataset = PretrainDataset(texts, tokenizer, options.sequence_length)
    if not dataset:
        raise ValueError("pretraining corpus produced no examples")
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_lm_batch(rows, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    losses: list[float] = []
    batches = iter(infinite_batches(loader))
    for _ in range(options.steps):
        batch = {name: value.to(device) for name, value in next(batches).items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if output.loss is None:
            raise RuntimeError("model did not return a pretraining loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        losses.append(float(output.loss.detach()))

    metrics = {
        "loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
    }
    checkpoint = save_checkpoint(
        Path(output_dir) / "pretrain.pt",
        model,
        stage="pretrain",
        step=options.steps,
        metrics=metrics,
    )
    return {**metrics, "checkpoint": str(checkpoint), "device": str(device)}
