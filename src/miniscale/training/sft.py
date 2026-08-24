from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from miniscale.data import SFTDataset, collate_lm_batch
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer
from .common import infinite_batches, resolve_device, save_checkpoint, seed_everything


@dataclass(slots=True)
class SFTOptions:
    steps: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"


def run_sft(
    model: MiniScaleForCausalLM,
    tokenizer: ByteTokenizer,
    conversations: list[list[dict[str, str]]],
    output_dir: str | Path,
    options: SFTOptions | None = None,
) -> dict[str, float | str]:
    options = options or SFTOptions()
    if options.steps < 1:
        raise ValueError("steps must be positive")
    seed_everything(options.seed)
    device = resolve_device(options.device)
    model.to(device).train()
    dataset = SFTDataset(conversations, tokenizer, model.config.max_position_embeddings)
    if not dataset:
        raise ValueError("SFT requires at least one conversation")
    loader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_lm_batch(rows, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    batches = iter(infinite_batches(loader))
    losses: list[float] = []
    for _ in range(options.steps):
        batch = {name: value.to(device) for name, value in next(batches).items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        if output.loss is None:
            raise RuntimeError("model did not return an SFT loss")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        losses.append(float(output.loss.detach()))
    metrics = {"loss": losses[-1], "mean_loss": sum(losses) / len(losses)}
    checkpoint = save_checkpoint(
        Path(output_dir) / "sft.pt",
        model,
        stage="sft",
        step=options.steps,
        metrics=metrics,
    )
    return {**metrics, "checkpoint": str(checkpoint), "device": str(device)}
