from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from miniscale.data import JsonlPretrainDataset, PretrainDataset, collate_lm_batch
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import ByteTokenizer, Tokenizer
from .common import append_metric, evaluate_lm, infinite_batches, resolve_device, save_checkpoint, seed_everything


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
    gradient_accumulation_steps: int = 1
    log_every: int = 10
    validation_every: int = 200
    validation_batches: int = 20
    num_workers: int = 0
    validation_fraction: float = 0.005


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


def run_pretrain_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    train_path: str | Path,
    output_dir: str | Path,
    options: PretrainOptions,
    validation_path: str | Path | None = None,
) -> dict[str, float | str]:
    seed_everything(options.seed)
    device = resolve_device(options.device)
    model.to(device).train()
    train_dataset = JsonlPretrainDataset(
        train_path, tokenizer, options.sequence_length, split="train",
        validation_fraction=options.validation_fraction,
    )
    collate = lambda rows: collate_lm_batch(rows, tokenizer.pad_token_id)
    loader = DataLoader(train_dataset, batch_size=options.batch_size, collate_fn=collate, num_workers=options.num_workers)
    batches = iter(infinite_batches(loader))
    validation_loader = None
    if validation_path or options.validation_fraction > 0:
        validation_loader = DataLoader(
            JsonlPretrainDataset(
                validation_path or train_path, tokenizer, options.sequence_length,
                split="all" if validation_path else "validation",
                validation_fraction=options.validation_fraction,
            ),
            batch_size=options.batch_size,
            collate_fn=collate,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay, betas=(0.9, 0.95))
    output = Path(output_dir)
    metrics_path = output / "pretrain_metrics.jsonl"
    last_loss = float("nan")
    tokens_seen = 0
    for step in range(1, options.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(options.gradient_accumulation_steps):
            batch = {name: value.to(device) for name, value in next(batches).items()}
            result = model(**batch)
            if result.loss is None:
                raise RuntimeError("model did not return a pretraining loss")
            (result.loss / options.gradient_accumulation_steps).backward()
            micro_losses.append(float(result.loss.detach()))
            tokens_seen += int(batch["attention_mask"].sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        last_loss = sum(micro_losses) / len(micro_losses)
        if step == 1 or step % options.log_every == 0 or step == options.steps:
            metric: dict[str, object] = {
                "stage": "pretrain", "step": step, "tokens_seen": tokens_seen,
                "train_loss": last_loss, "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
            }
            if validation_loader is not None and (step % options.validation_every == 0 or step == options.steps):
                val_loss = evaluate_lm(model, validation_loader, device, options.validation_batches)
                metric["validation_loss"] = val_loss
                metric["perplexity"] = float(torch.exp(torch.tensor(min(val_loss, 20.0))))
            append_metric(metrics_path, metric)
            print(metric, flush=True)
    metrics = {"loss": last_loss, "tokens_seen": float(tokens_seen)}
    checkpoint = save_checkpoint(output / "pretrain.pt", model, stage="pretrain", step=options.steps, metrics=metrics)
    return {**metrics, "checkpoint": str(checkpoint), "metrics": str(metrics_path), "device": str(device)}
