from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random

import torch
from torch import Tensor
import torch.nn.functional as F

from miniscale.data import collate_lm_batch, load_jsonl_rows
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from .common import append_metric, resolve_device, save_checkpoint, seed_everything


@dataclass(slots=True)
class DPOOptions:
    steps: int = 1000
    batch_size: int = 1
    learning_rate: float = 5e-6
    beta: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    data_limit: int | None = None
    seed: int = 42
    device: str = "auto"


def _encoded_batch(messages: list[list[dict[str, object]]], tokenizer: Tokenizer, max_length: int) -> dict[str, Tensor]:
    examples = []
    for conversation in messages:
        input_ids, labels = tokenizer.encode_sft(conversation)
        input_ids, labels = input_ids[-max_length:], labels[-max_length:]
        examples.append({"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)})
    return collate_lm_batch(examples, tokenizer.pad_token_id)


def completion_log_probability(model: MiniScaleForCausalLM, batch: dict[str, Tensor]) -> Tensor:
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    targets = labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(-1)


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(logits).mean(), (logits > 0).float().mean()


def run_dpo_jsonl(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    data_path: str | Path,
    output_dir: str | Path,
    options: DPOOptions | None = None,
) -> dict[str, float | str]:
    options = options or DPOOptions()
    seed_everything(options.seed)
    device = resolve_device(options.device)
    rows = load_jsonl_rows(data_path, options.data_limit)
    pairs = [(row.get("chosen"), row.get("rejected")) for row in rows]
    pairs = [(chosen, rejected) for chosen, rejected in pairs if isinstance(chosen, list) and isinstance(rejected, list)]
    if not pairs:
        raise ValueError("DPO dataset has no valid chosen/rejected pairs")
    random.Random(options.seed).shuffle(pairs)
    model.to(device).train()
    reference = deepcopy(model).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate)
    output = Path(output_dir)
    metrics_path = output / "dpo_metrics.jsonl"
    last_loss = last_accuracy = 0.0
    cursor = 0
    for step in range(1, options.steps + 1):
        selected = [pairs[(cursor + index) % len(pairs)] for index in range(options.batch_size)]
        cursor = (cursor + options.batch_size) % len(pairs)
        chosen = _encoded_batch([item[0] for item in selected], tokenizer, model.config.max_position_embeddings)
        rejected = _encoded_batch([item[1] for item in selected], tokenizer, model.config.max_position_embeddings)
        chosen = {name: tensor.to(device) for name, tensor in chosen.items()}
        rejected = {name: tensor.to(device) for name, tensor in rejected.items()}
        with torch.no_grad():
            ref_chosen = completion_log_probability(reference, chosen)
            ref_rejected = completion_log_probability(reference, rejected)
        policy_chosen = completion_log_probability(model, chosen)
        policy_rejected = completion_log_probability(model, rejected)
        loss, accuracy = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, options.beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        optimizer.step()
        last_loss, last_accuracy = float(loss.detach()), float(accuracy)
        if step == 1 or step % options.log_every == 0 or step == options.steps:
            metric = {"stage": "dpo", "step": step, "loss": last_loss, "preference_accuracy": last_accuracy, "grad_norm": float(grad_norm)}
            append_metric(metrics_path, metric)
            print(metric, flush=True)
    metrics = {"loss": last_loss, "preference_accuracy": last_accuracy}
    checkpoint = save_checkpoint(output / "dpo.pt", model, stage="dpo", step=options.steps, metrics=metrics)
    return {**metrics, "checkpoint": str(checkpoint), "metrics": str(metrics_path), "device": str(device)}
