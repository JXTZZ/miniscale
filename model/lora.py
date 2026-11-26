"""Low-rank adapters for the query and value projections."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from model.model_miniscale import MiniScaleForCausalLM


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(self.weight.new_empty((rank, base.in_features)))
        self.lora_b = nn.Parameter(self.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        return F.linear(inputs, self.weight, self.bias) + self.scale * F.linear(
            F.linear(inputs, self.lora_a), self.lora_b
        )


def enable_lora(model: MiniScaleForCausalLM, rank: int, alpha: float) -> None:
    config = {"rank": rank, "alpha": alpha}
    if model.lora_config is not None:
        if model.lora_config != config:
            raise ValueError("LoRA options do not match the model's existing adapters")
        return
    projections = [(layer.attention.query, layer.attention.value) for layer in model.layers]
    if rank < 1 or any(
        rank > min(linear.in_features, linear.out_features) for pair in projections for linear in pair
    ):
        raise ValueError("LoRA rank must fit both query and value projections")
    model.requires_grad_(False)
    for decoder, (query, value) in zip(model.layers, projections, strict=True):
        decoder.attention.query = LoRALinear(query, rank, alpha)
        decoder.attention.value = LoRALinear(value, rank, alpha)
    model.lora_config = config


def merge_lora_weights(state: dict[str, Tensor], config: dict[str, float | int]) -> dict[str, Tensor]:
    merged = dict(state)
    scale = config["alpha"] / config["rank"]
    for name in state:
        if name.endswith(".lora_a"):
            prefix = name.removesuffix("lora_a")
            a = merged.pop(name)
            b = merged.pop(prefix + "lora_b")
            weight = merged[prefix + "weight"]
            delta = b.float() @ a.float()
            merged[prefix + "weight"] = (weight.float() + scale * delta).to(weight.dtype)
    return merged
