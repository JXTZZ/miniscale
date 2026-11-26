"""Assistant-token validation and fixed generation probes for SFT."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from torch import Tensor
import torch

from dataset.file_utils import atomic_write_json
from model.model_miniscale import MiniScaleForCausalLM
from model.tokenizer import Tokenizer
from trainer.eval_utils import (
    evaluate_sft_generation_quality,
    generation_quality_score,
    load_generation_suite,
    SFT_GENERATION_METRICS_VERSION,
    SFT_GENERATION_PROMPTS,
)
from trainer.runtime.torch_utils import autocast_context


@torch.no_grad()
def evaluate_sft(
    model: MiniScaleForCausalLM,
    batches: Iterable[dict[str, Tensor]],
    device: torch.device,
    *,
    autocast_dtype: torch.dtype | None,
) -> tuple[float, float, int]:
    was_training = model.training
    model.eval()
    weighted_loss = 0.0
    targets = correct = 0
    try:
        for batch_index, batch in enumerate(batches):
            device_batch = {name: value.to(device) for name, value in batch.items()}
            with autocast_context(device, autocast_dtype):
                result = model(**device_batch)
            if result.loss is None or not bool(torch.isfinite(result.loss)):
                raise FloatingPointError(f"non-finite SFT validation loss at batch {batch_index}")
            labels = device_batch["labels"][:, 1:]
            mask = labels.ne(-100)
            target_count = int(mask.sum())
            if not target_count:
                continue
            predictions = result.logits[:, :-1].argmax(dim=-1)
            correct += int((predictions.eq(labels) & mask).sum())
            targets += target_count
            weighted_loss += float(result.loss) * target_count
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("SFT validation batches contain no supervised targets")
    return weighted_loss / targets, correct / targets, targets


@torch.no_grad()
def run_sft_generation_evaluation(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    output_dir: Path,
    *,
    step: int,
    device: torch.device,
    max_new_tokens: int,
    autocast_dtype: torch.dtype | None,
    suite_path: str | Path | None = None,
) -> tuple[Path, dict[str, float]]:
    probes = load_generation_suite(suite_path) if suite_path is not None else list(SFT_GENERATION_PROMPTS)
    samples, summary = evaluate_sft_generation_quality(
        model,
        tokenizer,
        probes,
        device=device,
        max_new_tokens=max_new_tokens,
        autocast_dtype=autocast_dtype,
    )
    target = output_dir / "generations" / f"step_{step:08d}.json"
    atomic_write_json(target, {
        "schema_version": 2,
        "metrics_version": SFT_GENERATION_METRICS_VERSION,
        "stage": "sft",
        "step": step,
        "suite": str(Path(suite_path).resolve()) if suite_path is not None else "builtin_smoke_v1",
        "decoding": {
            "profile": "raw_greedy",
            "temperature": 0.0,
            "top_k": None,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
        },
        "summary": summary,
        "samples": samples,
    })
    return target, summary


@dataclass(slots=True)
class SFTQuality:
    best_quality_score: float = float("-inf")
    best_quality_step: int = 0
    early_best_quality_score: float = float("-inf")
    stale_quality_evaluations: int = 0
    severe_loop_evaluations: int = 0
    recent_validation_losses: list[float] = field(default_factory=list)

    @classmethod
    def from_checkpoint(cls, state: dict[str, object]) -> SFTQuality:
        best = float(state.get("best_quality_score", float("-inf")))
        return cls(
            best_quality_score=best,
            best_quality_step=int(state.get("best_quality_step", 0)),
            early_best_quality_score=float(state.get("early_best_quality_score", best)),
            stale_quality_evaluations=int(state.get("stale_quality_evaluations", 0)),
            severe_loop_evaluations=int(state.get("severe_loop_evaluations", 0)),
            recent_validation_losses=[float(loss) for loss in state.get("recent_validation_losses", [])],
        )

    def record_loss(self, loss: float, patience: int) -> None:
        self.recent_validation_losses.append(loss)
        self.recent_validation_losses = self.recent_validation_losses[-max(patience + 1, 2):]

    def record_generation(self, summary: dict[str, float], options, step: int) -> tuple[float, bool]:
        score = generation_quality_score(summary)
        if score >= self.early_best_quality_score + options.early_stopping_quality_min_delta:
            self.early_best_quality_score = score
            self.stale_quality_evaluations = 0
        else:
            self.stale_quality_evaluations += 1
        self.severe_loop_evaluations = (
            self.severe_loop_evaluations + 1 if summary["generation_loop_rate"] > options.severe_loop_rate_threshold else 0
        )
        improved = score > self.best_quality_score
        if improved:
            self.best_quality_score = score
            self.best_quality_step = step
        return score, improved

    def stop_reason(self, options, step: int) -> str | None:
        patience = options.early_stopping_patience
        if not patience or step < options.early_stopping_min_steps:
            return None
        losses = self.recent_validation_losses
        loss_stalled = (
            len(losses) >= patience + 1
            and losses[0] - losses[-1] < options.early_stopping_validation_min_delta
        )
        severe_regression = self.severe_loop_evaluations >= 2
        if severe_regression or (self.stale_quality_evaluations >= patience and loss_stalled):
            return (
                f"quality_stale={self.stale_quality_evaluations}, "
                f"loss_stalled={loss_stalled}, severe_loop={severe_regression}"
            )
        return None
