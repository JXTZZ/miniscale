"""Resolved pretraining recipe and run manifest."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from miniscale.integrity import atomic_write_json
from miniscale.model import MiniScaleForCausalLM
from .config import (
    PRETRAIN_IMPLEMENTATION_VERSION,
    PRETRAIN_INITIALIZATION_SCHEME,
    PRETRAIN_OPTIMIZER_GROUPING,
    PretrainOptions,
)
from ..core.checkpoint import TRAINING_CHECKPOINT_FORMAT_VERSION


def _resolved_options(options: PretrainOptions) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(options).items()
        if name not in {"allow_legacy_resume", "resume_from"}
    }


def _write_run_manifest(
    path: Path,
    *,
    model: MiniScaleForCausalLM,
    options: PretrainOptions,
    resume_signature: dict[str, object],
    train_path: str | Path,
    validation_path: str | Path | None,
    resumed_step: int,
    resolved_precision: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "stage": "pretrain",
        "checkpoint_format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "implementation_version": PRETRAIN_IMPLEMENTATION_VERSION,
        "parameter_initialization": PRETRAIN_INITIALIZATION_SCHEME,
        "optimizer_parameter_groups": PRETRAIN_OPTIMIZER_GROUPING,
        "model": asdict(model.config),
        "num_parameters": model.num_parameters,
        "training": _resolved_options(options),
        "resolved_precision": resolved_precision,
        "derived": {
            "world_size": 1,
            "global_batch_sequences": options.batch_size * options.gradient_accumulation_steps,
            "input_tokens_per_update": (
                options.batch_size * options.gradient_accumulation_steps * options.sequence_length
            ),
            "target_tokens_per_update": (
                options.batch_size
                * options.gradient_accumulation_steps
                * (options.sequence_length - 1)
            ),
            "planned_input_tokens": (
                options.steps
                * options.batch_size
                * options.gradient_accumulation_steps
                * options.sequence_length
            ),
            "planned_target_tokens": (
                options.steps
                * options.batch_size
                * options.gradient_accumulation_steps
                * (options.sequence_length - 1)
            ),
            "warmup_ratio": min(options.warmup_steps, options.steps) / options.steps,
            "tokens_per_parameter": (
                options.steps
                * options.batch_size
                * options.gradient_accumulation_steps
                * options.sequence_length
                / model.num_parameters
            ),
        },
        "inputs": {
            "train": str(Path(train_path).resolve()),
            "validation": str(Path(validation_path).resolve()) if validation_path else None,
        },
        "resume": {
            "checkpoint": str(Path(options.resume_from).resolve()) if options.resume_from else None,
            "completed_step": resumed_step,
        },
        "resume_identity": resume_signature,
    }
    atomic_write_json(path, manifest)
