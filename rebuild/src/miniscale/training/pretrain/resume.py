"""Pretraining input identity checks and legacy checkpoint migration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import warnings

import torch

from miniscale.integrity import path_identity, tokenizer_identity
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from .config import (
    PRETRAIN_IMPLEMENTATION_VERSION,
    PRETRAIN_INITIALIZATION_SCHEME,
    PRETRAIN_OPTIMIZER_GROUPING,
    PRETRAIN_RESUME_SIGNATURE_VERSION,
    PretrainOptions,
)
from ..core.checkpoint import signature_differences


def _migrate_legacy_single_group_optimizer(
    payload: dict[str, object],
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    """Map an old model-order AdamW state onto the current named groups."""

    saved_optimizer = payload.get("optimizer")
    if not isinstance(saved_optimizer, dict):
        raise ValueError("checkpoint optimizer state must be a mapping")
    saved_groups = saved_optimizer.get("param_groups")
    saved_state = saved_optimizer.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        raise ValueError("checkpoint optimizer state has invalid groups or state")
    if len(saved_groups) == len(optimizer.param_groups):
        return payload
    if len(saved_groups) != 1 or not isinstance(saved_groups[0], dict):
        raise ValueError(
            "legacy optimizer parameter groups cannot be migrated automatically; "
            f"checkpoint has {len(saved_groups)} groups, expected 1"
        )

    saved_ids = saved_groups[0].get("params")
    model_parameters = list(model.parameters())
    if not isinstance(saved_ids, list) or len(saved_ids) != len(model_parameters):
        raise ValueError("legacy optimizer parameter order does not match the current model")
    saved_id_by_parameter = {
        id(parameter): saved_id for parameter, saved_id in zip(model_parameters, saved_ids, strict=True)
    }

    current_optimizer = optimizer.state_dict()
    current_groups = current_optimizer["param_groups"]
    migrated_state: dict[object, object] = {}
    migrated_groups: list[dict[str, object]] = []
    source_group = saved_groups[0]
    for live_group, serialized_group in zip(optimizer.param_groups, current_groups, strict=True):
        current_ids = serialized_group["params"]
        for parameter, current_id in zip(live_group["params"], current_ids, strict=True):
            saved_id = saved_id_by_parameter[id(parameter)]
            if saved_id in saved_state:
                migrated_state[current_id] = saved_state[saved_id]
        migrated_group = dict(serialized_group)
        for name, value in source_group.items():
            if name not in {"params", "weight_decay", "group_name"}:
                migrated_group[name] = value
        migrated_groups.append(migrated_group)

    scheduler_state = payload.get("scheduler")
    migrated_scheduler = dict(scheduler_state) if isinstance(scheduler_state, dict) else scheduler_state
    if isinstance(migrated_scheduler, dict):
        group_count = len(migrated_groups)
        for name in ("base_lrs", "_last_lr", "lr_lambdas"):
            value = migrated_scheduler.get(name)
            if isinstance(value, list) and len(value) == 1:
                migrated_scheduler[name] = value * group_count

    migrated = dict(payload)
    migrated["optimizer"] = {"state": migrated_state, "param_groups": migrated_groups}
    migrated["scheduler"] = migrated_scheduler
    warnings.warn(
        "migrated legacy single-group AdamW state to decay/no-decay parameter groups",
        RuntimeWarning,
        stacklevel=2,
    )
    return migrated


def _resume_signature(
    options: PretrainOptions,
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    train_path: str | Path,
    validation_path: str | Path | None,
    resolved_precision: str,
) -> dict[str, object]:
    train_identity = path_identity(train_path)
    validation_identity: dict[str, object]
    if validation_path is None:
        validation_identity = {
            "mode": "content_hash_split",
            "fraction": options.validation_fraction,
            "source": train_identity,
        }
    else:
        dedicated_identity = path_identity(validation_path)
        if dedicated_identity == train_identity:
            raise ValueError("dedicated validation data is identical to training data")
        validation_identity = {"mode": "dedicated_file", "source": dedicated_identity}
    return {
        "signature_version": PRETRAIN_RESUME_SIGNATURE_VERSION,
        "implementation_version": PRETRAIN_IMPLEMENTATION_VERSION,
        "total_steps": options.steps,
        "batch_size": options.batch_size,
        "sequence_length": options.sequence_length,
        "gradient_accumulation_steps": options.gradient_accumulation_steps,
        "learning_rate": options.learning_rate,
        "min_learning_rate": options.min_learning_rate,
        "warmup_steps": options.warmup_steps,
        "weight_decay": options.weight_decay,
        "adam_beta1": options.adam_beta1,
        "adam_beta2": options.adam_beta2,
        "adam_eps": options.adam_eps,
        "grad_clip": options.grad_clip,
        "shuffle_buffer_size": options.shuffle_buffer_size,
        "num_workers": options.num_workers,
        "validation_fraction": options.validation_fraction,
        "validation_every": options.validation_every,
        "validation_batches": options.validation_batches,
        "validation_sampling": "fixed_reservoir_v1",
        "seed": options.seed,
        "precision": resolved_precision,
        "parameter_initialization": PRETRAIN_INITIALIZATION_SCHEME,
        "optimizer_parameter_groups": PRETRAIN_OPTIMIZER_GROUPING,
        "world_size": 1,
        "model": asdict(model.config),
        "tokenizer": tokenizer_identity(tokenizer),
        "train_data": train_identity,
        "validation_data": validation_identity,
    }


def _validate_resume_signature(
    saved: object,
    current: dict[str, object],
    *,
    allow_legacy: bool = False,
) -> None:
    if not isinstance(saved, dict):
        raise ValueError("checkpoint resume_signature must be a mapping")
    if saved.get("signature_version") != PRETRAIN_RESUME_SIGNATURE_VERSION:
        if not allow_legacy:
            raise ValueError(
                "checkpoint predates strict resume identity checks; pass "
                "--allow-legacy-resume once to accept the documented migration risk"
            )
        warnings.warn(
            "resuming a legacy checkpoint without verified data/tokenizer/model identity; "
            "the next checkpoint will be upgraded to the current format",
            RuntimeWarning,
            stacklevel=2,
        )
        comparable = {
            name: value
            for name, value in saved.items()
            if name in current and name not in {"signature_version", "implementation_version"}
        }
        mismatches = signature_differences(comparable, {name: current[name] for name in comparable})
    else:
        mismatches = signature_differences(saved, current)
    if mismatches:
        raise ValueError(f"resume options do not match checkpoint: {mismatches}")
