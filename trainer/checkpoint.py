"""Save, load, and restore MiniScale checkpoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import pickle
import random
import warnings

import numpy as np
import torch

from dataset.file_utils import atomic_output_path, path_identity
from model.model_miniscale import MiniScaleConfig, MiniScaleForCausalLM


TRAINING_CHECKPOINT_FORMAT_VERSION = 2
PRETRAIN_RESUME_SIGNATURE_VERSION = 2


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) == ("miniscale.config", "MiniScaleConfig"):
            return MiniScaleConfig
        return super().find_class(module, name)


class _CheckpointPickle:
    Unpickler = _CheckpointUnpickler
    load = staticmethod(pickle.load)


def read_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, object]:
    """Read a trusted MiniScale payload, including the legacy config class."""

    return torch.load(path, map_location=device, weights_only=False, pickle_module=_CheckpointPickle)


def save_checkpoint(
    path: str | Path,
    model: MiniScaleForCausalLM,
    *,
    stage: str,
    step: int,
    metrics: dict[str, float],
) -> Path:
    target = Path(path)
    with atomic_output_path(target) as temporary:
        torch.save(
            {
                "stage": stage,
                "step": step,
                "metrics": metrics,
                "config": asdict(model.config),
                "model": model.state_dict(),
            },
            temporary,
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
    payload: dict[str, object] = {
        "format_version": TRAINING_CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "step": step,
        "metrics": metrics,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_state": training_state,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    for name in ("tokens_seen", "best_val_loss"):
        if name in training_state:
            payload[name] = training_state[name]
    with atomic_output_path(target) as temporary:
        torch.save(payload, temporary)
    return target


def read_training_checkpoint(
    path: str | Path,
    device: str | torch.device,
) -> dict[str, object]:
    """Read and structurally validate a resumable checkpoint without mutation."""

    payload = read_checkpoint(path, device)
    required = {"model", "optimizer", "scheduler", "training_state", "step"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint cannot resume training; missing: {', '.join(sorted(missing))}")
    version = payload.get("format_version", 1)
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"invalid checkpoint format_version: {version!r}")
    if version > TRAINING_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {version} is newer than supported format "
            f"{TRAINING_CHECKPOINT_FORMAT_VERSION}"
        )
    return payload


def restore_training_checkpoint(
    payload: dict[str, object],
    model: MiniScaleForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    restore_rng: bool = True,
) -> None:
    """Restore a payload after the caller has validated run compatibility."""

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        restore_rng_state(payload.get("rng_state"))


def restore_rng_state(rng_state: object) -> None:
    """Restore Python, NumPy, CPU and CUDA RNG state captured in a checkpoint."""

    if isinstance(rng_state, dict):
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("torch") is not None:
            torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            cuda_states = [state.cpu() for state in rng_state["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> MiniScaleForCausalLM:
    payload = read_checkpoint(path, device)
    config = payload["config"]
    if isinstance(config, dict):
        config = MiniScaleConfig(**config)
    model = MiniScaleForCausalLM(config)
    model.load_state_dict(payload["model"])
    return model.to(device)


def signature_differences(
    saved: object,
    current: object,
    prefix: str = "",
) -> dict[str, tuple[object, object]]:
    if isinstance(saved, dict) and isinstance(current, dict):
        differences: dict[str, tuple[object, object]] = {}
        for name in sorted(set(saved) | set(current)):
            path = f"{prefix}.{name}" if prefix else str(name)
            if name not in saved:
                differences[path] = ("<missing>", current[name])
            elif name not in current:
                differences[path] = (saved[name], "<missing>")
            else:
                differences.update(signature_differences(saved[name], current[name], path))
        return differences
    return {} if saved == current else {prefix: (saved, current)}


def migrate_legacy_single_group_optimizer(
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


def validate_pretrain_signature(
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



def read_strict_checkpoint(path, device, stage: str, label: str):
    if path is None:
        return None, {}
    payload = read_training_checkpoint(path, device)
    if payload.get("stage") != stage:
        raise ValueError(f"--resume requires a full {label} checkpoint")
    state = payload["training_state"]
    if not isinstance(state, dict) or not isinstance(state.get("resume_signature"), dict):
        raise ValueError(f"{label} checkpoint does not contain a strict resume signature")
    return payload, state


def check_resume_signature(saved, current, label: str) -> None:
    if saved is not None:
        differences = signature_differences(saved, current)
        if differences:
            raise ValueError(f"{label} resume options do not match checkpoint: {differences}")


def load_reference(path, stage, identity, device, model, label):
    if identity is None or path_identity(path) != identity:
        raise ValueError(f"{label} frozen reference snapshot does not match the resume checkpoint")
    payload = read_checkpoint(path, device)
    if payload.get("stage") != f"{stage}_reference":
        raise ValueError(f"{label} reference snapshot has an invalid stage")
    model.load_state_dict(payload["model"])


def resume_pretrain(model, optimizer, scheduler, signature, options, device):
    if options.resume_from is None:
        return None
    payload = read_training_checkpoint(options.resume_from, device)
    if payload.get("stage") != "pretrain":
        raise ValueError("--resume must point to a full pretrain checkpoint")
    state = payload["training_state"]
    if not isinstance(state, dict):
        raise ValueError("checkpoint training_state must be a mapping")
    saved_signature = state.get("resume_signature")
    validate_pretrain_signature(saved_signature, signature, allow_legacy=options.allow_legacy_resume)
    if saved_signature.get("signature_version") != PRETRAIN_RESUME_SIGNATURE_VERSION:
        payload = migrate_legacy_single_group_optimizer(payload, model, optimizer)
    restore_training_checkpoint(payload, model, optimizer, scheduler, restore_rng=False)
    return payload


def freeze_reference(model: MiniScaleForCausalLM, device: torch.device) -> MiniScaleForCausalLM:
    return deepcopy(model).to(device).eval().requires_grad_(False)
