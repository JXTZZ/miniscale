"""Backward-compatible exports for the split training infrastructure modules."""

from miniscale.integrity import atomic_write_json

from .artifacts import append_metric, mirror_checkpoint, prune_periodic_checkpoints, truncate_metrics_after
from .checkpoint import (
    TRAINING_CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    load_training_checkpoint,
    read_training_checkpoint,
    restore_rng_state,
    restore_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    signature_differences,
)
from .runtime import (
    autocast_context,
    build_adamw_optimizer,
    build_warmup_cosine_scheduler,
    evaluate_lm,
    infinite_batches,
    resolve_autocast_dtype,
    resolve_device,
    seed_everything,
    seed_worker,
    warmup_cosine_multiplier,
)

__all__ = [
    "TRAINING_CHECKPOINT_FORMAT_VERSION",
    "append_metric",
    "atomic_write_json",
    "autocast_context",
    "build_adamw_optimizer",
    "build_warmup_cosine_scheduler",
    "evaluate_lm",
    "infinite_batches",
    "load_checkpoint",
    "load_training_checkpoint",
    "mirror_checkpoint",
    "prune_periodic_checkpoints",
    "read_training_checkpoint",
    "resolve_autocast_dtype",
    "resolve_device",
    "restore_rng_state",
    "restore_training_checkpoint",
    "save_checkpoint",
    "save_training_checkpoint",
    "seed_everything",
    "seed_worker",
    "signature_differences",
    "truncate_metrics_after",
    "warmup_cosine_multiplier",
]
