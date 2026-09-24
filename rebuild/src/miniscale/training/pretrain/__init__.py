"""Pretraining configuration, evaluation, resume checks, and loops."""

from .config import PretrainOptions, SmokePretrainOptions, pretrain_option_default
from .resume import _validate_resume_signature
from .runner import (
    build_pretrain_optimizer,
    build_warmup_cosine_scheduler,
    resolve_autocast_dtype,
    run_pretrain,
    run_pretrain_jsonl,
)
