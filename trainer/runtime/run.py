"""Output, tracking, and resume operations shared by the training stages."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from dataset.file_utils import atomic_output_path, atomic_write_text
from trainer.checkpoint import capture_rng_state, restore_rng_state, save_training_checkpoint
from trainer.runtime.tracking import WandbTracker
from trainer.runtime.torch_utils import infinite_batches


class RunArtifacts:
    def __init__(self, directory: str | Path, stage: str) -> None:
        self.output = Path(directory)
        self.stage = stage
        self.metrics = self.output / f"{stage}_metrics.jsonl"
        self.manifest = self.output / f"{stage}_run.json"
        self.checkpoints = self.output / "checkpoints"
        self.last_checkpoint: Path | None = None

    def check_empty(self, resume_from: str | Path | None, label: str, *names: str) -> None:
        if resume_from is not None:
            self.last_checkpoint = Path(resume_from)
            return
        files = (self.output / name for name in names)
        existing = next((path for path in files if path.exists()), None)
        if existing is None:
            existing = next(self.checkpoints.glob("*.pt"), None)
        if existing is not None:
            raise FileExistsError(
                f"output already contains {label} artifacts ({existing}); "
                "choose a new --output or use --resume"
            )

    def checkpoint_writer(self, model, optimizer, scheduler, training_state):
        def save(path: Path, step: int, metrics: dict[str, float]) -> Path:
            checkpoint = save_training_checkpoint(
                path, model, optimizer, scheduler, stage=self.stage, step=step,
                metrics=metrics, training_state=training_state(),
            )
            self.last_checkpoint = checkpoint
            return checkpoint
        return save

    def save_periodic(self, save, step: int, metrics, options, *, label: str | None = None) -> None:
        if options.save_every and step % options.save_every == 0:
            checkpoint = save(self.checkpoints / f"step_{step:08d}.pt", step, metrics)
            prune_periodic_checkpoints(self.checkpoints, options.keep_last_checkpoints)
            if label is not None:
                prefix = f"{label} " if label else ""
                print(f"saved {prefix}checkpoint: {checkpoint}", flush=True)

    def log(self, metric: dict[str, object], tracker: WandbTracker | None, generation: Path | None = None) -> None:
        append_metric(self.metrics, metric)
        if tracker is not None:
            tracker.log(metric, generation_path=generation)
        print(format_training_metric(metric), flush=True)


def start_tracking(options, output: Path, step: int, saved_run_id: str | None, config: dict[str, object]):
    tracker = WandbTracker.start(
        enabled=options.wandb_enabled, project=options.wandb_project,
        entity=options.wandb_entity, name=options.wandb_run_name,
        run_id=options.wandb_run_id or saved_run_id, mode=options.wandb_mode,
        config=config, directory=output, retry_every_steps=options.wandb_retry_every_steps,
        initial_step=step if options.resume_from is not None else None,
    )
    return tracker, tracker.run_id if tracker is not None else saved_run_id


def check_resume_progress(step: int, total_steps: int, run_id: str | None, saved_run_id: str | None) -> None:
    if step >= total_steps:
        raise ValueError("resume checkpoint is already at or beyond the requested total steps")
    if run_id and saved_run_id and run_id != saved_run_id:
        raise ValueError("--wandb-run-id does not match the run id stored in the checkpoint")


def restore_batches(loader, consumed: int, payload: dict[str, object] | None):
    batches = iter(infinite_batches(loader))
    for _ in range(consumed):
        next(batches)
    # Iterator construction and replay can consume RNG before training resumes.
    if payload is not None:
        restore_rng_state(payload.get("rng_state"))
    return batches


def start_update(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    return time.perf_counter()


def update_duration(started: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started


class TrainingStep:
    """Allow emergency saves before mutation or after a complete training step."""

    def __init__(self) -> None:
        self._rng: dict[str, object] | None = None
        self._updating = False

    def begin(self) -> None:
        self._rng = capture_rng_state()

    def updating(self) -> None:
        self._updating = True
        self._rng = None

    def complete(self) -> None:
        self._updating = False

    def save_emergency(self, artifacts, save, step, metrics, tracker, label: str = "training") -> None:
        if self._updating:
            fallback = artifacts.last_checkpoint
            recovery = f"resume from {fallback}" if fallback else "no previous checkpoint is available"
            print(f"{label} interrupted during an incomplete step. {recovery}", flush=True)
        else:
            restore_rng_state(self._rng)
            checkpoint = save(artifacts.checkpoints / f"emergency_step_{step:08d}.pt", step, metrics)
            print(f"{label} interrupted. Emergency checkpoint saved to {checkpoint}", flush=True)
        if tracker is not None:
            tracker.finish(exit_code=1, summary={"interrupted_step": step})


def training_result(artifacts, checkpoint, metrics, device, tracker, run_id, completed_step):
    if tracker is not None:
        tracker.finish(summary={**metrics, "final_step": completed_step})
    result = {
        **metrics, "checkpoint": str(checkpoint), "metrics": str(artifacts.metrics),
        "manifest": str(artifacts.manifest), "device": str(device),
    }
    if run_id is not None:
        result["wandb_run_id"] = run_id
    return result


def should_log(step: int, options, validation_due: bool, generation_due: bool = False) -> bool:
    return step in (1, options.steps) or step % options.log_every == 0 or validation_due or generation_due


def append_metric(path: str | Path, metric: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metric, ensure_ascii=False) + "\n")


def truncate_metrics_after(path: str | Path, step: int) -> None:
    target = Path(path)
    if not target.exists():
        return
    retained: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            metric = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(metric.get("step", -1)) <= step:
            retained.append(json.dumps(metric, ensure_ascii=False))
    atomic_write_text(target, "".join(f"{line}\n" for line in retained))


def prune_periodic_checkpoints(path: str | Path, keep_last: int) -> None:
    checkpoint_dir = Path(path)
    for checkpoint in sorted(checkpoint_dir.glob("step_*.pt"))[:-keep_last]:
        checkpoint.unlink()


def mirror_checkpoint(source: str | Path, destination: str | Path) -> Path:
    """Atomically expose a second checkpoint name without duplicating disk use when possible."""

    source_path = Path(source)
    target = Path(destination)
    with atomic_output_path(target) as temporary:
        temporary.unlink()
        try:
            os.link(source_path, temporary)
        except OSError:
            shutil.copyfile(source_path, temporary)
    return target


def _number(metric: Mapping[str, object], name: str) -> float | None:
    value = metric.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _rate(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.1f}"


def _append_number(
    fields: list[str],
    metric: Mapping[str, object],
    name: str,
    label: str,
    format_spec: str,
) -> None:
    value = _number(metric, name)
    if value is not None:
        fields.append(f"{label} {value:{format_spec}}")


def format_training_metric(metric: Mapping[str, object]) -> str:
    """Return a compact terminal line without changing the persisted metric.

    Every stage shares the same loss/LR/gradient/throughput/memory spine.
    Preference and RL stages append only their most useful live signals.
    Unknown or unavailable values are omitted instead of rendering ``None``.
    """

    stage = str(metric.get("stage", "train"))
    step = metric.get("step", "?")
    fields = [f"[{stage}] step {step}"]

    loss = _number(metric, "train_loss")
    if loss is None:
        loss = _number(metric, "loss")
    if loss is not None:
        fields.append(f"loss {loss:.3f}")

    learning_rate = _number(metric, "learning_rate")
    if learning_rate is not None:
        fields.append(f"lr {_scientific(learning_rate)}")

    grad_norm = _number(metric, "grad_norm")
    if grad_norm is not None:
        clipped = "*" if metric.get("grad_was_clipped") is True else ""
        fields.append(f"grad {grad_norm:.2f}{clipped}")

    tokens_per_second = _number(metric, "tokens_per_second")
    if tokens_per_second is not None:
        fields.append(f"tok/s {_rate(tokens_per_second)}")

    memory = _number(metric, "cuda_peak_memory_mb")
    if memory is not None:
        fields.append(f"mem {memory:.0f}MB")

    if stage == "dpo":
        _append_number(fields, metric, "preference_accuracy", "acc", ".1%")
    elif stage == "grpo":
        _append_number(fields, metric, "reward_mean", "reward", ".3f")
        _append_number(fields, metric, "kl", "kl", ".3f")
    elif stage == "agent_rl":
        _append_number(fields, metric, "reward_mean", "reward", ".3f")
        _append_number(fields, metric, "success_rate", "success", ".1%")

    if tokens_per_second is None:
        rollouts_per_second = _number(metric, "rollouts_per_second")
        if rollouts_per_second is not None:
            fields.append(f"roll/s {_rate(rollouts_per_second)}")

    validation_loss = _number(metric, "validation_loss")
    if validation_loss is not None and math.isfinite(validation_loss):
        fields.append(f"val {validation_loss:.3f}")
    elif stage in {"grpo", "agent_rl"}:
        _append_number(fields, metric, "validation_reward", "val_reward", ".3f")

    return " | ".join(fields)
