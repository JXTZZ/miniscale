from __future__ import annotations

import json
from pathlib import Path
import secrets
from typing import Any
import warnings


class WandbTracker:
    """Fault-tolerant W&B adapter backed by a persistent local upload queue."""

    def __init__(
        self,
        wandb: Any,
        run: Any | None,
        *,
        run_id: str | None = None,
        init_kwargs: dict[str, object] | None = None,
        pending_path: str | Path | None = None,
        retry_every_steps: int = 200,
        initial_step: int = 0,
    ) -> None:
        self._wandb = wandb
        self._run = run
        self._run_id = str(run.id) if run is not None else (run_id or secrets.token_hex(4))
        self._init_kwargs = dict(init_kwargs or {})
        self._pending_path = Path(pending_path) if pending_path is not None else None
        self._retry_every_steps = retry_every_steps
        self._next_retry_step = initial_step
        self._connection_attempted = run is not None

    @property
    def run_id(self) -> str:
        return self._run_id

    @classmethod
    def start(
        cls,
        *,
        enabled: bool,
        project: str,
        entity: str | None,
        name: str | None,
        run_id: str | None,
        mode: str,
        config: dict[str, object],
        directory: str | Path,
        retry_every_steps: int = 200,
        initial_step: int = 0,
    ) -> WandbTracker | None:
        if not enabled:
            return None
        if retry_every_steps < 1:
            raise ValueError("retry_every_steps must be positive")
        try:
            import wandb
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "W&B tracking requires the optional dependency; run `uv sync --extra tracking`"
            ) from error

        output = Path(directory).resolve()
        stable_run_id = run_id or secrets.token_hex(4)
        tracker = cls(
            wandb,
            None,
            run_id=stable_run_id,
            init_kwargs={
                "project": project,
                "entity": entity,
                "name": name,
                "id": stable_run_id,
                "resume": "allow",
                "mode": mode,
                "config": config,
                "dir": str(output),
                "save_code": True,
            },
            pending_path=output / "wandb_pending.jsonl",
            retry_every_steps=retry_every_steps,
            initial_step=initial_step,
        )
        tracker._connect(initial_step)
        return tracker

    def log(
        self,
        metric: dict[str, object],
        *,
        generation_path: str | Path | None = None,
    ) -> None:
        step = int(metric["step"])
        self._append_pending({"kind": "metric", "metric": metric})
        if generation_path is not None:
            self._append_pending({
                "kind": "generation",
                "step": step,
                "path": str(Path(generation_path).resolve()),
            })

        if self._run is not None:
            self._flush_pending(step)
        elif step >= self._next_retry_step:
            self._connect(step)

    def finish(self, *, exit_code: int = 0, summary: dict[str, object] | None = None) -> None:
        # Give queued events one final upload attempt. Failure is deliberately
        # non-fatal: the queue survives for the next resumed process.
        if self._run is None and self._read_pending():
            self._connect(self._next_retry_step)
        if self._run is None:
            return
        try:
            if summary:
                self._run.summary.update(summary)
            self._run.finish(exit_code=exit_code)
        except Exception as error:
            warnings.warn(
                f"W&B finalization failed; training artifacts and pending uploads are still valid: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            self._run = None

    def _connect(self, step: int) -> None:
        kwargs = dict(self._init_kwargs)
        if self._connection_attempted:
            kwargs["reinit"] = "finish_previous"
        self._connection_attempted = True
        try:
            run = self._wandb.init(**kwargs)
            if run is None:
                raise RuntimeError("wandb.init returned no run")
            self._run = run
            self._run_id = str(run.id)
            self._flush_pending(step)
        except Exception as error:
            self._run = None
            self._schedule_retry(step)
            warnings.warn(
                f"W&B connection failed; training will continue and retry at step "
                f"{self._next_retry_step}. Pending events remain local: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def _flush_pending(self, step: int) -> None:
        if self._run is None:
            return
        pending = self._read_pending()
        if not pending:
            return

        # Upload scalars before media. A slow or failing Table upload must not
        # prevent loss/LR history from being backfilled.
        ordered = [record for record in pending if record.get("kind") == "metric"]
        ordered.extend(record for record in pending if record.get("kind") != "metric")
        for index, record in enumerate(ordered):
            try:
                self._send_record(record)
            except Exception as error:
                self._write_pending(ordered[index:])
                self._run = None
                self._schedule_retry(step)
                warnings.warn(
                    f"W&B upload failed; training and local checkpointing will continue. "
                    f"Queued events will be retried at step {self._next_retry_step}: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
        self._write_pending([])

    def _send_record(self, record: dict[str, object]) -> None:
        if self._run is None:
            raise RuntimeError("W&B run is not connected")
        kind = record.get("kind")
        if kind == "metric":
            metric = record.get("metric")
            if not isinstance(metric, dict):
                raise ValueError("pending W&B metric is malformed")
            self._run.log(self._metric_values(metric), step=int(metric["step"]))
            return
        if kind == "generation":
            path = record.get("path")
            if not isinstance(path, str):
                raise ValueError("pending W&B generation path is malformed")
            generation = json.loads(Path(path).read_text(encoding="utf-8"))
            if generation.get("stage") == "dpo":
                columns = [
                    "step", "language", "name", "prompt", "policy_response",
                    "reference_response", "policy_generated_tokens", "reference_generated_tokens",
                ]
            else:
                columns = ["step", "language", "name", "prompt", "response", "generated_tokens"]
            table = self._wandb.Table(
                columns=columns,
                data=[
                    [generation["step"], *[sample[column] for column in columns[1:]]]
                    for sample in generation["samples"]
                ],
            )
            # Do not attach old media to an explicit historic W&B step. The
            # original training step remains a column in the table.
            self._run.log({"eval/generations": table})
            return
        raise ValueError(f"unknown pending W&B record kind: {kind!r}")

    @staticmethod
    def _metric_values(metric: dict[str, object]) -> dict[str, object]:
        values: dict[str, object] = {
            "train/loss": metric["train_loss"],
            "train/learning_rate": metric["learning_rate"],
            "train/grad_norm": metric["grad_norm"],
            "train/tokens_seen": metric["tokens_seen"],
        }
        optional_metrics = {
            "examples_seen": "train/examples_seen",
            "pairs_seen": "train/pairs_seen",
            "target_tokens_seen": "train/target_tokens_seen",
            "grad_was_clipped": "train/grad_was_clipped",
            "update_seconds": "performance/update_seconds",
            "tokens_per_second": "performance/tokens_per_second",
            "supervised_tokens_per_second": "performance/supervised_tokens_per_second",
            "samples_per_second": "performance/samples_per_second",
            "pairs_per_second": "performance/pairs_per_second",
            "cuda_peak_memory_mb": "performance/cuda_peak_memory_mb",
            "preference_accuracy": "train/preference_accuracy",
            "reward_accuracy": "train/reward_accuracy",
            "reward_margin": "train/reward_margin",
            "chosen_reward": "train/chosen_reward",
            "rejected_reward": "train/rejected_reward",
            "policy_chosen_logp": "train/policy_chosen_logp",
            "policy_rejected_logp": "train/policy_rejected_logp",
            "policy_logp_margin": "train/policy_logp_margin",
            "reference_logp_margin": "train/reference_logp_margin",
            "policy_preference_accuracy": "train/policy_preference_accuracy",
            "reward_mean": "train/reward_mean",
            "reward_std": "train/reward_std",
            "reward_max": "train/reward_max",
            "policy_loss": "train/policy_loss",
            "kl": "train/kl",
            "clip_fraction": "train/clip_fraction",
            "zero_advantage_group_rate": "train/zero_advantage_group_rate",
            "prompts_seen": "train/prompts_seen",
            "rollouts_seen": "train/rollouts_seen",
            "action_tokens_seen": "train/action_tokens_seen",
            "optimizer_steps": "train/optimizer_steps",
            "success_rate": "train/success_rate",
            "tool_call_rate": "train/tool_call_rate",
            "invalid_call_rate": "train/invalid_call_rate",
            "mean_turns": "train/mean_turns",
            "rollouts_per_second": "performance/rollouts_per_second",
        }
        for source, target in optional_metrics.items():
            if source in metric:
                values[target] = metric[source]
        if "validation_loss" in metric:
            values["eval/loss"] = metric["validation_loss"]
            values["eval/best_loss"] = metric["best_val_loss"]
            if "perplexity" in metric:
                values["eval/perplexity"] = metric["perplexity"]
            if "validation_token_accuracy" in metric:
                values["eval/token_accuracy"] = metric["validation_token_accuracy"]
            if "validation_target_tokens" in metric:
                values["eval/target_tokens"] = metric["validation_target_tokens"]
            dpo_validation_metrics = {
                "validation_reward_accuracy": "eval/reward_accuracy",
                "validation_reward_margin": "eval/reward_margin",
                "validation_chosen_reward": "eval/chosen_reward",
                "validation_rejected_reward": "eval/rejected_reward",
                "validation_policy_logp_margin": "eval/policy_logp_margin",
                "validation_reference_logp_margin": "eval/reference_logp_margin",
                "validation_policy_preference_accuracy": "eval/policy_preference_accuracy",
                "validation_pairs": "eval/pairs",
                "validation_chosen_tokens": "eval/chosen_tokens",
                "validation_rejected_tokens": "eval/rejected_tokens",
            }
            for source, target in dpo_validation_metrics.items():
                if source in metric:
                    values[target] = metric[source]
        rl_validation_metrics = {
            "validation_reward": "eval/reward",
            "validation_exact_match": "eval/exact_match",
            "validation_success_rate": "eval/success_rate",
            "validation_tool_call_rate": "eval/tool_call_rate",
            "validation_invalid_call_rate": "eval/invalid_call_rate",
            "validation_mean_turns": "eval/mean_turns",
            "best_validation_reward": "eval/best_reward",
        }
        for source, target in rl_validation_metrics.items():
            if source in metric:
                values[target] = metric[source]
        return values

    def _schedule_retry(self, step: int) -> None:
        self._next_retry_step = step + self._retry_every_steps

    def _append_pending(self, record: dict[str, object]) -> None:
        if self._pending_path is None:
            # Direct construction is useful for small integrations/tests. Keep
            # the same semantics with a transient in-memory queue.
            current = getattr(self, "_memory_pending", [])
            current.append(record)
            self._memory_pending = current
            return
        self._pending_path.parent.mkdir(parents=True, exist_ok=True)
        with self._pending_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def _read_pending(self) -> list[dict[str, object]]:
        if self._pending_path is None:
            return list(getattr(self, "_memory_pending", []))
        if not self._pending_path.exists():
            return []
        records: list[dict[str, object]] = []
        for line in self._pending_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("pending W&B record must be a JSON object")
            records.append(value)
        return records

    def _write_pending(self, records: list[dict[str, object]]) -> None:
        if self._pending_path is None:
            self._memory_pending = records
            return
        if not records:
            self._pending_path.unlink(missing_ok=True)
            return
        temporary = self._pending_path.with_suffix(self._pending_path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(self._pending_path)
