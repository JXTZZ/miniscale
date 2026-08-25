from __future__ import annotations

from pathlib import Path
import json
from typing import Any


class WandbTracker:
    """Small optional W&B adapter; importing MiniScale does not require wandb."""

    def __init__(self, wandb: Any, run: Any) -> None:
        self._wandb = wandb
        self._run = run

    @property
    def run_id(self) -> str:
        return str(self._run.id)

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
    ) -> WandbTracker | None:
        if not enabled:
            return None
        try:
            import wandb
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "W&B tracking requires the optional dependency; run `uv sync --extra tracking`"
            ) from error

        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            id=run_id,
            resume="allow" if run_id else None,
            mode=mode,
            config=config,
            dir=str(Path(directory).resolve()),
            save_code=True,
        )
        if run is None:
            raise RuntimeError("wandb.init() did not return a run")
        return cls(wandb, run)

    def log(
        self,
        metric: dict[str, object],
        *,
        generation_path: str | Path | None = None,
    ) -> None:
        values: dict[str, object] = {
            "train/loss": metric["train_loss"],
            "train/learning_rate": metric["learning_rate"],
            "train/grad_norm": metric["grad_norm"],
            "train/tokens_seen": metric["tokens_seen"],
        }
        if "validation_loss" in metric:
            values["eval/loss"] = metric["validation_loss"]
            values["eval/perplexity"] = metric["perplexity"]
            values["eval/best_loss"] = metric["best_val_loss"]
        if generation_path is not None:
            generation = json.loads(Path(generation_path).read_text(encoding="utf-8"))
            values["eval/generations"] = self._wandb.Table(
                columns=["step", "language", "name", "prompt", "response", "generated_tokens"],
                data=[
                    [
                        generation["step"],
                        sample["language"],
                        sample["name"],
                        sample["prompt"],
                        sample["response"],
                        sample["generated_tokens"],
                    ]
                    for sample in generation["samples"]
                ],
            )
        self._run.log(values, step=int(metric["step"]))

    def finish(self, *, exit_code: int = 0, summary: dict[str, object] | None = None) -> None:
        if summary:
            self._run.summary.update(summary)
        self._run.finish(exit_code=exit_code)
