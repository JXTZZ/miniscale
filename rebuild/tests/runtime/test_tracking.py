import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from miniscale.tracking import WandbTracker


class FakeRun:
    id = "run-123"

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.summary: dict[str, object] = {}
        self.exit_code: int | None = None

    def log(self, values: dict[str, object], *, step: int | None = None) -> None:
        self.logged.append((values, step))

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


class FailingRun(FakeRun):
    def __init__(self) -> None:
        super().__init__()
        self.log_calls = 0

    def log(self, values: dict[str, object], *, step: int | None = None) -> None:
        self.log_calls += 1
        raise ConnectionError("temporary W&B outage")


class FakeWandb:
    def __init__(self, outcomes: list[object] | None = None) -> None:
        self.outcomes = list(outcomes or [])
        self.init_calls: list[dict[str, object]] = []

    def init(self, **kwargs: object) -> object:
        self.init_calls.append(kwargs)
        if not self.outcomes:
            return FakeRun()
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @staticmethod
    def Table(*, columns: list[str], data: list[list[object]]) -> dict[str, object]:
        return {"columns": columns, "data": data}


def metric(step: int) -> dict[str, object]:
    return {
        "step": step,
        "tokens_seen": step * 10,
        "train_loss": 2.0,
        "learning_rate": 1e-4,
        "grad_norm": 0.5,
    }


class TrackingTests(unittest.TestCase):
    def test_initialization_failure_queues_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recovered_run = FakeRun()
            wandb = FakeWandb([ConnectionError("service unavailable"), recovered_run])
            with (
                patch.dict("sys.modules", {"wandb": wandb}),
                self.assertWarnsRegex(RuntimeWarning, "retry at step 2"),
            ):
                tracker = WandbTracker.start(
                    enabled=True,
                    project="MiniScale",
                    entity=None,
                    name=None,
                    run_id="stable-id",
                    mode="online",
                    config={},
                    directory=directory,
                    retry_every_steps=2,
                )
            self.assertIsNotNone(tracker)
            assert tracker is not None
            self.assertEqual(tracker.run_id, "stable-id")
            tracker.log(metric(1))
            self.assertTrue((Path(directory) / "wandb_pending.jsonl").exists())
            tracker.log(metric(2))
            self.assertEqual([step for _, step in recovered_run.logged], [1, 2])
            self.assertFalse((Path(directory) / "wandb_pending.jsonl").exists())

    def test_disabled_tracker_does_not_import_wandb(self) -> None:
        tracker = WandbTracker.start(
            enabled=False,
            project="MiniScale",
            entity=None,
            name=None,
            run_id=None,
            mode="disabled",
            config={},
            directory=".",
        )
        self.assertIsNone(tracker)

    def test_metrics_and_generations_upload_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generation = Path(directory) / "generation.json"
            generation.write_text(json.dumps({
                "step": 10,
                "samples": [{
                    "language": "zh",
                    "name": "chinese",
                    "prompt": "你好",
                    "response": "世界",
                    "generated_tokens": 2,
                }],
            }), encoding="utf-8")
            run = FakeRun()
            tracker = WandbTracker(
                FakeWandb(), run,
                pending_path=Path(directory) / "pending.jsonl",
            )
            value = metric(10)
            value.update({
                "validation_loss": 2.1,
                "perplexity": 8.17,
                "best_val_loss": 2.1,
                "target_tokens_seen": 90,
                "examples_seen": 12,
                "tokens_per_second": 1234.0,
                "supervised_tokens_per_second": 432.0,
                "update_seconds": 0.25,
                "validation_token_accuracy": 0.25,
                "validation_target_tokens": 80,
            })
            tracker.log(value, generation_path=generation)

            scalar_values, scalar_step = run.logged[0]
            table_values, table_step = run.logged[1]
            self.assertEqual(scalar_step, 10)
            self.assertEqual(scalar_values["train/loss"], 2.0)
            self.assertEqual(scalar_values["eval/loss"], 2.1)
            self.assertEqual(scalar_values["train/target_tokens_seen"], 90)
            self.assertEqual(scalar_values["train/examples_seen"], 12)
            self.assertEqual(scalar_values["performance/tokens_per_second"], 1234.0)
            self.assertEqual(scalar_values["eval/token_accuracy"], 0.25)
            self.assertNotIn("eval/generations", scalar_values)
            self.assertIsNone(table_step)
            self.assertIn("eval/generations", table_values)

    def test_logging_failure_is_non_fatal_then_backfills(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed_run = FailingRun()
            recovered_run = FakeRun()
            wandb = FakeWandb([recovered_run])
            tracker = WandbTracker(
                wandb,
                failed_run,
                init_kwargs={"id": "run-123", "resume": "allow"},
                pending_path=Path(directory) / "pending.jsonl",
                retry_every_steps=2,
            )
            with self.assertWarnsRegex(RuntimeWarning, "retried at step 12"):
                tracker.log(metric(10))
            tracker.log(metric(11))
            self.assertEqual(len(wandb.init_calls), 0)
            tracker.log(metric(12))

            self.assertEqual(failed_run.log_calls, 1)
            self.assertEqual([step for _, step in recovered_run.logged], [10, 11, 12])
            self.assertFalse((Path(directory) / "pending.jsonl").exists())

    def test_dpo_metrics_and_comparison_generations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generation = Path(directory) / "dpo-generation.json"
            generation.write_text(json.dumps({
                "stage": "dpo",
                "step": 5,
                "samples": [{
                    "language": "zh",
                    "name": "comparison",
                    "prompt": "你好",
                    "policy_response": "新回答",
                    "reference_response": "旧回答",
                    "policy_generated_tokens": 3,
                    "reference_generated_tokens": 3,
                }],
            }), encoding="utf-8")
            run = FakeRun()
            tracker = WandbTracker(FakeWandb(), run)
            value = metric(5)
            value.update({
                "pairs_seen": 16,
                "preference_accuracy": 0.75,
                "reward_margin": 0.2,
                "validation_loss": 0.6,
                "validation_reward_accuracy": 0.8,
                "validation_reward_margin": 0.3,
                "validation_pairs": 10,
                "best_val_loss": 0.6,
            })
            tracker.log(value, generation_path=generation)
            scalars, _ = run.logged[0]
            table, _ = run.logged[1]
            self.assertEqual(scalars["train/pairs_seen"], 16)
            self.assertEqual(scalars["eval/reward_accuracy"], 0.8)
            self.assertIn("reference_response", table["eval/generations"]["columns"])

    def test_pending_queue_survives_process_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pending = Path(directory) / "wandb_pending.jsonl"
            first = WandbTracker(
                FakeWandb(),
                FailingRun(),
                pending_path=pending,
                retry_every_steps=100,
            )
            with self.assertWarns(RuntimeWarning):
                first.log(metric(5))
            self.assertTrue(pending.exists())

            recovered_run = FakeRun()
            wandb = FakeWandb([recovered_run])
            with patch.dict("sys.modules", {"wandb": wandb}):
                second = WandbTracker.start(
                    enabled=True,
                    project="MiniScale",
                    entity=None,
                    name=None,
                    run_id="run-123",
                    mode="online",
                    config={},
                    directory=directory,
                    initial_step=5,
                )
            self.assertIsNotNone(second)
            self.assertEqual([step for _, step in recovered_run.logged], [5])
            self.assertFalse(pending.exists())


if __name__ == "__main__":
    unittest.main()
