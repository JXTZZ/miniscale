import json
from pathlib import Path
import tempfile
import unittest

from miniscale.tracking import WandbTracker


class FakeRun:
    id = "run-123"

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int]] = []
        self.summary: dict[str, object] = {}
        self.exit_code: int | None = None

    def log(self, values: dict[str, object], *, step: int) -> None:
        self.logged.append((values, step))

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


class FakeWandb:
    @staticmethod
    def Table(*, columns: list[str], data: list[list[object]]) -> dict[str, object]:
        return {"columns": columns, "data": data}


class TrackingTests(unittest.TestCase):
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

    def test_metrics_and_generations_are_mapped_to_wandb(self) -> None:
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
            tracker = WandbTracker(FakeWandb(), run)
            tracker.log({
                "step": 10,
                "tokens_seen": 100,
                "train_loss": 2.0,
                "learning_rate": 1e-4,
                "grad_norm": 0.5,
                "validation_loss": 2.1,
                "perplexity": 8.17,
                "best_val_loss": 2.1,
            }, generation_path=generation)
            values, step = run.logged[0]
            self.assertEqual(step, 10)
            self.assertEqual(values["train/loss"], 2.0)
            self.assertEqual(values["eval/loss"], 2.1)
            self.assertIn("eval/generations", values)


if __name__ == "__main__":
    unittest.main()
