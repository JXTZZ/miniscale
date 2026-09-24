import json
from pathlib import Path
import tempfile
import unittest

from miniscale.training.core.artifacts import append_metric
from miniscale.training.core.logging import format_training_metric


class TrainingLoggingTests(unittest.TestCase):
    def test_formats_compact_pretrain_metric(self) -> None:
        metric = {
            "stage": "pretrain",
            "step": 1900,
            "tokens_seen": 23_347_200,
            "target_tokens_seen": 23_316_800,
            "train_loss": 3.2176,
            "learning_rate": 0.0002995,
            "grad_norm": 0.7858,
            "grad_was_clipped": False,
            "update_seconds": 0.3082,
            "tokens_per_second": 39_867.0,
            "samples_per_second": 51.91,
            "cuda_peak_memory_mb": 5290.45,
        }
        self.assertEqual(
            format_training_metric(metric),
            "[pretrain] step 1900 | loss 3.218 | lr 3.00e-4 | grad 0.79 | "
            "tok/s 39.9k | mem 5290MB",
        )

    def test_marks_clipping_and_omits_unavailable_memory(self) -> None:
        metric = {
            "stage": "sft",
            "step": 540,
            "train_loss": 2.648,
            "learning_rate": 1.9e-5,
            "grad_norm": 1.31,
            "grad_was_clipped": True,
            "tokens_per_second": 6300.0,
        }
        self.assertEqual(
            format_training_metric(metric),
            "[sft] step 540 | loss 2.648 | lr 1.90e-5 | grad 1.31* | tok/s 6.3k",
        )

    def test_formats_stage_specific_live_signals(self) -> None:
        dpo = {
            "stage": "dpo", "step": 20, "train_loss": 0.61,
            "learning_rate": 5e-6, "grad_norm": 0.4,
            "grad_was_clipped": False, "tokens_per_second": 999.0,
            "preference_accuracy": 0.75, "validation_loss": 0.58,
        }
        self.assertEqual(
            format_training_metric(dpo),
            "[dpo] step 20 | loss 0.610 | lr 5.00e-6 | grad 0.40 | "
            "tok/s 999.0 | acc 75.0% | val 0.580",
        )
        grpo = {
            "stage": "grpo", "step": 7, "train_loss": -0.02,
            "learning_rate": 1e-5, "grad_norm": 0.9,
            "grad_was_clipped": False, "reward_mean": 0.4, "kl": 0.003,
            "rollouts_per_second": 12.4, "validation_reward": 0.45,
        }
        self.assertEqual(
            format_training_metric(grpo),
            "[grpo] step 7 | loss -0.020 | lr 1.00e-5 | grad 0.90 | "
            "reward 0.400 | kl 0.003 | roll/s 12.4 | val_reward 0.450",
        )

    def test_jsonl_retains_the_complete_metric(self) -> None:
        metric = {
            "stage": "sft", "step": 3, "train_loss": 2.0,
            "tokens_seen": 100, "target_tokens_seen": 40,
            "examples_seen": 2, "samples_per_second": 7.5,
            "supervised_tokens_per_second": 300.0, "update_seconds": 0.2,
            "grad_was_clipped": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            append_metric(path, metric)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), metric)


if __name__ == "__main__":
    unittest.main()
