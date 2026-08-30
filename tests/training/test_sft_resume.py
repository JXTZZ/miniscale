from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.common import load_checkpoint
from miniscale.training.sft import SFTOptions, run_sft_jsonl


def write_corpus(path: Path) -> None:
    rows = [
        {
            "conversations": [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        }
        for index in range(8)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def options(*, resume: Path | None = None, learning_rate: float = 3e-4) -> SFTOptions:
    return SFTOptions(
        steps=4,
        batch_size=1,
        max_length=128,
        min_context_tokens=8,
        learning_rate=learning_rate,
        min_learning_rate=3e-5,
        warmup_steps=1,
        gradient_accumulation_steps=1,
        validation_fraction=0,
        save_every=2,
        keep_last_checkpoints=2,
        generation_every=0,
        log_every=1,
        seed=19,
        device="cpu",
        resume_from=resume,
    )


def new_model() -> MiniScaleForCausalLM:
    torch.manual_seed(123)
    return MiniScaleForCausalLM(MiniScaleConfig.smoke())


class SFTResumeTests(unittest.TestCase):
    def test_non_finite_loss_writes_emergency_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "sft.jsonl"
            write_corpus(corpus)
            model = new_model()
            original_forward = model.forward

            def non_finite_forward(*args: object, **kwargs: object):
                result = original_forward(*args, **kwargs)
                assert result.loss is not None
                result.loss = result.loss * float("nan")
                return result

            model.forward = non_finite_forward  # type: ignore[method-assign]
            run_options = SFTOptions(
                steps=1,
                batch_size=1,
                max_length=128,
                min_context_tokens=8,
                gradient_accumulation_steps=1,
                validation_fraction=0,
                save_every=0,
                generation_every=0,
                device="cpu",
            )
            with self.assertRaisesRegex(FloatingPointError, "non-finite SFT loss"):
                run_sft_jsonl(model, ByteTokenizer(), corpus, root / "run", run_options)
            self.assertTrue((root / "run/checkpoints/emergency_step_00000000.pt").is_file())

    def test_validation_best_generation_and_output_safety(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "sft.jsonl"
            validation = root / "validation.jsonl"
            write_corpus(corpus)
            write_corpus(validation)
            # Dedicated validation must differ by content identity.
            validation.write_text(
                json.dumps({
                    "conversations": [
                        {"role": "user", "content": "held out question"},
                        {"role": "assistant", "content": "held out answer"},
                    ]
                }) + "\n",
                encoding="utf-8",
            )
            run_options = SFTOptions(
                steps=1,
                batch_size=1,
                max_length=128,
                min_context_tokens=8,
                learning_rate=3e-4,
                min_learning_rate=3e-5,
                warmup_steps=0,
                gradient_accumulation_steps=1,
                validation_fraction=0,
                validation_every=1,
                validation_batches=1,
                save_every=0,
                generation_every=1,
                generation_max_new_tokens=2,
                device="cpu",
            )
            output = root / "run"
            result = run_sft_jsonl(
                new_model(),
                ByteTokenizer(),
                corpus,
                output,
                run_options,
                validation_path=validation,
            )
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "generations/step_00000001.json").is_file())
            self.assertTrue((output / "sft_run.json").is_file())
            metric = json.loads((output / "sft_metrics.jsonl").read_text().splitlines()[-1])
            self.assertIn("validation_token_accuracy", metric)
            self.assertTrue(Path(str(result["checkpoint"])).is_file())
            with self.assertRaisesRegex(FileExistsError, "output already contains SFT artifacts"):
                run_sft_jsonl(
                    new_model(),
                    ByteTokenizer(),
                    corpus,
                    output,
                    run_options,
                    validation_path=validation,
                )

    def test_resume_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "sft.jsonl"
            write_corpus(corpus)
            tokenizer = ByteTokenizer()
            run_sft_jsonl(new_model(), tokenizer, corpus, root / "reference", options())
            run_sft_jsonl(new_model(), tokenizer, corpus, root / "resumed", options())
            step_two = root / "resumed/checkpoints/step_00000002.pt"
            run_sft_jsonl(
                new_model(),
                tokenizer,
                corpus,
                root / "resumed",
                options(resume=step_two),
            )
            reference = load_checkpoint(root / "reference/sft.pt")
            resumed = load_checkpoint(root / "resumed/sft.pt")
            for expected, actual in zip(reference.parameters(), resumed.parameters(), strict=True):
                self.assertTrue(torch.equal(expected, actual))
            payload = torch.load(root / "resumed/sft.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["step"], 4)
            self.assertIn("optimizer", payload)
            self.assertIn("resume_signature", payload["training_state"])

    def test_resume_rejects_trajectory_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "sft.jsonl"
            write_corpus(corpus)
            tokenizer = ByteTokenizer()
            run_sft_jsonl(new_model(), tokenizer, corpus, root / "run", options())
            checkpoint = root / "run/checkpoints/step_00000002.pt"
            with self.assertRaisesRegex(ValueError, "resume options do not match"):
                run_sft_jsonl(
                    new_model(),
                    tokenizer,
                    corpus,
                    root / "run",
                    options(resume=checkpoint, learning_rate=2e-4),
                )

    def test_quality_and_loss_patience_stop_training_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "sft.jsonl"
            validation = root / "validation.jsonl"
            write_corpus(corpus)
            validation.write_text(
                json.dumps({
                    "conversations": [
                        {"role": "user", "content": "held out"},
                        {"role": "assistant", "content": "answer"},
                    ]
                }) + "\n",
                encoding="utf-8",
            )
            summary = {
                "generation_eos_rate": 1.0,
                "generation_max_length_rate": 0.0,
                "generation_loop_rate": 0.0,
                "generation_repeated_4gram_fraction": 0.0,
                "generation_task_pass_rate": 1.0,
                "generation_prompt_echo_rate": 0.0,
                "generation_special_token_leak_rate": 0.0,
                "generation_think_leak_rate": 0.0,
                "generation_average_tokens": 4.0,
            }
            run_options = SFTOptions(
                steps=6,
                batch_size=1,
                max_length=128,
                min_context_tokens=8,
                gradient_accumulation_steps=1,
                validation_fraction=0,
                validation_every=1,
                validation_batches=1,
                generation_every=1,
                generation_suite=None,
                save_every=0,
                early_stopping_patience=2,
                early_stopping_min_steps=1,
                early_stopping_validation_min_delta=0.01,
                early_stopping_quality_min_delta=0.01,
                device="cpu",
            )
            with (
                patch("miniscale.training.stages.sft.evaluate_sft", return_value=(2.0, 0.5, 10)),
                patch(
                    "miniscale.training.stages.sft.run_sft_generation_evaluation",
                    return_value=(root / "generation.json", summary),
                ),
            ):
                result = run_sft_jsonl(
                    new_model(),
                    ByteTokenizer(),
                    corpus,
                    root / "run",
                    run_options,
                    validation_path=validation,
                )

            payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(payload["step"], 3)
            self.assertEqual(result["stopped_early"], 1.0)


if __name__ == "__main__":
    unittest.main()
