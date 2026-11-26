from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model.model_miniscale import MiniScaleConfig, MiniScaleForCausalLM
from model.tokenizer import ByteTokenizer
from trainer.checkpoint import load_checkpoint, read_checkpoint
from trainer.train_full_sft import SFTOptions, run_sft_jsonl


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

    def test_lora_checkpoint_resumes_and_loads_as_dense_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "sft.jsonl"
            write_corpus(corpus)
            tokenizer = ByteTokenizer()
            model = new_model()
            original = {name: value.clone() for name, value in model.state_dict().items()}
            settings = replace(options(), lora_rank=2)
            output = root / "lora"
            result = run_sft_jsonl(model, tokenizer, corpus, output, settings)
            checkpoint = read_checkpoint(result["checkpoint"])
            self.assertEqual(checkpoint["lora"], {"rank": 2, "alpha": 2.0})
            self.assertTrue(all("lora_" in name for name, p in model.named_parameters() if p.requires_grad))
            for name, weight in original.items():
                torch.testing.assert_close(checkpoint["model"][name], weight, rtol=0, atol=0)
            merged = load_checkpoint(result["checkpoint"]).eval()
            ids = torch.tensor([[1, 10, 15]])
            torch.testing.assert_close(merged(ids).logits, model.eval()(ids).logits)
            with self.assertRaisesRegex(ValueError, "resume options do not match"):
                run_sft_jsonl(
                    new_model(), tokenizer, corpus, output,
                    replace(options(), resume_from=output / "checkpoints/step_00000002.pt"),
                )
            run_sft_jsonl(
                new_model(), tokenizer, corpus, output,
                replace(settings, resume_from=output / "checkpoints/step_00000002.pt"),
            )
            resumed = read_checkpoint(result["checkpoint"])
            for name, weight in checkpoint["model"].items():
                torch.testing.assert_close(resumed["model"][name], weight, rtol=0, atol=0)

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
                save_every=1,
                early_stopping_patience=2,
                early_stopping_min_steps=1,
                early_stopping_validation_min_delta=0.01,
                early_stopping_quality_min_delta=0.01,
                device="cpu",
            )
            with (
                patch("trainer.train_full_sft.evaluate_sft", return_value=(2.0, 0.5, 10)),
                patch(
                    "trainer.train_full_sft.run_sft_generation_evaluation",
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
                best_loss = root / "run/best_loss.pt"
                best_state = read_checkpoint(best_loss)["training_state"]
                self.assertEqual(best_state["best_quality_step"], 1)
                resumed = run_sft_jsonl(
                    new_model(), ByteTokenizer(), corpus, root / "resumed",
                    replace(run_options, resume_from=best_loss), validation_path=validation,
                )

                old_best = read_checkpoint(best_loss)
                old_state = old_best["training_state"]
                del old_state["early_stop_reason"]
                old_state.update(best_quality_step=0, best_quality_score=float("-inf"),
                                 early_best_quality_score=float("-inf"))
                old_path = root / "old_best_loss.pt"
                torch.save(old_best, old_path)
                migrated = run_sft_jsonl(
                    new_model(), ByteTokenizer(), corpus, root / "migrated",
                    replace(run_options, resume_from=old_path), validation_path=validation,
                )
                stopped = root / "run/checkpoints/step_00000003.pt"
                with self.assertRaisesRegex(ValueError, "already reached early stopping"):
                    run_sft_jsonl(
                        new_model(), ByteTokenizer(), corpus, root / "already_stopped",
                        replace(run_options, resume_from=stopped), validation_path=validation,
                    )

            payload = read_checkpoint(result["checkpoint"])
            self.assertEqual(payload["step"], 3)
            self.assertEqual(result["stopped_early"], 1.0)
            for continuation in (resumed, migrated):
                continued = read_checkpoint(continuation["checkpoint"])
                self.assertEqual(continued["step"], 3)
                for name, expected in payload["model"].items():
                    self.assertTrue(torch.equal(expected, continued["model"][name]), name)


if __name__ == "__main__":
    unittest.main()
