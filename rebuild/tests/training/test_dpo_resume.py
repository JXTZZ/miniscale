from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.common import load_checkpoint
from miniscale.training.dpo import DPOOptions, run_dpo_jsonl


def preference(index: int) -> dict[str, object]:
    return {
        "chosen": [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": f"good answer {index}"},
        ],
        "rejected": [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": f"bad answer {index}"},
        ],
    }


def write_corpus(path: Path, *, offset: int = 0) -> None:
    path.write_text(
        "".join(json.dumps(preference(index + offset)) + "\n" for index in range(8)),
        encoding="utf-8",
    )


def new_model() -> MiniScaleForCausalLM:
    torch.manual_seed(123)
    return MiniScaleForCausalLM(MiniScaleConfig.smoke())


def options(*, resume: Path | None = None, learning_rate: float = 3e-4) -> DPOOptions:
    return DPOOptions(
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


class DPOResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dpo.jsonl"
            write_corpus(corpus)
            tokenizer = ByteTokenizer()
            run_dpo_jsonl(new_model(), tokenizer, corpus, root / "reference", options())
            run_dpo_jsonl(new_model(), tokenizer, corpus, root / "resumed", options())
            step_two = root / "resumed/checkpoints/step_00000002.pt"
            run_dpo_jsonl(
                new_model(),
                tokenizer,
                corpus,
                root / "resumed",
                options(resume=step_two),
            )
            reference = load_checkpoint(root / "reference/dpo.pt")
            resumed = load_checkpoint(root / "resumed/dpo.pt")
            for expected, actual in zip(reference.parameters(), resumed.parameters(), strict=True):
                self.assertTrue(torch.equal(expected, actual))
            payload = torch.load(root / "resumed/dpo.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["step"], 4)
            self.assertTrue((root / "resumed/reference.pt").is_file())
            self.assertIn("reference_identity", payload["training_state"])
            self.assertIn("resume_signature", payload["training_state"])

    def test_resume_rejects_trajectory_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dpo.jsonl"
            write_corpus(corpus)
            run_dpo_jsonl(new_model(), ByteTokenizer(), corpus, root / "run", options())
            checkpoint = root / "run/checkpoints/step_00000002.pt"
            with self.assertRaisesRegex(ValueError, "resume options do not match"):
                run_dpo_jsonl(
                    new_model(),
                    ByteTokenizer(),
                    corpus,
                    root / "run",
                    options(resume=checkpoint, learning_rate=2e-4),
                )

    def test_validation_best_generation_and_output_safety(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dpo.jsonl"
            validation = root / "validation.jsonl"
            write_corpus(corpus)
            write_corpus(validation, offset=100)
            run_options = DPOOptions(
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
            result = run_dpo_jsonl(
                new_model(),
                ByteTokenizer(),
                corpus,
                output,
                run_options,
                validation_path=validation,
            )
            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "generations/step_00000001.json").is_file())
            self.assertTrue((output / "dpo_run.json").is_file())
            metric = json.loads((output / "dpo_metrics.jsonl").read_text().splitlines()[-1])
            self.assertIn("validation_reward_margin", metric)
            self.assertTrue(Path(str(result["checkpoint"])).is_file())
            with self.assertRaisesRegex(FileExistsError, "output already contains DPO artifacts"):
                run_dpo_jsonl(
                    new_model(),
                    ByteTokenizer(),
                    corpus,
                    output,
                    run_options,
                    validation_path=validation,
                )

    def test_non_finite_metric_writes_emergency_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dpo.jsonl"
            write_corpus(corpus)
            model = new_model()
            with torch.no_grad():
                next(model.parameters()).fill_(float("nan"))
            run_options = DPOOptions(
                steps=1,
                batch_size=1,
                max_length=128,
                min_context_tokens=8,
                gradient_accumulation_steps=1,
                validation_fraction=0,
                warmup_steps=0,
                save_every=0,
                generation_every=0,
                device="cpu",
            )
            with self.assertRaisesRegex(FloatingPointError, "non-finite DPO metric"):
                run_dpo_jsonl(model, ByteTokenizer(), corpus, root / "run", run_options)
            self.assertTrue((root / "run/checkpoints/emergency_step_00000000.pt").is_file())

    def test_parent_sft_target_mode_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "dpo.jsonl"
            write_corpus(corpus)
            model = new_model()
            checkpoint = root / "sft.pt"
            torch.save({
                "stage": "sft",
                "config": model.config,
                "model": model.state_dict(),
                "training_state": {"resolved_options": {"target_mode": "response_only"}},
            }, checkpoint)
            run_options = DPOOptions(
                steps=1,
                batch_size=1,
                target_mode="reasoning_and_response",
                gradient_accumulation_steps=1,
                validation_fraction=0,
                warmup_steps=0,
                save_every=0,
                generation_every=0,
                device="cpu",
            )
            with self.assertRaisesRegex(ValueError, "does not match parent SFT mode"):
                run_dpo_jsonl(
                    model,
                    ByteTokenizer(),
                    corpus,
                    root / "run",
                    run_options,
                    initial_checkpoint_path=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
