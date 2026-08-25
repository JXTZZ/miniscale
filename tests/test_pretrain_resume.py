import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.pretrain import (
    PretrainOptions,
    _validate_resume_signature,
    build_warmup_cosine_scheduler,
    run_pretrain_jsonl,
)
from miniscale.training.common import restore_rng_state


def write_corpus(path: Path) -> None:
    rows = [{"text": "resume-safe pretraining data 中文。" * 100} for _ in range(3)]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class PretrainResumeTests(unittest.TestCase):
    def test_legacy_resume_signature_allows_new_data_order_fields(self) -> None:
        _validate_resume_signature(
            {"batch_size": 1, "sequence_length": 16},
            {"batch_size": 1, "sequence_length": 16, "shuffle_buffer_size": 8192},
        )
        with self.assertRaises(ValueError):
            _validate_resume_signature(
                {"batch_size": 8, "sequence_length": 16},
                {"batch_size": 1, "sequence_length": 16, "shuffle_buffer_size": 8192},
            )

    def test_cuda_rng_restore_passes_cpu_byte_tensors(self) -> None:
        state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_rng_state_all") as set_rng_state_all,
        ):
            restore_rng_state({"cuda": [state]})
        restored = set_rng_state_all.call_args.args[0][0]
        self.assertEqual(restored.device.type, "cpu")
        self.assertEqual(restored.dtype, torch.uint8)

    def test_warmup_cosine_reaches_peak_and_floor(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4)
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=10,
            warmup_steps=2,
            min_learning_rate=3e-5,
        )
        used_lrs = []
        for _ in range(10):
            used_lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(used_lrs[0], 1.5e-4)
        self.assertAlmostEqual(used_lrs[1], 3e-4)
        self.assertAlmostEqual(used_lrs[-1], 3e-5)
        self.assertTrue(all(left >= right for left, right in zip(used_lrs[1:], used_lrs[2:])))

    def test_periodic_checkpoint_retention_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pretrain.jsonl"
            write_corpus(corpus)
            options = PretrainOptions(
                steps=6,
                batch_size=1,
                sequence_length=16,
                learning_rate=3e-4,
                min_learning_rate=3e-5,
                warmup_steps=2,
                gradient_accumulation_steps=1,
                log_every=1,
                validation_every=2,
                validation_batches=1,
                validation_fraction=0,
                save_every=2,
                keep_last_checkpoints=2,
                generation_every=0,
                device="cpu",
            )
            uninterrupted_model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
            run_pretrain_jsonl(
                uninterrupted_model,
                ByteTokenizer(),
                corpus,
                root / "uninterrupted",
                options,
                validation_path=corpus,
            )
            checkpoints = sorted((root / "uninterrupted/checkpoints").glob("step_*.pt"))
            self.assertEqual(
                [path.name for path in checkpoints],
                ["step_00000004.pt", "step_00000006.pt"],
            )
            periodic = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            self.assertIn("optimizer", periodic)
            self.assertIn("scheduler", periodic)
            self.assertIn("rng_state", periodic)
            self.assertIn("tokens_seen", periodic)
            self.assertIn("best_val_loss", periodic)
            self.assertEqual(periodic["step"], 4)

            resumed_model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
            resumed_options = PretrainOptions(
                **{
                    name: getattr(options, name)
                    for name in options.__dataclass_fields__
                    if name != "resume_from"
                },
                resume_from=checkpoints[0],
            )
            result = run_pretrain_jsonl(
                resumed_model,
                ByteTokenizer(),
                corpus,
                root / "resumed",
                resumed_options,
                validation_path=corpus,
            )
            self.assertEqual(float(result["tokens_seen"]), 96.0)
            self.assertTrue((root / "uninterrupted/final.pt").exists())
            uninterrupted_final = torch.load(
                root / "uninterrupted/final.pt", map_location="cpu", weights_only=False
            )
            resumed_final = torch.load(root / "resumed/final.pt", map_location="cpu", weights_only=False)
            self.assertEqual(uninterrupted_final["scheduler"], resumed_final["scheduler"])
            self.assertAlmostEqual(
                float(uninterrupted_final["training_state"]["best_val_loss"]),
                float(resumed_final["training_state"]["best_val_loss"]),
            )
            self.assertAlmostEqual(
                float(result["best_val_loss"]),
                float(resumed_final["training_state"]["best_val_loss"]),
            )
            for name, expected in uninterrupted_model.state_dict().items():
                self.assertTrue(torch.equal(expected, resumed_model.state_dict()[name]), name)

    def test_validation_best_final_and_generation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pretrain.jsonl"
            write_corpus(corpus)
            result = run_pretrain_jsonl(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                corpus,
                root / "run",
                PretrainOptions(
                    steps=1,
                    batch_size=1,
                    sequence_length=16,
                    warmup_steps=1,
                    gradient_accumulation_steps=1,
                    log_every=1,
                    validation_every=1,
                    validation_batches=1,
                    validation_fraction=0,
                    save_every=1,
                    keep_last_checkpoints=1,
                    generation_every=1,
                    generation_max_new_tokens=2,
                    device="cpu",
                ),
                validation_path=corpus,
            )

            run = root / "run"
            self.assertTrue((run / "best.pt").exists())
            self.assertTrue((run / "final.pt").exists())
            self.assertTrue((run / "checkpoints/step_00000001.pt").exists())
            self.assertTrue(torch.isfinite(torch.tensor(float(result["best_val_loss"]))))

            generation = json.loads((run / "generations/step_00000001.json").read_text(encoding="utf-8"))
            self.assertFalse(generation["decoding"]["do_sample"])
            self.assertEqual(generation["decoding"]["strategy"], "greedy")
            self.assertEqual({sample["language"] for sample in generation["samples"]}, {"zh", "en", "python"})

            best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(best["step"], 1)
            self.assertEqual(best["best_val_loss"], best["training_state"]["best_val_loss"])
            metric = json.loads((run / "pretrain_metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertIn("validation_loss", metric)
            self.assertIn("perplexity", metric)

    def test_keyboard_interrupt_writes_emergency_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pretrain.jsonl"
            write_corpus(corpus)
            model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
            original_forward = model.forward
            calls = 0

            def interrupt_after_one_step(*args: object, **kwargs: object):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise KeyboardInterrupt
                return original_forward(*args, **kwargs)

            model.forward = interrupt_after_one_step  # type: ignore[method-assign]
            with self.assertRaises(KeyboardInterrupt):
                run_pretrain_jsonl(
                    model,
                    ByteTokenizer(),
                    corpus,
                    root / "interrupted",
                    PretrainOptions(
                        steps=3,
                        batch_size=1,
                        sequence_length=16,
                        warmup_steps=1,
                        gradient_accumulation_steps=1,
                        validation_fraction=0,
                        save_every=0,
                        generation_every=0,
                        device="cpu",
                    ),
                )
            emergency = root / "interrupted/checkpoints/emergency_step_00000001.pt"
            self.assertTrue(emergency.exists())
            payload = torch.load(emergency, map_location="cpu", weights_only=False)
            self.assertEqual(payload["step"], 1)
            self.assertEqual(payload["training_state"]["micro_batches_seen"], 1)


if __name__ == "__main__":
    unittest.main()
