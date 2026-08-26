import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.pretrain import (
    PretrainOptions,
    _validate_resume_signature,
    build_warmup_cosine_scheduler,
    run_pretrain_jsonl,
)
from miniscale.training.common import restore_rng_state


def write_corpus(path: Path, prefix: str = "resume-safe pretraining data 中文。") -> None:
    rows = [{"text": prefix * 100} for _ in range(3)]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class PretrainResumeTests(unittest.TestCase):
    def test_legacy_resume_requires_explicit_migration_opt_in(self) -> None:
        legacy = {"batch_size": 1, "sequence_length": 16}
        current = {"batch_size": 1, "sequence_length": 16, "shuffle_buffer_size": 8192}
        with self.assertRaisesRegex(ValueError, "allow-legacy-resume"):
            _validate_resume_signature(legacy, current)
        with self.assertWarnsRegex(RuntimeWarning, "legacy checkpoint"):
            _validate_resume_signature(legacy, current, allow_legacy=True)
        with self.assertRaises(ValueError):
            with self.assertWarns(RuntimeWarning):
                _validate_resume_signature(
                    {"batch_size": 8, "sequence_length": 16},
                    current,
                    allow_legacy=True,
                )

    def test_numpy_rng_state_is_restored(self) -> None:
        np.random.seed(123)
        state = np.random.get_state()
        expected = np.random.random(4)
        np.random.random(10)
        restore_rng_state({"numpy": state})
        np.testing.assert_array_equal(np.random.random(4), expected)

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
            validation = root / "validation.jsonl"
            write_corpus(corpus)
            write_corpus(validation, "held-out validation data English. ")
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
                validation_path=validation,
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
                validation_path=validation,
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
            validation = root / "validation.jsonl"
            write_corpus(corpus)
            write_corpus(validation, "held-out validation data English. ")
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
                validation_path=validation,
            )

            run = root / "run"
            self.assertTrue((run / "best.pt").exists())
            self.assertTrue((run / "final.pt").exists())
            self.assertTrue((run / "checkpoints/step_00000001.pt").exists())
            self.assertTrue(torch.isfinite(torch.tensor(float(result["best_val_loss"]))))
            manifest = json.loads((run / "pretrain_run.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["checkpoint_format_version"], 2)
            self.assertEqual(manifest["training"]["sequence_length"], 16)
            self.assertEqual(manifest["resume_identity"]["train_data"]["kind"], "file")
            self.assertEqual(manifest["derived"]["input_tokens_per_update"], 16)
            self.assertEqual(manifest["derived"]["target_tokens_per_update"], 15)

            generation = json.loads((run / "generations/step_00000001.json").read_text(encoding="utf-8"))
            self.assertFalse(generation["decoding"]["do_sample"])
            self.assertEqual(generation["decoding"]["strategy"], "greedy")
            self.assertEqual({sample["language"] for sample in generation["samples"]}, {"zh", "en", "python"})

            best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(best["format_version"], 2)
            self.assertIn("numpy", best["rng_state"])
            self.assertEqual(best["training_state"]["resume_signature"]["signature_version"], 2)
            self.assertIn("resolved_options", best["training_state"])
            self.assertEqual(best["step"], 1)
            self.assertEqual(best["best_val_loss"], best["training_state"]["best_val_loss"])
            metric = json.loads((run / "pretrain_metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
            self.assertIn("validation_loss", metric)
            self.assertIn("perplexity", metric)
            self.assertEqual(metric["target_tokens_seen"], 15)
            self.assertGreater(metric["tokens_per_second"], 0)
            self.assertGreater(metric["update_seconds"], 0)
            self.assertIsInstance(metric["grad_was_clipped"], bool)

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

    def test_resume_rejects_changed_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pretrain.jsonl"
            write_corpus(corpus)
            options = PretrainOptions(
                steps=2,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                validation_fraction=0,
                save_every=1,
                keep_last_checkpoints=2,
                generation_every=0,
                device="cpu",
            )
            run_pretrain_jsonl(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                corpus,
                root / "initial",
                options,
            )
            checkpoint = root / "initial/checkpoints/step_00000001.pt"
            write_corpus(corpus, "changed training corpus. ")
            resumed = PretrainOptions(
                **{
                    name: getattr(options, name)
                    for name in options.__dataclass_fields__
                    if name != "resume_from"
                },
                resume_from=checkpoint,
            )
            with self.assertRaisesRegex(ValueError, "train_data.sha256"):
                run_pretrain_jsonl(
                    MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                    ByteTokenizer(),
                    corpus,
                    root / "resumed",
                    resumed,
                )

    def test_dedicated_validation_must_not_duplicate_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pretrain.jsonl"
            duplicate = root / "validation.jsonl"
            write_corpus(corpus)
            duplicate.write_bytes(corpus.read_bytes())
            with self.assertRaisesRegex(ValueError, "identical to training"):
                run_pretrain_jsonl(
                    MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                    ByteTokenizer(),
                    corpus,
                    root / "run",
                    PretrainOptions(
                        steps=1,
                        batch_size=1,
                        sequence_length=16,
                        gradient_accumulation_steps=1,
                        validation_fraction=0,
                        save_every=0,
                        generation_every=0,
                        device="cpu",
                    ),
                    validation_path=duplicate,
                )

    def test_new_run_refuses_to_overwrite_existing_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pretrain.jsonl"
            output = root / "run"
            write_corpus(corpus)
            options = PretrainOptions(
                steps=1,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                validation_fraction=0,
                save_every=0,
                generation_every=0,
                device="cpu",
            )
            run_pretrain_jsonl(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                corpus,
                output,
                options,
            )
            with self.assertRaisesRegex(FileExistsError, "use --resume"):
                run_pretrain_jsonl(
                    MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                    ByteTokenizer(),
                    corpus,
                    output,
                    options,
                )

    def test_legacy_checkpoint_is_upgraded_after_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pretrain.jsonl"
            write_corpus(corpus)
            options = PretrainOptions(
                steps=2,
                batch_size=1,
                sequence_length=16,
                gradient_accumulation_steps=1,
                validation_fraction=0,
                save_every=1,
                keep_last_checkpoints=2,
                generation_every=0,
                device="cpu",
            )
            run_pretrain_jsonl(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                corpus,
                root / "initial",
                options,
            )
            payload = torch.load(
                root / "initial/checkpoints/step_00000001.pt",
                map_location="cpu",
                weights_only=False,
            )
            legacy_model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
            legacy_model.load_state_dict(payload["model"])
            legacy_optimizer = torch.optim.AdamW(
                legacy_model.parameters(),
                lr=options.learning_rate,
                weight_decay=options.weight_decay,
                betas=(options.adam_beta1, options.adam_beta2),
                eps=options.adam_eps,
            )
            legacy_scheduler = build_warmup_cosine_scheduler(
                legacy_optimizer,
                total_steps=options.steps,
                warmup_steps=options.warmup_steps,
                min_learning_rate=options.min_learning_rate,
            )
            for parameter in legacy_model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            legacy_optimizer.step()
            legacy_scheduler.step()
            payload["optimizer"] = legacy_optimizer.state_dict()
            payload["scheduler"] = legacy_scheduler.state_dict()
            payload.pop("format_version")
            payload["training_state"]["resume_signature"] = {
                "total_steps": 2,
                "batch_size": 1,
                "sequence_length": 16,
                "gradient_accumulation_steps": 1,
                "learning_rate": options.learning_rate,
                "min_learning_rate": options.min_learning_rate,
                "warmup_steps": options.warmup_steps,
                "shuffle_buffer_size": options.shuffle_buffer_size,
                "seed": options.seed,
            }
            legacy = root / "legacy.pt"
            torch.save(payload, legacy)
            resumed = PretrainOptions(
                **{
                    name: getattr(options, name)
                    for name in options.__dataclass_fields__
                    if name not in {"resume_from", "allow_legacy_resume"}
                },
                resume_from=legacy,
                allow_legacy_resume=True,
            )
            with self.assertWarnsRegex(RuntimeWarning, "legacy checkpoint"):
                run_pretrain_jsonl(
                    MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                    ByteTokenizer(),
                    corpus,
                    root / "migrated",
                    resumed,
                )
            upgraded = torch.load(root / "migrated/final.pt", map_location="cpu", weights_only=False)
            self.assertEqual(upgraded["format_version"], 2)
            self.assertEqual(upgraded["training_state"]["resume_signature"]["signature_version"], 2)
            self.assertEqual(
                [group["group_name"] for group in upgraded["optimizer"]["param_groups"]],
                ["decay", "no_decay"],
            )
            self.assertEqual(
                [group["weight_decay"] for group in upgraded["optimizer"]["param_groups"]],
                [options.weight_decay, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
