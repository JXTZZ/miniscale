import json
from pathlib import Path
import tempfile
import unittest

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.pretrain import (
    PretrainOptions,
    build_warmup_cosine_scheduler,
    run_pretrain_jsonl,
)


def write_corpus(path: Path) -> None:
    rows = [{"text": "resume-safe pretraining data 中文。" * 100} for _ in range(3)]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class PretrainResumeTests(unittest.TestCase):
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
                validation_fraction=0,
                save_every=2,
                keep_last_checkpoints=1,
                device="cpu",
            )
            uninterrupted_model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
            run_pretrain_jsonl(
                uninterrupted_model,
                ByteTokenizer(),
                corpus,
                root / "uninterrupted",
                options,
            )
            checkpoints = list((root / "uninterrupted/checkpoints").glob("step_*.pt"))
            self.assertEqual([path.name for path in checkpoints], ["step_00000004.pt"])
            periodic = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            self.assertIn("optimizer", periodic)
            self.assertIn("scheduler", periodic)
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
            )
            self.assertEqual(float(result["tokens_seen"]), 96.0)
            for name, expected in uninterrupted_model.state_dict().items():
                self.assertTrue(torch.equal(expected, resumed_model.state_dict()[name]), name)

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
