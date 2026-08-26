from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.cli import build_parser, main
from miniscale.training.common import load_checkpoint
from miniscale.training.pretrain import (
    PretrainOptions,
    SmokePretrainOptions,
    build_pretrain_optimizer,
    pretrain_option_default,
    resolve_autocast_dtype,
    run_pretrain,
    run_pretrain_jsonl,
)


class PretrainTests(unittest.TestCase):
    def test_production_steps_are_explicit_and_cli_defaults_have_one_source(self) -> None:
        with self.assertRaises(TypeError):
            PretrainOptions()  # type: ignore[call-arg]
        arguments = build_parser().parse_args(["pretrain", "--steps", "10"])
        mappings = {
            "batch_size": "batch_size",
            "sequence_length": "sequence_length",
            "gradient_accumulation": "gradient_accumulation_steps",
            "learning_rate": "learning_rate",
            "min_learning_rate": "min_learning_rate",
            "weight_decay": "weight_decay",
            "adam_beta1": "adam_beta1",
            "adam_beta2": "adam_beta2",
            "adam_eps": "adam_eps",
            "grad_clip": "grad_clip",
            "seed": "seed",
            "device": "device",
            "precision": "precision",
            "log_every": "log_every",
            "validation_every": "validation_every",
            "validation_batches": "validation_batches",
            "validation_fraction": "validation_fraction",
            "warmup_steps": "warmup_steps",
            "save_every": "save_every",
            "keep_last": "keep_last_checkpoints",
            "generation_every": "generation_every",
            "generation_max_new_tokens": "generation_max_new_tokens",
            "shuffle_buffer_size": "shuffle_buffer_size",
            "wandb": "wandb_enabled",
            "wandb_project": "wandb_project",
            "wandb_entity": "wandb_entity",
            "wandb_run_name": "wandb_run_name",
            "wandb_run_id": "wandb_run_id",
            "wandb_mode": "wandb_mode",
            "wandb_retry_every": "wandb_retry_every_steps",
            "resume": "resume_from",
            "allow_legacy_resume": "allow_legacy_resume",
            "num_workers": "num_workers",
        }
        for argument_name, option_name in mappings.items():
            self.assertEqual(getattr(arguments, argument_name), pretrain_option_default(option_name))

    def test_precision_resolution_is_explicit_and_rejects_unsupported_devices(self) -> None:
        self.assertIsNone(resolve_autocast_dtype("fp32", torch.device("cpu")))
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
            resolve_autocast_dtype("bf16", torch.device("cpu"))
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertIs(resolve_autocast_dtype("bf16", torch.device("cuda")), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "precision"):
            resolve_autocast_dtype("fp16", torch.device("cuda"))

    def test_optimizer_excludes_norms_and_tied_embedding_from_weight_decay(self) -> None:
        model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
        optimizer = build_pretrain_optimizer(model, PretrainOptions(steps=1))
        groups = {group["group_name"]: group for group in optimizer.param_groups}
        decay_ids = {id(parameter) for parameter in groups["decay"]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups["no_decay"]["params"]}
        self.assertEqual(groups["decay"]["weight_decay"], 0.1)
        self.assertEqual(groups["no_decay"]["weight_decay"], 0.0)
        self.assertIn(id(model.layers[0].attention.query.weight), decay_ids)
        self.assertIn(id(model.layers[0].attention_norm.weight), no_decay_ids)
        self.assertIn(id(model.embedding.weight), no_decay_ids)
        self.assertFalse(decay_ids & no_decay_ids)

    def test_cli_seed_controls_model_initialization(self) -> None:
        initialized: list[torch.Tensor] = []

        def capture_model(model: MiniScaleForCausalLM, *_args: object, **_kwargs: object) -> dict[str, str]:
            initialized.append(model.embedding.weight.detach().clone())
            return {"checkpoint": "unused.pt"}

        with (
            patch("miniscale.cli.load_tokenizer", return_value=ByteTokenizer()),
            patch("miniscale.cli.MiniScaleConfig.small_64m", return_value=MiniScaleConfig.smoke()),
            patch("miniscale.cli.run_pretrain_jsonl", side_effect=capture_model),
            redirect_stdout(StringIO()),
        ):
            main(["pretrain", "--steps", "1", "--seed", "123"])
            main(["pretrain", "--steps", "1", "--seed", "123"])
        self.assertTrue(torch.equal(initialized[0], initialized[1]))

    def test_pretrain_writes_loadable_checkpoint(self) -> None:
        torch.manual_seed(1)
        model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
        tokenizer = ByteTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_pretrain(
                model,
                tokenizer,
                ["small models make training loops testable " * 4],
                directory,
                SmokePretrainOptions(steps=2, batch_size=1, sequence_length=48, device="cpu"),
            )
            checkpoint = Path(str(metrics["checkpoint"]))
            self.assertTrue(checkpoint.exists())
            restored = load_checkpoint(checkpoint)
            ids = torch.tensor([tokenizer.encode("hello", bos=True)])
            self.assertEqual(restored(ids).logits.shape[:2], ids.shape)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

    def test_production_pretrain_rejects_non_finite_loss(self) -> None:
        tokenizer = ByteTokenizer()
        model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
        original_forward = model.forward

        def non_finite_forward(*args: object, **kwargs: object):
            output = original_forward(*args, **kwargs)
            assert output.loss is not None
            output.loss = output.loss * float("nan")
            return output

        model.forward = non_finite_forward  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "pretrain.jsonl"
            corpus.write_text('{"text":"enough tokens for a full training block enough tokens"}\n')
            with self.assertRaisesRegex(FloatingPointError, "non-finite pretraining loss"):
                run_pretrain_jsonl(
                    model,
                    tokenizer,
                    corpus,
                    Path(directory) / "out",
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
                )


if __name__ == "__main__":
    unittest.main()
