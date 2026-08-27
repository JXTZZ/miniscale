from pathlib import Path
import tempfile
import unittest

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.cli import build_parser
from miniscale.training.sft import SFTOptions, SmokeSFTOptions, run_sft, sft_option_default


class SFTTests(unittest.TestCase):
    def test_production_steps_are_explicit_and_cli_defaults_have_one_source(self) -> None:
        with self.assertRaises(TypeError):
            SFTOptions()  # type: ignore[call-arg]
        arguments = build_parser().parse_args(["sft", "--steps", "10", "--checkpoint", "base.pt"])
        mappings = {
            "batch_size": "batch_size",
            "min_context_tokens": "min_context_tokens",
            "target_mode": "target_mode",
            "gradient_accumulation": "gradient_accumulation_steps",
            "learning_rate": "learning_rate",
            "min_learning_rate": "min_learning_rate",
            "weight_decay": "weight_decay",
            "adam_beta1": "adam_beta1",
            "adam_beta2": "adam_beta2",
            "adam_eps": "adam_eps",
            "grad_clip": "grad_clip",
            "warmup_steps": "warmup_steps",
            "precision": "precision",
            "validation_fraction": "validation_fraction",
            "validation_every": "validation_every",
            "validation_batches": "validation_batches",
            "save_every": "save_every",
            "keep_last": "keep_last_checkpoints",
            "generation_every": "generation_every",
            "generation_max_new_tokens": "generation_max_new_tokens",
            "deduplicate_exact": "deduplicate_exact",
            "seed": "seed",
            "num_workers": "num_workers",
            "log_every": "log_every",
            "wandb": "wandb_enabled",
            "wandb_project": "wandb_project",
            "wandb_mode": "wandb_mode",
            "wandb_retry_every": "wandb_retry_every_steps",
        }
        for argument_name, option_name in mappings.items():
            self.assertEqual(getattr(arguments, argument_name), sft_option_default(option_name))

    def test_checkpoint_and_resume_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["sft", "--steps", "1"])
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "sft", "--steps", "1", "--checkpoint", "base.pt", "--resume", "sft.pt"
            ])

    def test_sft_stage_writes_checkpoint(self) -> None:
        conversations = [
            [
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": "hello"},
            ],
            [
                {"role": "user", "content": "2+2"},
                {"role": "assistant", "content": "4"},
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_sft(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                conversations,
                directory,
                SmokeSFTOptions(steps=2, batch_size=2, device="cpu"),
            )
            self.assertTrue(Path(str(metrics["checkpoint"])).exists())
            self.assertGreater(float(metrics["loss"]), 0.0)

    def test_long_conversation_is_limited_to_model_context(self) -> None:
        conversations = [[
            {"role": "user", "content": "x" * 300},
            {"role": "assistant", "content": "kept"},
        ]]
        with tempfile.TemporaryDirectory() as directory:
            run_sft(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                conversations,
                directory,
                SmokeSFTOptions(steps=1, device="cpu"),
            )


if __name__ == "__main__":
    unittest.main()
