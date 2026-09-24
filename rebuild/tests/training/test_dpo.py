from __future__ import annotations

import unittest

from miniscale.cli import build_parser
from miniscale.training.dpo import DPOOptions, dpo_option_default


class DPOTests(unittest.TestCase):
    def test_production_steps_are_explicit_and_cli_defaults_have_one_source(self) -> None:
        with self.assertRaises(TypeError):
            DPOOptions()  # type: ignore[call-arg]
        arguments = build_parser().parse_args(["dpo", "--steps", "10", "--checkpoint", "sft.pt"])
        mappings = {
            "batch_size": "batch_size",
            "gradient_accumulation": "gradient_accumulation_steps",
            "learning_rate": "learning_rate",
            "min_learning_rate": "min_learning_rate",
            "weight_decay": "weight_decay",
            "adam_beta1": "adam_beta1",
            "adam_beta2": "adam_beta2",
            "adam_eps": "adam_eps",
            "beta": "beta",
            "grad_clip": "grad_clip",
            "warmup_steps": "warmup_steps",
            "precision": "precision",
            "min_context_tokens": "min_context_tokens",
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
            self.assertEqual(getattr(arguments, argument_name), dpo_option_default(option_name))
        self.assertIsNone(arguments.target_mode)

    def test_checkpoint_and_resume_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["dpo", "--steps", "1"])
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "dpo", "--steps", "1", "--checkpoint", "sft.pt", "--resume", "dpo.pt"
            ])


if __name__ == "__main__":
    unittest.main()
