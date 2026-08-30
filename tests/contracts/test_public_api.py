import argparse
import importlib
import unittest

import miniscale
import miniscale.training as training
from miniscale.cli import build_parser, environment_report, main


class PublicAPIContractTests(unittest.TestCase):
    def test_root_exports_remain_stable(self) -> None:
        self.assertEqual(
            set(miniscale.__all__),
            {
                "ByteTokenizer",
                "HuggingFaceTokenizer",
                "MiniScaleConfig",
                "MiniScaleForCausalLM",
                "SentencePieceTokenizer",
            },
        )

    def test_training_exports_remain_stable(self) -> None:
        self.assertEqual(
            set(training.__all__),
            {
                "AgentRLOptions",
                "DPOOptions",
                "GRPOOptions",
                "PretrainOptions",
                "SmokePretrainOptions",
                "RLTask",
                "SFTOptions",
                "SmokeSFTOptions",
                "run_agent_grpo",
                "run_agent_grpo_jsonl",
                "run_dpo_jsonl",
                "run_grpo",
                "run_grpo_jsonl",
                "run_pretrain",
                "run_pretrain_jsonl",
                "run_sft",
                "run_sft_jsonl",
            },
        )

    def test_cli_commands_remain_stable(self) -> None:
        parser = build_parser()
        subcommands = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(
            set(subcommands.choices),
            {
                "doctor",
                "pipeline",
                "generate",
                "evaluate",
                "evaluate-sft",
                "tokenize",
                "audit-pretrain-data",
                "audit-sft-data",
                "audit-dpo-data",
                "audit-grpo-data",
                "audit-agent-data",
                "prepare-sft-data",
                "train-tokenizer",
                "pretrain",
                "sft",
                "dpo",
                "grpo",
                "agent-rl",
            },
        )
        self.assertTrue(callable(main))
        self.assertIn("python", environment_report())

    def test_legacy_compatibility_modules_still_import(self) -> None:
        for module_name in (
            "model.model",
            "dataset.lm_dataset",
            "trainer.trainer_utils",
            "miniscale.agent_data",
            "miniscale.data_audit",
            "miniscale.dpo_data_audit",
            "miniscale.preference_data",
            "miniscale.rl_data",
            "miniscale.sft_data",
            "miniscale.sft_data_audit",
            "miniscale.sft_data_prepare",
            "miniscale.training.agent_rl",
            "miniscale.training.checkpoint",
            "miniscale.training.common",
            "miniscale.training.dpo",
            "miniscale.training.dpo_config",
            "miniscale.training.dpo_objective",
            "miniscale.training.grpo",
            "miniscale.training.grpo_objective",
            "miniscale.training.pretrain",
            "miniscale.training.rl_config",
            "miniscale.training.runtime",
            "miniscale.training.sft",
            "miniscale.training.sft_config",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
