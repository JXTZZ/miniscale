import argparse
import importlib
import unittest

from scripts import audit_data, doctor, evaluate_checkpoints
from trainer import train_agent, train_dpo, train_full_sft, train_grpo, train_pretrain


class PublicAPIContractTests(unittest.TestCase):
    def test_modules_expose_their_source_entry_points(self) -> None:
        for module_name in (
            "eval_llm",
            "trainer.train_pretrain",
            "trainer.train_full_sft",
            "trainer.train_dpo",
            "trainer.train_grpo",
            "trainer.train_agent",
            "trainer.train_tokenizer",
            "scripts.audit_data",
            "scripts.prepare_data",
            "scripts.inspect_tokenizer",
            "scripts.evaluate_checkpoints",
            "scripts.inference",
            "scripts.doctor",
            "scripts.smoke_pipeline",
        ):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(module.main))
        self.assertIs(importlib.import_module("eval_llm").main, importlib.import_module("scripts.inference").main)
        self.assertIn("python", doctor.environment_report())

    def test_tool_subcommands(self) -> None:
        for module, expected in (
            (audit_data, {"pretrain", "sft", "dpo", "grpo", "agent"}),
            (evaluate_checkpoints, {"rl", "sft"}),
        ):
            with self.subTest(module=module.__name__):
                subcommands = next(
                    action for action in module.build_parser()._actions
                    if isinstance(action, argparse._SubParsersAction)
                )
                self.assertEqual(set(subcommands.choices), expected)

    def test_training_steps_are_explicit_and_checkpoint_sources_are_exclusive(self) -> None:
        for module in (train_pretrain, train_full_sft, train_dpo, train_grpo, train_agent):
            with self.subTest(module=module.__name__):
                parser = module.build_parser()
                required = {action.dest for action in parser._actions if action.required}
                self.assertIn("steps", required)
                if module is train_pretrain:
                    continue
                source_groups = [
                    {action.dest for action in group._group_actions}
                    for group in parser._mutually_exclusive_groups
                    if group.required
                ]
                self.assertIn({"checkpoint", "resume"}, source_groups)


if __name__ == "__main__":
    unittest.main()
