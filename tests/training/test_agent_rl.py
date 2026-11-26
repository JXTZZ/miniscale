from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from dataset.rl_dataset import CalculatorTask, filter_calculator_tools
from model.model_miniscale import MiniScaleConfig, MiniScaleForCausalLM
from model.tokenizer import ByteTokenizer
from trainer.rl.environment import parse_tool_call, parse_tool_call_payload, safe_calculate
from trainer.rl.options import AgentRLOptions
from trainer.rl.rollout import rollout_agent
from trainer.train_agent import run_agent_grpo


class AgentRLTests(unittest.TestCase):
    def test_calculator_is_functional_and_restricted(self) -> None:
        self.assertEqual(safe_calculate("(2+3)*4"), 20)
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('id')")
        self.assertEqual(
            parse_tool_call('<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'),
            "2+3",
        )
        self.assertIsNone(parse_tool_call_payload(
            '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'
            '<tool_call>{"name":"calculator","arguments":{"expression":"3+4"}}</tool_call>'
        ))

    def test_unsupported_tool_schemas_are_not_exposed(self) -> None:
        tools = [
            {"function": {"name": "calculate_math"}},
            {"function": {"name": "get_current_weather"}},
        ]
        filtered = filter_calculator_tools(tools)
        self.assertIsNotNone(filtered)
        self.assertEqual(len(filtered or []), 1)

    def test_scripted_multiturn_rollout_masks_observation(self) -> None:
        responses = [
            '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>',
            "5",
        ]
        trajectory = rollout_agent(
            MiniScaleForCausalLM(MiniScaleConfig.smoke()),
            ByteTokenizer(),
            CalculatorTask("Calculate 2+3", "2+3", "5"),
            AgentRLOptions(max_turns=2, max_new_tokens=80, device="cpu"),
            torch.device("cpu"),
            response_fn=lambda _transcript, turn: responses[turn],
        )
        self.assertGreater(trajectory.observation_tokens, 0)
        for start, end in trajectory.observation_ranges:
            self.assertEqual(sum(trajectory.action_mask[start:end]), 0)
        self.assertGreaterEqual(trajectory.reward, 1.2)

    def test_agent_stage_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_agent_grpo(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                [CalculatorTask("2+2?", "2+2", "4")],
                directory,
                AgentRLOptions(
                    steps=1, group_size=2, max_new_tokens=4, device="cpu",
                    loss_type="cispo", use_cache=True,
                ),
            )
            self.assertTrue(Path(str(metrics["checkpoint"])).exists())

    def test_reward_scores_all_expected_answers(self) -> None:
        from trainer.rl.environment import CalculatorEnv

        env = CalculatorEnv(CalculatorTask("two calculations", "", ("4", "9")))
        self.assertGreater(env.reward("4 and 9"), env.reward("4"))

    def test_in_memory_resume_rejects_changed_system_prompt_or_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
            tokenizer = ByteTokenizer()
            task = CalculatorTask("2+2?", "2+2", "4", system_prompt="Answer briefly")
            options = AgentRLOptions(steps=2, group_size=2, max_new_tokens=4, save_every=1, device="cpu")
            run_agent_grpo(model, tokenizer, [task], directory, options)
            checkpoint = Path(directory) / "checkpoints/step_00000001.pt"
            for changes in (
                {"system_prompt": "Explain your arithmetic"},
                {"tools": [{"function": {"name": "calculator"}}]},
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "data.train.sha256"):
                    run_agent_grpo(
                        model, tokenizer, [replace(task, **changes)], directory,
                        replace(options, resume_from=checkpoint),
                    )
            run_agent_grpo(model, tokenizer, [task], directory, replace(options, resume_from=checkpoint))


if __name__ == "__main__":
    unittest.main()
