from pathlib import Path
import tempfile
import unittest

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.agent_env import CalculatorTask, parse_tool_call, safe_calculate
from miniscale.training.agent_rl import AgentRLOptions, rollout_agent, run_agent_grpo


class AgentRLTests(unittest.TestCase):
    def test_calculator_is_functional_and_restricted(self) -> None:
        self.assertEqual(safe_calculate("(2+3)*4"), 20)
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('id')")
        self.assertEqual(
            parse_tool_call('<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>'),
            "2+3",
        )

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
                AgentRLOptions(steps=1, group_size=2, max_new_tokens=4, device="cpu"),
            )
            self.assertTrue(Path(str(metrics["checkpoint"])).exists())

    def test_reward_scores_all_expected_answers(self) -> None:
        from miniscale.agent_env import CalculatorEnv

        env = CalculatorEnv(CalculatorTask("two calculations", "", ("4", "9")))
        self.assertGreater(env.reward("4 and 9"), env.reward("4"))


if __name__ == "__main__":
    unittest.main()
