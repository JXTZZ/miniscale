from pathlib import Path
import tempfile
import unittest

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.grpo import (
    GRPOOptions,
    RLTask,
    grpo_objective,
    math_reward,
    normalize_group_rewards,
    run_grpo,
)


class GRPOTests(unittest.TestCase):
    def test_group_advantages_are_centered(self) -> None:
        advantages = normalize_group_rewards(torch.tensor([1.0, 2.0, 4.0, 8.0]), group_size=2)
        self.assertTrue(torch.allclose(advantages.view(2, 2).mean(1), torch.zeros(2), atol=1e-5))

    def test_objective_has_gradients_and_masks_prompt(self) -> None:
        policy = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
        old = torch.zeros_like(policy)
        reference = torch.zeros_like(policy)
        mask = torch.tensor([[0.0, 1.0, 1.0]])
        loss, stats = grpo_objective(
            policy, old, reference, mask, torch.tensor([1.0]), clip_epsilon=0.2, beta=0.01
        )
        loss.backward()
        self.assertEqual(float(policy.grad[0, 0]), 0.0)
        self.assertIn("kl", stats)

    def test_math_reward_prefers_exact_answer(self) -> None:
        self.assertGreater(math_reward("The answer is 5", "5"), math_reward("The answer is 4", "5"))

    def test_grpo_stage_writes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_grpo(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                [RLTask("2+2?", "4")],
                directory,
                GRPOOptions(steps=1, group_size=2, max_new_tokens=4, device="cpu"),
            )
            self.assertTrue(Path(str(metrics["checkpoint"])).exists())


if __name__ == "__main__":
    unittest.main()
