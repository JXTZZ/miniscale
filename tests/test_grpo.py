from pathlib import Path
import tempfile
import unittest
import json

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.common import load_checkpoint
from miniscale.training.grpo import (
    GRPOOptions,
    RLTask,
    grpo_objective,
    math_reward,
    normalize_group_rewards,
    run_grpo,
)
from miniscale.rewards import score_math_answer
from miniscale.rl_data import build_rl_corpus


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
        self.assertGreater(math_reward("answers: 5 and 7", ("5", "7")), math_reward("answer: 5", ("5", "7")))

    def test_math_reward_penalizes_answer_stuffing_and_tool_arguments(self) -> None:
        exact = score_math_answer("The answer is 5", "5")
        stuffed = score_math_answer("Possible answers: 1 2 3 4 5 6", "5")
        tool_only = score_math_answer(
            '<tool_call>{"name":"calculator","arguments":{"expression":"2+3"}}</tool_call>',
            "5",
        )
        self.assertTrue(exact.exact)
        self.assertGreater(exact.total, stuffed.total)
        self.assertEqual(tool_only.correctness, 0.0)
        self.assertTrue(score_math_answer("20,758,280", "20758280").exact)

    def test_corpus_split_is_stable_and_limit_is_global(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rl.jsonl"
            rows = [
                {"conversations": [{"role": "user", "content": f"q{index}"}], "gt": [str(index)]}
                for index in range(20)
            ]
            rows.append(rows[0])
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            first = build_rl_corpus(path, validation_fraction=0.2, train_limit=5, seed=7)
            second = build_rl_corpus(path, validation_fraction=0.2, train_limit=5, seed=7)
            self.assertEqual(first.train, second.train)
            self.assertEqual(first.validation, second.validation)
            self.assertEqual(first.stats.duplicate_rows, 1)
            self.assertEqual(len(first.train), 5)

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
            payload = torch.load(metrics["checkpoint"], map_location="cpu", weights_only=False)
            self.assertIn("optimizer", payload)
            self.assertIn("resume_signature", payload["training_state"])

    def test_clipping_is_observable_after_policy_moves(self) -> None:
        policy = torch.tensor([[0.5, -0.5]], requires_grad=True)
        old = torch.zeros_like(policy)
        reference = torch.zeros_like(policy)
        loss, stats = grpo_objective(
            policy,
            old,
            reference,
            torch.ones_like(policy),
            torch.tensor([1.0]),
            clip_epsilon=0.2,
            beta=0.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(stats["clip_fraction"]), 0.0)

    def test_resume_from_periodic_checkpoint_matches_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            options = GRPOOptions(
                steps=2,
                group_size=2,
                max_new_tokens=4,
                policy_epochs=1,
                save_every=1,
                log_every=1,
                device="cpu",
            )
            tokenizer = ByteTokenizer()
            run_grpo(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                tokenizer,
                [RLTask("2+2?", "4")],
                output,
                options,
            )
            expected = torch.load(output / "rl.pt", map_location="cpu", weights_only=False)["model"]
            resume_path = output / "checkpoints/step_00000001.pt"
            run_grpo(
                load_checkpoint(resume_path),
                tokenizer,
                [RLTask("2+2?", "4")],
                output,
                GRPOOptions(
                    steps=2,
                    group_size=2,
                    max_new_tokens=4,
                    policy_epochs=1,
                    save_every=1,
                    log_every=1,
                    device="cpu",
                    resume_from=resume_path,
                ),
            )
            resumed = torch.load(output / "rl.pt", map_location="cpu", weights_only=False)["model"]
            for name in expected:
                self.assertTrue(torch.equal(expected[name], resumed[name]), name)


if __name__ == "__main__":
    unittest.main()
