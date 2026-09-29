from pathlib import Path
import tempfile
import unittest

from miniscale import MiniScaleConfig, MiniScaleForCausalLM
from miniscale.inference import GenerationOptions, generate_from_checkpoint
from miniscale.training.common import save_checkpoint


class InferenceTests(unittest.TestCase):
    def test_checkpoint_generates_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(
                Path(directory) / "model.pt",
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                stage="test",
                step=0,
                metrics={},
            )
            result = generate_from_checkpoint(
                checkpoint,
                "2+3?",
                GenerationOptions(
                    max_new_tokens=4,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=4,
                    seed=7,
                    device="cpu",
                ),
            )
            self.assertIsInstance(result["response"], str)
            self.assertEqual(result["generated_tokens"], 4)
            self.assertEqual(result["top_p"], 0.9)
            self.assertEqual(result["finish_reason"], "max_tokens")

    def test_missing_checkpoint_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            generate_from_checkpoint("does-not-exist.pt", "hello")

    def test_calculator_mode_runs_bounded_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(
                Path(directory) / "model.pt",
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                stage="agent_rl",
                step=0,
                metrics={},
            )
            result = generate_from_checkpoint(
                checkpoint,
                "Calculate 2+3",
                GenerationOptions(
                    max_new_tokens=4,
                    max_turns=2,
                    temperature=0,
                    calculator=True,
                    device="cpu",
                ),
            )
            self.assertIn("tool_calls", result)
            self.assertLessEqual(result["turns"], 2)
            self.assertIsInstance(result["transcript"], str)


class CalculatorInferenceRegressionTests(unittest.TestCase):
    def test_calculator_seed_and_decoding_settings_are_effective(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(Path(directory) / "model.pt",
                                         MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                                         stage="test", step=0, metrics={})
            options = GenerationOptions(max_new_tokens=4, max_turns=1, calculator=True,
                                        seed=9, top_p=0.9, repetition_penalty=1.2,
                                        no_repeat_ngram_size=2, device="cpu")
            first = generate_from_checkpoint(checkpoint, "Calculate 2+3", options)
            torch.rand(20)
            second = generate_from_checkpoint(checkpoint, "Calculate 2+3", options)
            self.assertEqual(first, second)
            self.assertEqual(first["top_p"], 0.9)
            options.top_p = 0
            with self.assertRaisesRegex(ValueError, "top_p"):
                generate_from_checkpoint(checkpoint, "Calculate 2+3", options)


if __name__ == "__main__":
    unittest.main()
