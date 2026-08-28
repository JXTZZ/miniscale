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
                GenerationOptions(max_new_tokens=4, temperature=0, device="cpu"),
            )
            self.assertIsInstance(result["response"], str)
            self.assertEqual(result["generated_tokens"], 4)

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


if __name__ == "__main__":
    unittest.main()
