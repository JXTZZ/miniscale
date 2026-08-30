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


if __name__ == "__main__":
    unittest.main()
