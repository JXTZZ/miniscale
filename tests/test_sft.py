from pathlib import Path
import tempfile
import unittest

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.sft import SFTOptions, run_sft


class SFTTests(unittest.TestCase):
    def test_sft_stage_writes_checkpoint(self) -> None:
        conversations = [
            [
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": "hello"},
            ],
            [
                {"role": "user", "content": "2+2"},
                {"role": "assistant", "content": "4"},
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_sft(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                conversations,
                directory,
                SFTOptions(steps=2, batch_size=2, device="cpu"),
            )
            self.assertTrue(Path(str(metrics["checkpoint"])).exists())
            self.assertGreater(float(metrics["loss"]), 0.0)

    def test_long_conversation_is_limited_to_model_context(self) -> None:
        conversations = [[
            {"role": "user", "content": "x" * 300},
            {"role": "assistant", "content": "kept"},
        ]]
        with tempfile.TemporaryDirectory() as directory:
            run_sft(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()),
                ByteTokenizer(),
                conversations,
                directory,
                SFTOptions(steps=1, device="cpu"),
            )


if __name__ == "__main__":
    unittest.main()
