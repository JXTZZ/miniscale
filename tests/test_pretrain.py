from pathlib import Path
import tempfile
import unittest

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.common import load_checkpoint
from miniscale.training.pretrain import PretrainOptions, run_pretrain


class PretrainTests(unittest.TestCase):
    def test_pretrain_writes_loadable_checkpoint(self) -> None:
        torch.manual_seed(1)
        model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
        tokenizer = ByteTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_pretrain(
                model,
                tokenizer,
                ["small models make training loops testable " * 4],
                directory,
                PretrainOptions(steps=2, batch_size=1, sequence_length=48, device="cpu"),
            )
            checkpoint = Path(str(metrics["checkpoint"]))
            self.assertTrue(checkpoint.exists())
            restored = load_checkpoint(checkpoint)
            ids = torch.tensor([tokenizer.encode("hello", bos=True)])
            self.assertEqual(restored(ids).logits.shape[:2], ids.shape)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))


if __name__ == "__main__":
    unittest.main()
