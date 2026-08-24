import unittest

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.data import PretrainDataset, SFTDataset, collate_lm_batch


class ModelAndDataTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.tokenizer = ByteTokenizer()
        self.config = MiniScaleConfig.smoke()
        self.model = MiniScaleForCausalLM(self.config)

    def test_tokenizer_round_trip(self) -> None:
        text = "MiniScale 你好"
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_forward_and_backward(self) -> None:
        ids = torch.randint(4, self.config.vocab_size, (2, 12))
        output = self.model(ids, labels=ids)
        self.assertEqual(output.logits.shape, (2, 12, self.config.vocab_size))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(self.model.embedding.weight.grad)

    def test_causality(self) -> None:
        self.model.eval()
        first = torch.tensor([[1, 10, 11, 12, 13]])
        second = torch.tensor([[1, 10, 11, 99, 98]])
        with torch.no_grad():
            first_logits = self.model(first).logits[:, :3]
            second_logits = self.model(second).logits[:, :3]
        torch.testing.assert_close(first_logits, second_logits)

    def test_sft_masks_non_assistant_tokens(self) -> None:
        dataset = SFTDataset(
            [[{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]],
            self.tokenizer,
        )
        example = dataset[0]
        self.assertGreater(int((example["labels"] == -100).sum()), 0)
        self.assertGreater(int((example["labels"] != -100).sum()), 0)

    def test_collate_pads_labels_with_ignore_index(self) -> None:
        dataset = PretrainDataset(["abc", "a much longer example"], self.tokenizer, sequence_length=8)
        batch = collate_lm_batch([dataset[0], dataset[-1]])
        self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
        self.assertEqual(batch["attention_mask"].shape, batch["labels"].shape)


if __name__ == "__main__":
    unittest.main()
