import unittest

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.data import PretrainDataset, SFTDataset, collate_lm_batch, reservoir_sample_lm_batches


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

    def test_tied_embedding_is_initialized_once_with_zero_padding_row(self) -> None:
        self.assertIs(self.model.lm_head.weight, self.model.embedding.weight)
        self.assertEqual(int(torch.count_nonzero(self.model.embedding.weight[self.config.pad_token_id])), 0)

    def test_residual_projection_initialization_is_depth_scaled(self) -> None:
        torch.manual_seed(7)
        config = MiniScaleConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        model = MiniScaleForCausalLM(config)
        expected = 0.02 / (2 * config.num_hidden_layers) ** 0.5
        self.assertAlmostEqual(
            float(model.layers[0].attention.output.weight.detach().std()), expected, delta=0.001
        )
        self.assertAlmostEqual(float(model.layers[0].mlp.down.weight.detach().std()), expected, delta=0.001)
        self.assertAlmostEqual(
            float(model.layers[0].attention.query.weight.detach().std()), 0.02, delta=0.001
        )

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

    def test_validation_reservoir_is_deterministic_and_spans_stream(self) -> None:
        examples = [
            {
                "input_ids": torch.tensor([index, index + 1]),
                "labels": torch.tensor([index, index + 1]),
            }
            for index in range(100)
        ]

        def selected() -> list[int]:
            batches = reservoir_sample_lm_batches(
                examples,
                batch_size=2,
                batches=2,
                pad_token_id=0,
                seed=17,
            )
            return [int(value) for batch in batches for value in batch["input_ids"][:, 0]]

        self.assertEqual(selected(), selected())
        self.assertTrue(any(index >= 4 for index in selected()))


if __name__ == "__main__":
    unittest.main()
