from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from miniscale import MiniScaleConfig, MiniScaleForCausalLM
from miniscale.huggingface import export_huggingface, to_huggingface_model
from miniscale.training.core.checkpoint import load_checkpoint, read_training_checkpoint, save_checkpoint


class ModelRegressionTests(unittest.TestCase):
    def test_invalid_geometry_fails_at_config_boundary(self):
        for changes in (
            {"num_attention_heads": 0}, {"num_key_value_heads": 0},
            {"hidden_size": 12, "num_attention_heads": 4},
            {"intermediate_size": 0}, {"vocab_size": 0},
            {"rope_theta": float("nan")}, {"rms_norm_eps": 0},
            {"dropout": 1}, {"eos_token_id": 260},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(MiniScaleConfig.smoke(), **changes)

    def test_empty_supervision_is_an_actionable_error(self):
        model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
        ids = torch.tensor([[1, 10, 11]])
        with self.assertRaisesRegex(ValueError, "no supervised"):
            model(ids, labels=torch.full_like(ids, -100))

    def test_last_position_logits_equal_full_forward(self):
        model = MiniScaleForCausalLM(MiniScaleConfig.smoke()).eval()
        ids = torch.tensor([[1, 10, 11, 12]])
        torch.testing.assert_close(model(ids).logits[:, -1:], model(ids, logits_to_keep=1).logits)

    def test_autocast_preserves_bf16_at_attention_boundary(self):
        devices = ["cpu"]
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            devices.append("cuda")
        for device in devices:
            with self.subTest(device=device):
                model = MiniScaleForCausalLM(MiniScaleConfig.smoke()).to(device)
                ids = torch.tensor([[1, 10, 11, 12]], device=device)
                observed = []
                original = F.scaled_dot_product_attention

                def inspect(q, k, v, **kwargs):
                    observed.append((q.dtype, k.dtype, v.dtype))
                    return original(q, k, v, **kwargs)

                with patch("miniscale.model.F.scaled_dot_product_attention", side_effect=inspect):
                    with torch.autocast(device, dtype=torch.bfloat16):
                        result = model(ids, labels=ids)
                    result.loss.backward()
                self.assertEqual(observed, [(torch.bfloat16,) * 3])
                self.assertTrue(torch.isfinite(result.loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_load_checkpoint_preserves_rng_and_deserializes_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
            path = save_checkpoint(Path(directory) / "model.pt", model, stage="test", step=0, metrics={})
            rng = torch.get_rng_state().clone()
            with patch("torch.load", wraps=torch.load) as read:
                restored = load_checkpoint(path, "cuda" if torch.cuda.is_available() else "cpu")
            self.assertEqual(read.call_args.kwargs["map_location"], "cpu")
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name].cpu()))

    def test_training_checkpoint_requires_rng_before_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pt"
            torch.save({"format_version": 2, "config": {}, "model": {}, "optimizer": {},
                        "scheduler": {}, "training_state": {}, "step": 1}, path)
            with self.assertRaisesRegex(ValueError, "rng_state"):
                read_training_checkpoint(path, "cpu")

    def test_hf_mapping_preserves_logits_loss_and_cached_generation(self):
        native = MiniScaleForCausalLM(MiniScaleConfig.smoke()).eval()
        hf = to_huggingface_model(native)
        ids = torch.tensor([[1, 10, 11, 12], [1, 40, 2, 0]])
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        labels = ids.masked_fill(~mask.bool(), -100)
        expected = native(ids, labels=labels, attention_mask=mask)
        actual = hf(ids, labels=labels, attention_mask=mask)
        torch.testing.assert_close(expected.logits[mask.bool()], actual.logits[mask.bool()], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(expected.loss, actual.loss)
        prompt = ids[:1]
        expected_ids = native.generate(prompt, max_new_tokens=4, temperature=0)
        actual_ids = hf.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=4, do_sample=False)
        self.assertTrue(torch.equal(expected_ids, actual_ids))

    def test_hf_export_loads_offline_with_auto_classes(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from miniscale.tokenizer import HuggingFaceTokenizer

        tokenizer_path = Path(__file__).parents[2] / "data/tokenizer/minimind"
        tokenizer = HuggingFaceTokenizer(tokenizer_path)
        model = MiniScaleForCausalLM(replace(MiniScaleConfig.smoke(), vocab_size=tokenizer.vocab_size)).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_checkpoint(root / "model.pt", model, stage="test", step=0, metrics={})
            output = export_huggingface(checkpoint, root / "hf", tokenizer_path)
            restored = AutoModelForCausalLM.from_pretrained(output, local_files_only=True).eval()
            restored_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
            ids = torch.tensor([tokenizer.encode("你好，MiniScale", bos=True)])
            torch.testing.assert_close(model(ids).logits, restored(ids).logits, atol=1e-6, rtol=1e-5)
            self.assertEqual(restored_tokenizer.encode("你好", add_special_tokens=False), tokenizer.encode("你好"))
            self.assertIs(restored.lm_head.weight, restored.model.embed_tokens.weight)
            self.assertTrue((output / "model.safetensors").is_file())
            with self.assertRaises(FileExistsError):
                export_huggingface(checkpoint, output, tokenizer_path)
