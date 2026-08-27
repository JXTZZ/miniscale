from pathlib import Path
import json
import tempfile
import unittest

from miniscale import MiniScaleConfig, MiniScaleForCausalLM
from miniscale.evaluation import evaluate_rl_checkpoints
from miniscale.rl_data import prompt_in_validation
from miniscale.tokenizer import load_tokenizer
from miniscale.training.common import save_checkpoint


class RLEvaluationTests(unittest.TestCase):
    def test_comparison_report_records_input_identities(self) -> None:
        tokenizer_path = Path("data/tokenizer/minimind")
        tokenizer = load_tokenizer(tokenizer_path)
        config = MiniScaleConfig.smoke()
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        prompt = next(
            f"question-{index}" for index in range(1000)
            if prompt_in_validation(f"question-{index}", 0.5)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "rl.jsonl"
            data.write_text(json.dumps({
                "conversations": [{"role": "user", "content": prompt}],
                "gt": ["4"],
            }) + "\n", encoding="utf-8")
            checkpoint = save_checkpoint(
                root / "model.pt",
                MiniScaleForCausalLM(config),
                stage="test",
                step=0,
                metrics={},
            )
            report_path = root / "evaluation.json"
            report = evaluate_rl_checkpoints(
                [checkpoint],
                data,
                tokenizer_path,
                validation_fraction=0.5,
                prompts=1,
                max_new_tokens=1,
                device="cpu",
                output_path=report_path,
            )
            self.assertTrue(report_path.is_file())
            self.assertEqual(len(report["results"]), 1)
            self.assertIn("sha256", report["results"][0]["identity"])


if __name__ == "__main__":
    unittest.main()
