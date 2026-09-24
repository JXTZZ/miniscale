from pathlib import Path
import tempfile
import unittest

from miniscale.training.evaluators.generation_quality import (
    generation_quality_score,
    load_generation_suite,
    repeated_ngram_fraction,
    score_generation,
    summarize_generations,
)


class SFTGenerationQualityTests(unittest.TestCase):
    def test_repeated_ngram_metric_detects_a_loop(self) -> None:
        clean = [1, 2, 3, 4, 5, 6, 7]
        loop = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        self.assertEqual(repeated_ngram_fraction(clean, 4), 0.0)
        self.assertGreater(repeated_ngram_fraction(loop, 4), 0.2)

    def test_generation_rules_and_summary_are_transparent(self) -> None:
        probe = {
            "id": "capital",
            "prompt": "中国的首都是哪里？",
            "required": ["北京"],
            "forbidden": ["上海"],
            "max_tokens": 16,
        }
        good = {
            "generated_tokens": 4,
            **score_generation(probe, "中国的首都是北京。", [1, 2, 3, 4], finish_reason="eos"),
        }
        bad = {
            "generated_tokens": 12,
            **score_generation(
                probe,
                "上海是首都。上海是首都。上海是首都。",
                [1, 2, 3, 4] * 3,
                finish_reason="max_tokens",
            ),
        }
        summary = summarize_generations([good, bad])
        self.assertEqual(summary["generation_eos_rate"], 0.5)
        self.assertEqual(summary["generation_task_pass_rate"], 0.5)
        self.assertEqual(summary["generation_loop_rate"], 0.5)
        self.assertGreater(generation_quality_score(summary), 0)

    def test_suite_requires_unique_ids_and_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.jsonl"
            path.write_text(
                '{"id":"same","prompt":"one"}\n'
                '{"id":"same","prompt":"two"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate ids"):
                load_generation_suite(path)


if __name__ == "__main__":
    unittest.main()
