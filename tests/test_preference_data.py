from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from miniscale import ByteTokenizer
from miniscale.dpo_data_audit import audit_dpo_jsonl
from miniscale.preference_data import (
    IndexedPreferenceDataset,
    PreferenceCorpusIndex,
    collate_preference_batch,
    encode_preference_pair,
    fixed_preference_validation_batches,
    parse_preference_pair,
)


def row(prompt: str, chosen: str, rejected: str) -> dict[str, object]:
    return {
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8")


class PreferenceDataTests(unittest.TestCase):
    def test_pair_contract_rejects_mismatched_prompt_and_identical_response(self) -> None:
        mismatched = row("first", "good", "bad")
        mismatched["rejected"][0]["content"] = "second"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "identical prompt"):
            parse_preference_pair(
                mismatched,
                target_mode="reasoning_and_response",
                location="test:1",
            )
        with self.assertRaisesRegex(ValueError, "responses are identical"):
            parse_preference_pair(
                row("prompt", "same", "same"),
                target_mode="reasoning_and_response",
                location="test:2",
            )

    def test_pair_aware_truncation_keeps_identical_context(self) -> None:
        pair = parse_preference_pair(
            row("context-" * 40, "chosen-" * 20, "rejected-" * 30),
            target_mode="reasoning_and_response",
            location="test",
        )
        encoded = encode_preference_pair(
            pair,
            ByteTokenizer(),
            max_length=96,
            min_context_tokens=16,
            target_mode="reasoning_and_response",
        )
        self.assertEqual(
            encoded.chosen_input_ids[: encoded.context_tokens],
            encoded.rejected_input_ids[: encoded.context_tokens],
        )
        self.assertEqual(encoded.chosen_labels[: encoded.context_tokens], [-100] * encoded.context_tokens)
        self.assertEqual(encoded.rejected_labels[: encoded.context_tokens], [-100] * encoded.context_tokens)
        self.assertGreater(encoded.chosen_target_tokens, 0)
        self.assertGreater(encoded.rejected_target_tokens, 0)
        self.assertGreater(encoded.dropped_context_tokens, 0)

    def test_index_deduplicates_and_keeps_prompt_in_one_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dpo.jsonl"
            rows = [row(f"prompt-{index}", "good", "bad") for index in range(40)]
            rows.extend([rows[0], row("prompt-0", "better", "worse")])
            write_jsonl(path, rows)
            index = PreferenceCorpusIndex.build(
                path,
                validation_fraction=0.25,
                target_mode="reasoning_and_response",
            )
            self.assertEqual(index.stats.duplicate_pairs, 1)
            self.assertEqual(index.stats.repeated_prompt_pairs, 1)
            self.assertEqual(index.stats.train_pairs + index.stats.validation_pairs, 41)
            # The two prompt-0 pairs must land in the same split.
            all_offsets = list(index.train_offsets) + list(index.validation_offsets)
            prompt_zero_offsets = []
            with path.open("rb") as source:
                for offset in all_offsets:
                    source.seek(offset)
                    value = json.loads(source.readline())
                    if value["chosen"][0]["content"] == "prompt-0":
                        prompt_zero_offsets.append(offset)
            self.assertEqual(len(prompt_zero_offsets), 2)
            self.assertTrue(
                all(offset in index.train_offsets for offset in prompt_zero_offsets)
                or all(offset in index.validation_offsets for offset in prompt_zero_offsets)
            )

    def test_collation_and_fixed_validation_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dpo.jsonl"
            write_jsonl(path, [row(f"prompt-{index}", "good", "bad") for index in range(30)])
            index = PreferenceCorpusIndex.build(
                path,
                validation_fraction=0.5,
                target_mode="reasoning_and_response",
            )
            dataset = IndexedPreferenceDataset(
                index,
                ByteTokenizer(),
                split="validation",
                max_length=128,
                min_context_tokens=8,
                target_mode="reasoning_and_response",
            )
            collated = collate_preference_batch([dataset[0], dataset[1]], 0)
            self.assertEqual(collated["chosen"]["input_ids"].shape, collated["rejected"]["input_ids"].shape)
            first = fixed_preference_validation_batches(
                dataset, batch_size=2, batches=2, pad_token_id=0, seed=9
            )
            second = fixed_preference_validation_batches(
                dataset, batch_size=2, batches=2, pad_token_id=0, seed=9
            )
            for left, right in zip(first, second, strict=True):
                self.assertTrue(torch.equal(left["chosen"]["input_ids"], right["chosen"]["input_ids"]))
            dataset.close()

    def test_audit_reports_invalid_pairs_and_token_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dpo.jsonl"
            write_jsonl(path, [row("q1", "a", "b"), row("q2", "same", "same")])
            report = audit_dpo_jsonl(
                path,
                ByteTokenizer(),
                max_length=64,
                min_context_tokens=8,
                validation_fraction=0,
                sample_size=2,
            )
            self.assertEqual(report["structure"]["invalid_pairs"], 1)
            self.assertEqual(report["structure"]["invalid_reasons"]["identical_responses"], 1)
            self.assertEqual(report["token_sample"]["pairs"], 1)


if __name__ == "__main__":
    unittest.main()
