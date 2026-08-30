from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.data import collate_lm_batch
from miniscale.sft_data import (
    IndexedJsonlSFTDataset,
    SFTCorpusIndex,
    fixed_validation_batches,
    truncate_sft_example,
)
from miniscale.sft_data_audit import audit_sft_jsonl
from miniscale.sft_data_prepare import prepare_sft_jsonl
from miniscale.data.sft_quality import response_has_severe_repetition


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def conversation(index: int) -> dict[str, object]:
    return {
        "conversations": [
            {"role": "user", "content": f"question-{index}"},
            {"role": "assistant", "content": f"answer-{index}"},
        ]
    }


class SFTDataTests(unittest.TestCase):
    def test_prepare_is_non_destructive_deduplicated_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            output = root / "prepared.jsonl"
            branded = {
                "conversations": [
                    {"role": "user", "content": "who are you"},
                    {"role": "assistant", "content": "MiniMind"},
                ]
            }
            write_jsonl(source, [branded, branded, conversation(2), conversation(2), conversation(3)])
            source_before = source.read_bytes()
            report = prepare_sft_jsonl(
                source,
                output,
                exclude_patterns=["MiniMind"],
            )
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(report["counts"]["duplicate_rows"], 2)
            self.assertEqual(report["counts"]["excluded_rows"], 1)
            self.assertEqual(report["counts"]["written_rows"], 2)
            self.assertEqual(len(output.read_text().splitlines()), 2)
            self.assertTrue(Path(str(report["manifest"])).is_file())
            with self.assertRaises(FileExistsError):
                prepare_sft_jsonl(source, output)

    def test_prepare_replaces_nested_text_then_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            output = root / "prepared.jsonl"
            branded = {
                "conversations": [
                    {"role": "user", "content": "Who built MINIMIND?"},
                    {
                        "role": "assistant",
                        "content": "MiniMind was built by jingyaogong.",
                        "reasoning_content": "Ask JingYaoGong about minimind.",
                    },
                ]
            }
            already_clean = {
                "conversations": [
                    {"role": "user", "content": "Who built MiniScale?"},
                    {
                        "role": "assistant",
                        "content": "MiniScale was built by LoTusY.",
                        "reasoning_content": "Ask LoTusY about MiniScale.",
                    },
                ]
            }
            write_jsonl(source, [branded, already_clean, conversation(3)])

            report = prepare_sft_jsonl(
                source,
                output,
                replace_patterns=[("MiniMind", "MiniScale"), ("jingyaogong", "LoTusY")],
            )

            prepared = output.read_text(encoding="utf-8")
            self.assertNotIn("minimind", prepared.casefold())
            self.assertNotIn("jingyaogong", prepared.casefold())
            self.assertIn("MiniScale", prepared)
            self.assertIn("LoTusY", prepared)
            self.assertEqual(report["counts"]["replaced_rows"], 1)
            self.assertEqual(report["counts"]["duplicate_rows"], 1)
            self.assertEqual(report["counts"]["written_rows"], 2)
            self.assertEqual(
                report["counts"]["replacement_occurrences"],
                {"MiniMind": 3, "jingyaogong": 2},
            )

    def test_quality_prepare_selects_targets_and_rejects_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            output = root / "quality.jsonl"
            loop = "北京是首都。北京是首都。北京是首都。"
            rows = [
                conversation(1),
                conversation(2),
                {
                    "conversations": [
                        {"role": "user", "content": "loop"},
                        {"role": "assistant", "content": loop},
                    ]
                },
            ]
            write_jsonl(source, rows)
            policy = {
                "version": "sft_quality_policy_v1",
                "seed": 7,
                "max_targets": 2,
                "max_targets_per_conversation": 1,
                "max_prompt_occurrences": 3,
                "max_tool_prompt_occurrences": 3,
                "max_response_occurrences": 3,
                "max_short_response_occurrences": 3,
                "max_identity_response_occurrences": 3,
                "short_response_characters": 16,
                "max_target_tokens": 64,
                "reject_severe_repetition": True,
            }

            report = prepare_sft_jsonl(
                source,
                output,
                quality_policy=policy,
                tokenizer=ByteTokenizer(),
            )

            prepared = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(report["counts"]["selected_targets"], 2)
            self.assertEqual(report["counts"]["rejected_targets"]["severe_repetition"], 1)
            self.assertTrue(all("sft_selection" in row for row in prepared))
            self.assertTrue(response_has_severe_repetition({"content": loop}))
            self.assertTrue(response_has_severe_repetition({
                "content": "传统媒体、传统媒体、传统媒体、传统媒体、传统媒体。"
            }))
            self.assertFalse(response_has_severe_repetition({
                "content": "传统媒体在数字化转型中仍承担公共信息传播职责。"
            }))

    def test_truncation_keeps_context_and_target_prefix(self) -> None:
        input_ids = list(range(30))
        labels = [-100] * 20 + list(range(20, 30))
        encoded = truncate_sft_example(
            input_ids,
            labels,
            max_length=12,
            min_context_tokens=4,
        )
        self.assertEqual(encoded.input_ids, [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
        self.assertEqual(encoded.labels[:4], [-100] * 4)
        self.assertEqual(encoded.labels[4:], list(range(20, 28)))
        self.assertEqual(encoded.dropped_context_tokens, 16)
        self.assertEqual(encoded.dropped_target_tokens, 2)

    def test_index_deduplicates_and_expands_assistant_turns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            multi_turn = {
                "conversations": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
            }
            write_jsonl(path, [multi_turn, multi_turn, conversation(3)])
            index = SFTCorpusIndex.build(
                path,
                validation_fraction=0,
                target_mode="reasoning_and_response",
                deduplicate_exact=True,
            )
            self.assertEqual(index.stats.duplicate_conversations, 1)
            self.assertEqual(index.stats.train_examples, 3)
            dataset = IndexedJsonlSFTDataset(
                index,
                ByteTokenizer(),
                split="train",
                max_length=128,
                min_context_tokens=8,
                target_mode="reasoning_and_response",
            )
            first = dataset[0]
            supervised = ByteTokenizer().decode(
                [label for label in first["labels"].tolist() if label != -100]
            )
            self.assertIn("a1", supervised)
            self.assertNotIn("q1", supervised)
            dataset.close()

    def test_index_honors_selected_target_positions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selected.jsonl"
            row = {
                "conversations": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer one"},
                    {"role": "user", "content": "second"},
                    {"role": "assistant", "content": "answer two"},
                ],
                "sft_selection": {"target_positions": [3]},
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            index = SFTCorpusIndex.build(
                path,
                validation_fraction=0,
                target_mode="response_only",
                destination="train",
            )

            self.assertEqual(index.stats.assistant_messages, 2)
            self.assertEqual(index.stats.selected_assistant_messages, 1)
            self.assertEqual(index.stats.unselected_assistant_messages, 1)
            self.assertEqual(index.stats.train_examples, 1)
            self.assertEqual(list(index.train_message_positions), [3])

    def test_hash_split_and_fixed_validation_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            write_jsonl(path, [conversation(index) for index in range(100)])
            first = SFTCorpusIndex.build(
                path,
                validation_fraction=0.2,
                target_mode="reasoning_and_response",
            )
            second = SFTCorpusIndex.build(
                path,
                validation_fraction=0.2,
                target_mode="reasoning_and_response",
            )
            self.assertEqual(first.train_offsets, second.train_offsets)
            self.assertEqual(first.validation_offsets, second.validation_offsets)
            self.assertGreater(first.stats.validation_examples, 0)
            dataset = IndexedJsonlSFTDataset(
                first,
                ByteTokenizer(),
                split="validation",
                max_length=128,
                min_context_tokens=8,
                target_mode="reasoning_and_response",
            )
            batches_a = fixed_validation_batches(
                dataset, batch_size=2, batches=3, pad_token_id=0, seed=9
            )
            batches_b = fixed_validation_batches(
                dataset, batch_size=2, batches=3, pad_token_id=0, seed=9
            )
            for left, right in zip(batches_a, batches_b, strict=True):
                self.assertTrue(torch.equal(left["input_ids"], right["input_ids"]))
            dataset.close()

    def test_token_weighted_accumulation_matches_combined_batch(self) -> None:
        torch.manual_seed(7)
        model_accumulated = MiniScaleForCausalLM(MiniScaleConfig.smoke())
        model_combined = MiniScaleForCausalLM(MiniScaleConfig.smoke())
        model_combined.load_state_dict(model_accumulated.state_dict())
        tokenizer = ByteTokenizer()
        rows = []
        for answer in ("x", "a much longer answer"):
            ids, labels = tokenizer.encode_sft(
                [{"role": "user", "content": "q"}, {"role": "assistant", "content": answer}],
                target_assistant_index=-1,
            )
            rows.append({"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)})
        micro_batches = [collate_lm_batch([row], tokenizer.pad_token_id) for row in rows]
        counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in micro_batches]
        for batch, count in zip(micro_batches, counts, strict=True):
            loss = model_accumulated(**batch).loss
            assert loss is not None
            (loss * count / sum(counts)).backward()
        combined = collate_lm_batch(rows, tokenizer.pad_token_id)
        combined_loss = model_combined(**combined).loss
        assert combined_loss is not None
        combined_loss.backward()
        for accumulated, reference in zip(
            model_accumulated.parameters(), model_combined.parameters(), strict=True
        ):
            self.assertTrue(torch.allclose(accumulated.grad, reference.grad, atol=1e-6, rtol=1e-5))

    def test_audit_reports_duplicates_patterns_and_token_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sft.jsonl"
            branded = {
                "conversations": [
                    {"role": "user", "content": "who are you"},
                    {"role": "assistant", "content": "MiniMind"},
                ]
            }
            write_jsonl(path, [branded, branded, conversation(2), conversation(3)])
            report = audit_sft_jsonl(
                path,
                ByteTokenizer(),
                max_length=64,
                min_context_tokens=8,
                validation_fraction=0.25,
                sample_size=4,
                identity_patterns=["MiniMind"],
            )
            structure = report["structure"]
            self.assertEqual(structure["duplicate_conversations"], 1)
            self.assertEqual(structure["identity_pattern_conversations"]["MiniMind"], 2)
            self.assertEqual(report["token_sample"]["examples"], 4)


if __name__ == "__main__":
    unittest.main()
