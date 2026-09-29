from functools import partial
from pathlib import Path
import json
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from miniscale import ByteTokenizer
from miniscale.data import JsonlPretrainDataset, collate_lm_batch
from miniscale.data.sft import SFTCorpusIndex, IndexedJsonlSFTDataset
from miniscale.data.preference import PreferenceCorpusIndex, IndexedPreferenceDataset, collate_preference_batch


class WorkerRegressionTests(unittest.TestCase):
    def test_multiworker_shuffle_changes_epoch_and_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("".join(json.dumps({"text": f"document {i} " * 3}) + "\n" for i in range(20)))

            def epochs(persistent=False):
                dataset = JsonlPretrainDataset(path, ByteTokenizer(), 16, shuffle_buffer_size=8, seed=17)
                loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=persistent,
                                    generator=torch.Generator().manual_seed(17))
                return [[tuple(row.tolist()) for batch in loader for row in batch["input_ids"]] for _ in range(2)]

            for persistent in (False, True):
                with self.subTest(persistent_workers=persistent):
                    first, second = epochs(persistent), epochs(persistent)
                    self.assertEqual(first, second)
                    self.assertNotEqual(first[0], first[1])
                    self.assertEqual(sorted(first[0]), sorted(first[1]))

    def test_preopened_indexed_files_are_private_to_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            rows = []
            for index in range(24):
                prompt = [{"role": "user", "content": f"q{index} " + "x" * 3000}]
                chosen = prompt + [{"role": "assistant", "content": f"answer {index}"}]
                rejected = prompt + [{"role": "assistant", "content": f"wrong {index}"}]
                rows.append({"conversations": chosen, "chosen": chosen, "rejected": rejected})
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            for preference in (False, True):
                with self.subTest(preference=preference):
                    index_type = PreferenceCorpusIndex if preference else SFTCorpusIndex
                    dataset_type = IndexedPreferenceDataset if preference else IndexedJsonlSFTDataset
                    index = index_type.build(path, validation_fraction=0, target_mode="response_only")
                    dataset = dataset_type(index, ByteTokenizer(), split="train", max_length=64,
                                           min_context_tokens=4, target_mode="response_only")
                    collate = partial(collate_preference_batch if preference else collate_lm_batch, pad_token_id=0)
                    expected = list(DataLoader(dataset, batch_size=2, collate_fn=collate))
                    # The parent handle is open here. Each forked worker must reopen it.
                    actual = list(DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate))
                    for left, right in zip(expected, actual, strict=True):
                        if preference:
                            left, right = left["chosen"], right["chosen"]
                        self.assertTrue(torch.equal(left["input_ids"], right["input_ids"]))
                    dataset.close()
