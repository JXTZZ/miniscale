import json
from pathlib import Path
import tempfile
import unittest

from miniscale import ByteTokenizer
from miniscale.data_audit import audit_pretrain_jsonl, save_data_audit


class DataAuditTests(unittest.TestCase):
    def test_full_audit_reports_quality_split_and_packing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "pretrain.jsonl"
            corpus.write_text(
                "\n"
                + json.dumps({"text": "abc"}) + "\n"
                + json.dumps({"text": "abc"}) + "\n"
                + json.dumps({"text": ""}) + "\n"
                + json.dumps({}) + "\n"
                + json.dumps([1, 2, 3]) + "\n"
                + "{invalid json\n",
                encoding="utf-8",
            )
            report = audit_pretrain_jsonl(
                corpus,
                ByteTokenizer(),
                validation_fraction=0,
                sequence_length=5,
                tokenizer_batch_size=1,
            )

            counts = report["counts"]
            self.assertEqual(counts["rows"], 7)
            self.assertEqual(counts["valid_documents"], 2)
            self.assertEqual(counts["exact_duplicate_rows"], 1)
            self.assertEqual(counts["invalid_json_rows"], 1)
            self.assertEqual(counts["non_object_rows"], 1)
            self.assertEqual(counts["missing_or_empty_text_rows"], 2)
            self.assertEqual(report["tokens"]["train"]["stream_tokens"], 10)
            self.assertEqual(report["tokens"]["train"]["packed_blocks"], 2)
            self.assertFalse(report["quality"]["training_compatible"])

            target = save_data_audit(report, root / "reports/data.json")
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
