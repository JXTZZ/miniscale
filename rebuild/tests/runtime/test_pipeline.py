import json
from pathlib import Path
import tempfile
import unittest

from miniscale.cli import environment_report
from miniscale.pipeline import run_training_pipeline


class PipelineTests(unittest.TestCase):
    def test_environment_report_uses_project_interpreter(self) -> None:
        report = environment_report()
        self.assertTrue(str(report["python"]).startswith("3.12."))

    def test_end_to_end_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_training_pipeline(directory, device="cpu")
            manifest = Path(str(result["manifest"]))
            self.assertTrue(manifest.exists())
            self.assertEqual(
                list(result["stages"]),
                ["pretrain", "sft", "rl", "agent_rl"],
            )
            self.assertEqual(json.loads(manifest.read_text())["kind"], "integration-smoke-test")
            for checkpoint in ("pretrain.pt", "sft.pt", "rl.pt", "agent_rl.pt"):
                self.assertTrue((Path(directory) / checkpoint).exists())


if __name__ == "__main__":
    unittest.main()
