"""Offline checks for the MiniMind downloader's CLI file selection."""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_minimind


class DownloadProfileTests(unittest.TestCase):
    def run_downloader(self, *options):
        names = set(download_minimind.PROFILES['mini']) | set(download_minimind.PROFILES['full'])
        listing = {'Data': {'Files': [{'Name': name, 'Size': 1} for name in names]}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with (
                patch.object(download_minimind, 'urlopen', return_value=io.BytesIO(json.dumps(listing).encode())),
                patch.object(download_minimind, 'download', side_effect=lambda item, _: {'path': item['Name']}),
                patch.object(sys, 'argv', ['download_minimind.py', '--output', str(output), *options]),
            ):
                download_minimind.main()
            manifest = json.loads((output / 'download_manifest.json').read_text())
        return {item['path'] for item in manifest['files']}

    def test_full_profile_selects_full_corpora_and_common_files(self):
        selected = self.run_downloader('--profile', 'full')
        self.assertEqual(selected, {
            'pretrain_t2t.jsonl', 'sft_t2t.jsonl', 'dpo.jsonl', 'rlaif.jsonl',
            'agent_rl_math.jsonl', 'agent_rl.jsonl',
        })

    def test_default_profile_keeps_mini_corpora(self):
        selected = self.run_downloader()
        self.assertEqual(selected, {
            'pretrain_t2t_mini.jsonl', 'sft_t2t_mini.jsonl', 'dpo.jsonl', 'rlaif.jsonl',
            'agent_rl_math.jsonl', 'agent_rl.jsonl',
        })


if __name__ == '__main__':
    unittest.main()
