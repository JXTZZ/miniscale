from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import torch

from miniscale import HuggingFaceTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.agent_env import CalculatorTask
from miniscale.cli import main
from miniscale.inference import GenerationOptions, generate_from_checkpoint
from miniscale.tokenizer import load_tokenizer
from miniscale.training.agent_rl import AgentRLOptions, rollout_agent
from miniscale.training.common import save_checkpoint


TOKENIZER_DIR = Path(__file__).parents[1] / "data/tokenizer/minimind"


class HuggingFaceTokenizerTests(unittest.TestCase):
    def test_tokenize_cli_reports_ids_and_round_trip(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            main(["tokenize", "--tokenizer", str(TOKENIZER_DIR), "--text", "你好，MiniScale！"])
        result = json.loads(output.getvalue())
        self.assertTrue(result["round_trip"])
        self.assertEqual(result["token_count"], len(result["token_ids"]))
        self.assertEqual(result["token_count"], len(result["tokens"]))

    def test_minimind_tokenizer_round_trip_and_special_ids(self) -> None:
        tokenizer = HuggingFaceTokenizer(TOKENIZER_DIR)
        text = "你好，MiniScale！😀"
        self.assertEqual(tokenizer.vocab_size, 6400)
        self.assertEqual((tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id), (0, 1, 2))
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertIsInstance(load_tokenizer(TOKENIZER_DIR / "tokenizer.json"), HuggingFaceTokenizer)
        texts = ["你好", "MiniScale"]
        self.assertEqual(tokenizer.encode_batch(texts, bos=True, eos=True), [
            tokenizer.encode(text, bos=True, eos=True) for text in texts
        ])

    def test_chat_template_and_sft_mask_match_minimind(self) -> None:
        tokenizer = HuggingFaceTokenizer(TOKENIZER_DIR)
        messages = [
            {"role": "user", "content": "2+2等于多少？"},
            {"role": "assistant", "content": "4"},
        ]
        rendered = tokenizer.format_messages(messages)
        prompt_ids = tokenizer.encode(rendered, bos=True)
        input_ids, labels = tokenizer.encode_sft(messages)
        supervised = tokenizer.decode([label for label in labels if label != -100], skip_special_tokens=False)
        self.assertIn("<|im_start|>user", rendered)
        self.assertIn("<|im_start|>assistant", rendered)
        self.assertNotEqual(prompt_ids[:2], [tokenizer.bos_token_id, tokenizer.bos_token_id])
        self.assertEqual(len(input_ids), len(labels))
        self.assertIn("4<|im_end|>", supervised)
        self.assertNotIn("2+2", supervised)
        observation = tokenizer.format_tool_observation("4", assistant_closed=True)
        self.assertIn("<tool_response>\n4\n</tool_response>", observation)
        self.assertIn("<|im_start|>assistant", observation)

    def test_sft_mask_can_target_one_turn_and_exclude_reasoning(self) -> None:
        tokenizer = HuggingFaceTokenizer(TOKENIZER_DIR)
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer", "reasoning_content": "first reasoning"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer", "reasoning_content": "private reasoning"},
        ]
        _, reasoning_labels = tokenizer.encode_sft(
            messages,
            target_mode="reasoning_and_response",
            target_assistant_index=-1,
        )
        reasoning_target = tokenizer.decode(
            [label for label in reasoning_labels if label != -100], skip_special_tokens=False
        )
        self.assertIn("private reasoning", reasoning_target)
        self.assertIn("second answer", reasoning_target)
        self.assertNotIn("first answer", reasoning_target)

        _, response_labels = tokenizer.encode_sft(
            messages,
            target_mode="response_only",
            target_assistant_index=-1,
        )
        response_target = tokenizer.decode(
            [label for label in response_labels if label != -100], skip_special_tokens=False
        )
        self.assertIn("second answer", response_target)
        self.assertNotIn("private reasoning", response_target)
        self.assertNotIn("first answer", response_target)

    def test_generation_loads_tokenizer_directory(self) -> None:
        tokenizer = HuggingFaceTokenizer(TOKENIZER_DIR)
        config = MiniScaleConfig.smoke()
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = save_checkpoint(
                Path(directory) / "model.pt",
                MiniScaleForCausalLM(config),
                stage="test",
                step=0,
                metrics={},
            )
            result = generate_from_checkpoint(
                checkpoint,
                "你好",
                GenerationOptions(
                    max_new_tokens=1,
                    temperature=0,
                    device="cpu",
                    tokenizer_path=TOKENIZER_DIR,
                ),
            )
        self.assertEqual(result["prompt"], "你好")

    def test_agent_observation_uses_minimind_template(self) -> None:
        tokenizer = HuggingFaceTokenizer(TOKENIZER_DIR)
        config = MiniScaleConfig.smoke()
        config.vocab_size = tokenizer.vocab_size
        responses = [
            '<tool_call>{"name":"calculate_math","arguments":{"expression":"2+2"}}</tool_call>',
            "4",
        ]
        trajectory = rollout_agent(
            MiniScaleForCausalLM(config),
            tokenizer,
            CalculatorTask("计算2+2", "2+2", "4"),
            AgentRLOptions(max_turns=2, max_new_tokens=80, device="cpu"),
            torch.device("cpu"),
            response_fn=lambda _transcript, turn: responses[turn],
        )
        self.assertIn("<tool_response>\n4\n</tool_response>", trajectory.transcript)
        self.assertGreater(trajectory.observation_tokens, 0)
        for start, end in trajectory.observation_ranges:
            self.assertEqual(sum(trajectory.action_mask[start:end]), 0)


if __name__ == "__main__":
    unittest.main()
