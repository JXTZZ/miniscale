import json
from pathlib import Path
import tempfile
import unittest

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM, SentencePieceTokenizer
from miniscale.inference import GenerationOptions, generate_from_checkpoint
from miniscale.tokenizer import train_sentencepiece
from miniscale.training import (
    AgentRLOptions,
    DPOOptions,
    GRPOOptions,
    PretrainOptions,
    SFTOptions,
    run_agent_grpo_jsonl,
    run_dpo_jsonl,
    run_grpo_jsonl,
    run_pretrain_jsonl,
    run_sft_jsonl,
)
from miniscale.training.common import save_checkpoint


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class JsonlTrainingChainTests(unittest.TestCase):
    def test_sentencepiece_training_and_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            write_jsonl(corpus, [{"text": f"你好，MiniScale tokenizer 测试 {index}。"} for index in range(200)])
            model_path = train_sentencepiece(corpus, root / "tokenizer", vocab_size=300, input_sentence_size=200)
            tokenizer = SentencePieceTokenizer(model_path)
            text = "你好，MiniScale！"
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            self.assertEqual(tokenizer.vocab_size, 300)
            config = MiniScaleConfig.smoke()
            config.vocab_size = tokenizer.vocab_size
            checkpoint = save_checkpoint(root / "model.pt", MiniScaleForCausalLM(config), stage="test", step=0, metrics={})
            result = generate_from_checkpoint(
                checkpoint,
                text,
                GenerationOptions(max_new_tokens=1, temperature=0, device="cpu", tokenizer_path=model_path),
            )
            self.assertEqual(result["prompt"], text)

    def test_64m_configuration_is_in_target_range(self) -> None:
        model = MiniScaleForCausalLM(MiniScaleConfig.small_64m())
        self.assertGreater(model.num_parameters, 60_000_000)
        self.assertLess(model.num_parameters, 70_000_000)

    def test_all_training_stages_read_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain = root / "pretrain.jsonl"
            sft = root / "sft.jsonl"
            dpo = root / "dpo.jsonl"
            rl = root / "rl.jsonl"
            agent = root / "agent.jsonl"
            write_jsonl(pretrain, [{"text": "中文预训练文本。" * 20}, {"text": "Another document." * 20}])
            write_jsonl(sft, [{"conversations": [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}]}])
            write_jsonl(dpo, [{
                "chosen": [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}],
                "rejected": [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "5"}],
            }])
            write_jsonl(rl, [{"conversations": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": ""}], "gt": ["4"]}])
            write_jsonl(agent, [{
                "conversations": [
                    {"role": "system", "content": "Use calculate_math."},
                    {"role": "user", "content": "Calculate 2+2"},
                    {"role": "assistant", "content": ""},
                ],
                "gt": ["4"],
            }])
            tokenizer = ByteTokenizer()
            model = MiniScaleForCausalLM(MiniScaleConfig.smoke())
            pretrain_result = run_pretrain_jsonl(
                model, tokenizer, pretrain, root / "pretrain-out",
                PretrainOptions(steps=1, batch_size=1, sequence_length=32, validation_fraction=0, device="cpu"),
            )
            self.assertTrue(Path(str(pretrain_result["checkpoint"])).exists())
            run_sft_jsonl(model, tokenizer, sft, root / "sft-out", SFTOptions(steps=1, batch_size=1, device="cpu"))
            run_dpo_jsonl(model, tokenizer, dpo, root / "dpo-out", DPOOptions(steps=1, batch_size=1, device="cpu"))
            run_grpo_jsonl(
                model, tokenizer, rl, root / "grpo-out",
                GRPOOptions(steps=1, group_size=2, max_new_tokens=4, data_limit=1, device="cpu"),
            )
            agent_result = run_agent_grpo_jsonl(
                model, tokenizer, agent, root / "agent-out",
                AgentRLOptions(steps=1, group_size=2, max_new_tokens=4, data_limit=1, device="cpu"),
            )
            self.assertTrue(Path(str(agent_result["checkpoint"])).exists())


if __name__ == "__main__":
    unittest.main()
