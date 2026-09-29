from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.training.pretrain import PretrainOptions, run_pretrain_jsonl
from miniscale.training.sft import SFTOptions, run_sft_jsonl
from miniscale.training.dpo import DPOOptions, run_dpo_jsonl
from miniscale.training.grpo import GRPOOptions, RLTask, run_grpo
from miniscale.training.agent_rl import AgentRLOptions, run_agent_grpo
from miniscale.agent_env import CalculatorTask
from miniscale.training.core.rl_runtime import optimize_policy_epochs
from miniscale.training.grpo.objective import sequence_token_log_probs


class UpdateBoundaryTests(unittest.TestCase):
    def test_emergency_rewinds_dropout_rng_before_replaying_micro_batches(self):
        for stage in ("pretrain", "sft", "dpo"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / "data.jsonl"
                rows = []
                for index in range(4):
                    prompt = [{"role": "user", "content": f"q{index}"}]
                    rows.append({"text": f"document {index} " * 10,
                                 "conversations": prompt + [{"role": "assistant", "content": "answer"}],
                                 "chosen": prompt + [{"role": "assistant", "content": "good"}],
                                 "rejected": prompt + [{"role": "assistant", "content": "bad"}]})
                data.write_text("".join(json.dumps(row) + "\n" for row in rows))
                common = dict(steps=3, batch_size=1, gradient_accumulation_steps=2,
                              validation_fraction=0, generation_every=0, save_every=0,
                              warmup_steps=0, device="cpu")
                if stage == "pretrain":
                    options = PretrainOptions(**common, sequence_length=16, shuffle_buffer_size=4)
                    run = run_pretrain_jsonl
                elif stage == "sft":
                    options = SFTOptions(**common, max_length=96, min_context_tokens=4, generation_suite=None)
                    run = run_sft_jsonl
                else:
                    options = DPOOptions(**common, max_length=96, min_context_tokens=4)
                    run = run_dpo_jsonl
                torch.manual_seed(123)
                initial = MiniScaleForCausalLM(replace(MiniScaleConfig.smoke(), dropout=0.2))
                expected, interrupted = deepcopy(initial), deepcopy(initial)
                run(expected, ByteTokenizer(), data, root / "expected", options)
                forward = MiniScaleForCausalLM.forward
                calls = 0

                def fail_after_forward(candidate, *args, **kwargs):
                    nonlocal calls
                    result = forward(candidate, *args, **kwargs)
                    if candidate is not interrupted:
                        return result
                    calls += 1
                    if calls == 4:  # one full update, then dropout + partial accumulation
                        raise KeyboardInterrupt
                    return result

                with patch.object(MiniScaleForCausalLM, "forward", fail_after_forward):
                    with self.assertRaises(KeyboardInterrupt):
                        run(interrupted, ByteTokenizer(), data, root / "actual", options)
                checkpoint = root / "actual/checkpoints/emergency_step_00000001.pt"
                self.assertTrue(checkpoint.is_file())
                resumed = deepcopy(initial)
                run(resumed, ByteTokenizer(), data, root / "actual", replace(options, resume_from=checkpoint))
                for name, value in expected.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.state_dict()[name]), f"{stage}: {name}")

    def test_optimizer_interruption_does_not_publish_inconsistent_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            data.write_text(json.dumps({"text": "training text " * 20}) + "\n")
            original = torch.optim.AdamW.step

            def interrupted_step(optimizer, *args, **kwargs):
                original(optimizer, *args, **kwargs)
                raise KeyboardInterrupt

            with patch.object(torch.optim.AdamW, "step", interrupted_step):
                with self.assertWarnsRegex(RuntimeWarning, "optimizer mutation"), self.assertRaises(KeyboardInterrupt):
                    run_pretrain_jsonl(
                        MiniScaleForCausalLM(MiniScaleConfig.smoke()), ByteTokenizer(), data, root / "run",
                        PretrainOptions(steps=2, sequence_length=16, gradient_accumulation_steps=1,
                                        validation_fraction=0, generation_every=0, save_every=0, device="cpu"),
                    )
            self.assertFalse(list((root / "run").rglob("*.pt")))

    def test_rl_rollout_interruption_preserves_task_cursor_and_rng(self):
        for agent in (False, True):
            with self.subTest(agent=agent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                torch.manual_seed(11)
                initial = MiniScaleForCausalLM(MiniScaleConfig.smoke())
                expected, interrupted = deepcopy(initial), deepcopy(initial)
                if agent:
                    run = run_agent_grpo
                    tasks = [CalculatorTask(f"compute {i}+1", f"{i}+1", str(i+1)) for i in range(2)]
                    options = AgentRLOptions(steps=2, group_size=2, max_new_tokens=2, max_turns=1,
                                             policy_epochs=1, save_every=0, device="cpu")
                else:
                    run = run_grpo
                    tasks = [RLTask(f"compute {i}+1", str(i+1)) for i in range(2)]
                    options = GRPOOptions(steps=2, group_size=2, max_new_tokens=2,
                                          policy_epochs=1, save_every=0, device="cpu")
                run(expected, ByteTokenizer(), tasks, root / "expected", options)
                generate = interrupted.generate
                calls = 0

                def fail_after_generation(*args, **kwargs):
                    nonlocal calls
                    result = generate(*args, **kwargs)
                    calls += 1
                    if calls == 3:
                        raise KeyboardInterrupt
                    return result

                interrupted.generate = fail_after_generation
                with self.assertRaises(KeyboardInterrupt):
                    run(interrupted, ByteTokenizer(), tasks, root / "actual", options)
                checkpoint = root / "actual/checkpoints/emergency_step_00000001.pt"
                payload = torch.load(checkpoint, weights_only=False)
                self.assertEqual(payload["training_state"]["task_cursor"], 1)
                resumed = deepcopy(initial)
                run(resumed, ByteTokenizer(), tasks, root / "actual", replace(options, resume_from=checkpoint))
                for name, value in expected.state_dict().items():
                    self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)

    def test_rl_policy_ratio_does_not_include_dropout_noise(self):
        model = MiniScaleForCausalLM(replace(MiniScaleConfig.smoke(), dropout=0.5)).eval()
        ids = torch.tensor([[1, 10, 11, 12]])
        mask = torch.ones_like(ids)
        with torch.no_grad():
            old = sequence_token_log_probs(model, ids, mask)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        update = optimize_policy_epochs(
            model, optimizer, scheduler, input_ids=ids, attention_mask=mask,
            action_mask=torch.ones_like(old), old_log_probs=old, reference_log_probs=old,
            advantages=torch.ones(1), policy_epochs=1, clip_epsilon=0.2, beta=0.01,
            grad_clip=1, device=torch.device("cpu"), autocast_dtype=None, step=1,
        )
        self.assertAlmostEqual(update.metrics["loss"], -1.0, places=6)
        self.assertEqual(update.metrics["kl"], 0.0)
        self.assertEqual(update.metrics["clip_fraction"], 0.0)

    def test_pretrain_multiworker_resume_crosses_epoch_with_dropout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            data.write_text(''.join(json.dumps({"text": f"abc{i}"}) + '\n' for i in range(4)))
            options = PretrainOptions(steps=7, batch_size=1, sequence_length=6, num_workers=2,
                                      gradient_accumulation_steps=1, validation_fraction=0,
                                      generation_every=0, save_every=3, shuffle_buffer_size=3, device="cpu")
            initial = MiniScaleForCausalLM(replace(MiniScaleConfig.smoke(), dropout=0.2))
            expected, resumed = deepcopy(initial), deepcopy(initial)
            run_pretrain_jsonl(expected, ByteTokenizer(), data, root / "expected", options)
            checkpoint = root / "expected/checkpoints/step_00000003.pt"
            run_pretrain_jsonl(resumed, ByteTokenizer(), data, root / "resumed", replace(options, resume_from=checkpoint))
            for name, value in expected.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[name]), name)

    def test_dedicated_validation_uses_every_training_document(self):
        from miniscale.data.pretrain import is_validation_text
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text = next(f"training document {i}" for i in range(100) if is_validation_text(f"training document {i}", 0.5))
            data, validation = root / "data.jsonl", root / "validation.jsonl"
            data.write_text(json.dumps({"text": text}) + '\n')
            validation.write_text(json.dumps({"text": "independent validation data"}) + '\n')
            result = run_pretrain_jsonl(
                MiniScaleForCausalLM(MiniScaleConfig.smoke()), ByteTokenizer(), data, root / "run",
                PretrainOptions(steps=1, batch_size=1, sequence_length=8, gradient_accumulation_steps=1,
                                validation_fraction=0.5, generation_every=0, save_every=0, device="cpu"),
                validation_path=validation,
            )
            self.assertEqual(result["tokens_seen"], 8)

    def test_rl_partial_policy_epochs_cannot_be_saved_as_completed_rollout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = torch.optim.AdamW.step
            calls = 0

            def fail_on_second_epoch(optimizer, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise FloatingPointError("injected second policy epoch failure")
                return original(optimizer, *args, **kwargs)

            with patch.object(torch.optim.AdamW, "step", fail_on_second_epoch):
                with self.assertWarnsRegex(RuntimeWarning, "optimizer mutation"), self.assertRaises(FloatingPointError):
                    run_grpo(MiniScaleForCausalLM(MiniScaleConfig.smoke()), ByteTokenizer(),
                             [RLTask("2+2?", "4")], root,
                             GRPOOptions(steps=1, group_size=2, max_new_tokens=2,
                                         policy_epochs=2, save_every=0, device="cpu"))
            self.assertEqual(calls, 2)
            self.assertFalse(list(root.rglob("emergency*.pt")))

    def test_implementation_change_cannot_bypass_strict_resume(self):
        from miniscale.training.pretrain.resume import _validate_resume_signature
        saved = {"signature_version": 2, "implementation_version": 2}
        current = {"signature_version": 2, "implementation_version": 3}
        with self.assertRaisesRegex(ValueError, "implementation_version"):
            _validate_resume_signature(saved, current, allow_legacy=True)
