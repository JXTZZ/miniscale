from __future__ import annotations

from copy import deepcopy
import unittest

import torch
import torch.nn.functional as F

from miniscale import ByteTokenizer, MiniScaleConfig, MiniScaleForCausalLM
from miniscale.preference_data import (
    PreferencePair,
    collate_preference_batch,
    encode_preference_pair,
)
from miniscale.training.dpo_objective import (
    completion_log_probability,
    concatenated_completion_log_probabilities,
    dpo_loss,
)


def example(prompt: str, chosen: str, rejected: str) -> dict[str, dict[str, torch.Tensor]]:
    tokenizer = ByteTokenizer()
    pair = PreferencePair(
        prompt=[{"role": "user", "content": prompt}],
        chosen=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        rejected=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
    )
    encoded = encode_preference_pair(
        pair,
        tokenizer,
        max_length=128,
        min_context_tokens=8,
        target_mode="reasoning_and_response",
    )
    return {
        "chosen": {
            "input_ids": torch.tensor(encoded.chosen_input_ids),
            "labels": torch.tensor(encoded.chosen_labels),
        },
        "rejected": {
            "input_ids": torch.tensor(encoded.rejected_input_ids),
            "labels": torch.tensor(encoded.rejected_labels),
        },
    }


class DPOObjectiveTests(unittest.TestCase):
    def test_loss_matches_direct_formula(self) -> None:
        policy_chosen = torch.tensor([-2.0, -3.0])
        policy_rejected = torch.tensor([-4.0, -2.0])
        reference_chosen = torch.tensor([-2.5, -2.5])
        reference_rejected = torch.tensor([-3.5, -2.5])
        loss, accuracy = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=0.2,
        )
        logits = 0.2 * (
            (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
        )
        self.assertTrue(torch.equal(loss, -F.logsigmoid(logits).mean()))
        self.assertTrue(torch.equal(accuracy, (logits > 0).float().mean()))

    def test_concatenated_scoring_matches_independent_scoring(self) -> None:
        torch.manual_seed(7)
        model = MiniScaleForCausalLM(MiniScaleConfig.smoke()).eval()
        batch = collate_preference_batch([example("q", "good", "bad")], 0)
        chosen, rejected, _, _ = concatenated_completion_log_probabilities(model, batch)
        self.assertTrue(torch.allclose(chosen, completion_log_probability(model, batch["chosen"])))
        self.assertTrue(torch.allclose(rejected, completion_log_probability(model, batch["rejected"])))

    def test_pair_weighted_accumulation_matches_combined_batch(self) -> None:
        torch.manual_seed(11)
        accumulated = MiniScaleForCausalLM(MiniScaleConfig.smoke())
        combined = deepcopy(accumulated)
        reference = deepcopy(accumulated).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        rows = [example("q1", "good", "bad"), example("q2", "better answer", "worse")]
        for row in rows:
            batch = collate_preference_batch([row], 0)
            with torch.no_grad():
                ref_chosen, ref_rejected, _, _ = concatenated_completion_log_probabilities(reference, batch)
            policy_chosen, policy_rejected, _, _ = concatenated_completion_log_probabilities(
                accumulated, batch
            )
            loss, _ = dpo_loss(
                policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
            )
            (loss / len(rows)).backward()

        batch = collate_preference_batch(rows, 0)
        with torch.no_grad():
            ref_chosen, ref_rejected, _, _ = concatenated_completion_log_probabilities(reference, batch)
        policy_chosen, policy_rejected, _, _ = concatenated_completion_log_probabilities(combined, batch)
        loss, _ = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )
        loss.backward()
        for left, right in zip(accumulated.parameters(), combined.parameters(), strict=True):
            self.assertTrue(torch.allclose(left.grad, right.grad, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
