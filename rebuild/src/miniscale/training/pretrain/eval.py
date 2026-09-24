"""Fixed greedy generation probes for base-model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch

from miniscale.integrity import atomic_write_json
from miniscale.model import MiniScaleForCausalLM
from miniscale.tokenizer import Tokenizer
from ..core.runtime import autocast_context as _autocast_context


GENERATION_EVAL_PROMPTS: tuple[dict[str, str], ...] = (
    {"name": "chinese", "language": "zh", "prompt": "人工智能的发展将会"},
    {"name": "english", "language": "en", "prompt": "The future of artificial intelligence is"},
    {
        "name": "code",
        "language": "python",
        "prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    },
)


@torch.no_grad()
def run_generation_evaluation(
    model: MiniScaleForCausalLM,
    tokenizer: Tokenizer,
    output_dir: str | Path,
    *,
    step: int,
    device: torch.device,
    max_new_tokens: int,
    autocast_dtype: torch.dtype | None = None,
) -> Path:
    """Generate fixed multilingual probes with deterministic greedy decoding."""

    was_training = model.training
    model.eval()
    samples: list[dict[str, object]] = []
    try:
        for probe in GENERATION_EVAL_PROMPTS:
            prompt_ids = tokenizer.encode(probe["prompt"], bos=True)
            if len(prompt_ids) >= model.config.max_position_embeddings:
                prompt_ids = prompt_ids[-(model.config.max_position_embeddings - 1) :]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with _autocast_context(device, autocast_dtype):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    top_k=None,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                )
            completion_ids = generated[0, len(prompt_ids) :].tolist()
            samples.append({
                **probe,
                "prompt_tokens": len(prompt_ids),
                "generated_tokens": len(completion_ids),
                "response": tokenizer.decode(completion_ids),
            })
    finally:
        model.train(was_training)

    target = Path(output_dir) / "generations" / f"step_{step:08d}.json"
    atomic_write_json(target, {
        "stage": "pretrain",
        "step": step,
        "decoding": {"do_sample": False, "strategy": "greedy", "temperature": 0.0},
        "samples": samples,
    })
    return target
