from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import ByteTokenizer, SentencePieceTokenizer
from .training.common import load_checkpoint, resolve_device


@dataclass(slots=True)
class GenerationOptions:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int | None = 50
    device: str = "auto"
    system_prompt: str | None = None
    tokenizer_path: str | Path | None = None
    raw_prompt: bool = False


def generate_from_checkpoint(
    checkpoint: str | Path,
    prompt: str,
    options: GenerationOptions | None = None,
) -> dict[str, str | int | float | None]:
    options = options or GenerationOptions()
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if options.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    device = resolve_device(options.device)
    model = load_checkpoint(checkpoint_path, device).eval()
    tokenizer = SentencePieceTokenizer(options.tokenizer_path) if options.tokenizer_path else ByteTokenizer()
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"checkpoint vocabulary ({model.config.vocab_size}) does not match tokenizer ({tokenizer.vocab_size})"
        )
    if options.raw_prompt:
        formatted_prompt = prompt
    else:
        messages: list[dict[str, str]] = []
        if options.system_prompt:
            messages.append({"role": "system", "content": options.system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted_prompt = tokenizer.format_messages(messages, generation_prompt=True)
    prompt_ids = tokenizer.encode(formatted_prompt, bos=True)
    prompt_budget = model.config.max_position_embeddings - options.max_new_tokens
    if prompt_budget < 1:
        raise ValueError("max_new_tokens must be smaller than the model context window")
    prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=options.max_new_tokens,
            temperature=options.temperature,
            top_k=options.top_k,
        )
    response_ids = generated[0, len(prompt_ids) :].tolist()
    response = tokenizer.decode(response_ids)
    if "<|end|>" in response:
        response = response.split("<|end|>", 1)[0]
    return {
        "checkpoint": str(checkpoint_path),
        "prompt": prompt,
        "response": response,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(response_ids),
        "temperature": options.temperature,
        "top_k": options.top_k,
        "device": str(device),
    }
