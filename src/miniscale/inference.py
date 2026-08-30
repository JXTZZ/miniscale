from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import ByteTokenizer, load_tokenizer
from .agent_env import CalculatorTask
from .training.configs.rl import AgentRLOptions
from .training.core.checkpoint import load_checkpoint
from .training.core.runtime import resolve_device
from .training.stages.agent_rl import rollout_agent


@dataclass(slots=True)
class GenerationOptions:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_k: int | None = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    seed: int | None = None
    device: str = "auto"
    system_prompt: str | None = None
    tokenizer_path: str | Path | None = None
    raw_prompt: bool = False
    calculator: bool = False
    max_turns: int = 6


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
    if options.max_turns < 1:
        raise ValueError("max_turns must be positive")
    if options.calculator and options.raw_prompt:
        raise ValueError("calculator inference requires the chat template")

    device = resolve_device(options.device)
    model = load_checkpoint(checkpoint_path, device).eval()
    tokenizer = load_tokenizer(options.tokenizer_path) if options.tokenizer_path else ByteTokenizer()
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"checkpoint vocabulary ({model.config.vocab_size}) does not match tokenizer ({tokenizer.vocab_size})"
        )
    if options.calculator:
        trajectory = rollout_agent(
            model,
            tokenizer,
            CalculatorTask(prompt, "", "", options.system_prompt),
            AgentRLOptions(
                max_turns=options.max_turns,
                max_new_tokens=options.max_new_tokens,
                temperature=options.temperature,
                top_k=options.top_k,
                device=str(device),
            ),
            device,
        )
        return {
            "checkpoint": str(checkpoint_path),
            "prompt": prompt,
            "response": trajectory.final_answer,
            "prompt_tokens": len(trajectory.input_ids) - sum(trajectory.action_mask),
            "generated_tokens": sum(trajectory.action_mask),
            "temperature": options.temperature,
            "top_k": options.top_k,
            "device": str(device),
            "tool_calls": trajectory.valid_calls,
            "invalid_tool_calls": trajectory.invalid_calls,
            "turns": trajectory.turns,
            "transcript": trajectory.transcript,
        }
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
    generator = None
    if options.seed is not None:
        generator = torch.Generator(device=device).manual_seed(options.seed)
    with torch.inference_mode():
        generated = model.generate(
            input_ids,
            max_new_tokens=options.max_new_tokens,
            temperature=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
            generator=generator,
        )
    raw_response_ids = generated[0, len(prompt_ids) :].tolist()
    finish_reason = "max_tokens"
    if tokenizer.eos_token_id in raw_response_ids:
        eos_index = raw_response_ids.index(tokenizer.eos_token_id)
        response_ids = raw_response_ids[: eos_index + 1]
        finish_reason = "eos"
    else:
        response_ids = raw_response_ids
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
        "top_p": options.top_p,
        "repetition_penalty": options.repetition_penalty,
        "no_repeat_ngram_size": options.no_repeat_ngram_size,
        "seed": options.seed,
        "finish_reason": finish_reason,
        "device": str(device),
    }
