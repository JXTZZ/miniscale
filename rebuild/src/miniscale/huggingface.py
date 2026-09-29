"""Export native weights as a standard Transformers Llama (no remote code)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from .integrity import atomic_write_json, path_identity, tokenizer_identity
from .model import MiniScaleForCausalLM
from .tokenizer import HuggingFaceTokenizer
from .training.core.checkpoint import load_checkpoint


def to_huggingface_model(model: MiniScaleForCausalLM):
    """Return an independent CPU Llama with the same geometry and weights."""
    from transformers import LlamaConfig, LlamaForCausalLM

    config = asdict(model.config)
    config["attention_dropout"] = config.pop("dropout")
    config.update(tie_word_embeddings=True, hidden_act="silu", attention_bias=False, mlp_bias=False)
    with torch.random.fork_rng(devices=[]):
        exported = LlamaForCausalLM(LlamaConfig(**config))
    names = {
        "embedding.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    layer_names = {
        "attention_norm": "input_layernorm",
        "mlp_norm": "post_attention_layernorm",
        "attention.query": "self_attn.q_proj",
        "attention.key": "self_attn.k_proj",
        "attention.value": "self_attn.v_proj",
        "attention.output": "self_attn.o_proj",
        "mlp.gate": "mlp.gate_proj",
        "mlp.up": "mlp.up_proj",
        "mlp.down": "mlp.down_proj",
    }
    for index in range(model.config.num_hidden_layers):
        for native, hf in layer_names.items():
            names[f"layers.{index}.{native}.weight"] = f"model.layers.{index}.{hf}.weight"
    exported.load_state_dict({names[name]: value.detach().cpu() for name, value in model.state_dict().items()})
    return exported.eval()


def export_huggingface(
    checkpoint: str | Path,
    output_dir: str | Path,
    tokenizer_path: str | Path,
) -> Path:
    """Write safetensors, config, generation config, tokenizer and provenance.

    This is a model export for inference/stage handoff. Native optimizer and
    data-stream state remain in the original training checkpoint.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Hugging Face export already exists: {output}")
    tokenizer = HuggingFaceTokenizer(tokenizer_path)
    model = load_checkpoint(checkpoint)
    for name in ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id"):
        if getattr(model.config, name) != getattr(tokenizer, name):
            raise ValueError(f"checkpoint {name} does not match tokenizer")
    exported = to_huggingface_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed conversion cannot leave a half-written serving directory.
    with tempfile.TemporaryDirectory(prefix=".hf-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        exported.save_pretrained(staging, safe_serialization=True)
        tokenizer.processor.save_pretrained(staging)
        atomic_write_json(staging / "miniscale_export.json", {
            "schema_version": 1,
            "architecture": "LlamaForCausalLM",
            "source_checkpoint": path_identity(checkpoint),
            "native_config": asdict(model.config),
            "tokenizer": tokenizer_identity(tokenizer),
        })
        staging.rename(output)
    return output
