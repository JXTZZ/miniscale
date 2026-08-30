from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import MiniScaleConfig


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(dtype)


def _rotate_half(hidden_states: Tensor) -> Tensor:
    left, right = hidden_states.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inverse_frequency.dtype)
        frequencies = torch.outer(positions, self.inverse_frequency)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(dtype)[None, None, :, :], embeddings.sin().to(dtype)[None, None, :, :]


def _apply_rope(query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor) -> tuple[Tensor, Tensor]:
    return query * cosine + _rotate_half(query) * sine, key * cosine + _rotate_half(key) * sine


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniScaleConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.key = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.dropout = config.dropout

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cosine, sine = self.rotary(seq_len, hidden_states.device, hidden_states.dtype)
        query, key = _apply_rope(query, key, cosine, sine)
        key = key.repeat_interleave(self.kv_groups, dim=1)
        value = value.repeat_interleave(self.kv_groups, dim=1)

        combined_mask = None
        is_causal = attention_mask is None
        if attention_mask is not None:
            causal = torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool).tril()
            combined_mask = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=combined_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: MiniScaleConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(hidden_states)) * self.up(hidden_states))


class DecoderLayer(nn.Module):
    def __init__(self, config: MiniScaleConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states), attention_mask)
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


class MiniScaleForCausalLM(nn.Module):
    def __init__(self, config: MiniScaleConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self._initialize_residual_projections()
        # Tie only after initialization. Tying first makes ``Module.apply``
        # initialize the same Parameter twice through embedding and lm_head,
        # and overwrites the embedding padding-row initialization.
        self.lm_head.weight = self.embedding.weight

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def _initialize_residual_projections(self) -> None:
        """Scale residual branch outputs to keep variance stable with depth."""

        residual_std = 0.02 / math.sqrt(2 * self.config.num_hidden_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attention.output.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.mlp.down.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> CausalLMOutput:
        if input_ids.size(1) > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: int | None = None,
        do_sample: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive when set")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least 1")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be non-negative")
        generated = input_ids
        prompt_length = input_ids.shape[1]
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            window = generated[:, -self.config.max_position_embeddings :]
            logits = self(window).logits[:, -1]
            completion = generated[:, prompt_length:]
            if repetition_penalty != 1.0 and completion.numel():
                for row in range(logits.shape[0]):
                    repeated = completion[row].unique()
                    values = logits[row, repeated]
                    logits[row, repeated] = torch.where(
                        values < 0,
                        values * repetition_penalty,
                        values / repetition_penalty,
                    )
            if no_repeat_ngram_size and completion.shape[1] + 1 >= no_repeat_ngram_size:
                for row in range(logits.shape[0]):
                    tokens = completion[row].tolist()
                    prefix_size = no_repeat_ngram_size - 1
                    prefix = tokens[-prefix_size:] if prefix_size else []
                    banned: set[int] = set()
                    prior_ngrams = len(tokens) - no_repeat_ngram_size + 1
                    for start in range(max(0, prior_ngrams)):
                        if tokens[start : start + prefix_size] == prefix:
                            banned.add(tokens[start + prefix_size])
                    if banned:
                        logits[row, list(banned)] = -math.inf
            if do_sample is False or temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, -math.inf)
                probabilities = F.softmax(logits, dim=-1)
                if top_p < 1.0:
                    sorted_probabilities, sorted_indices = torch.sort(
                        probabilities, descending=True, dim=-1
                    )
                    cumulative = sorted_probabilities.cumsum(dim=-1)
                    remove = cumulative - sorted_probabilities >= top_p
                    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
                    probabilities = torch.zeros_like(probabilities).scatter(
                        -1, sorted_indices, sorted_probabilities
                    )
                    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
            next_token = torch.where(
                finished[:, None],
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token), dim=1)
            finished |= next_token.squeeze(-1).eq(eos_token_id)
            if bool(finished.all()):
                break
        return generated

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
