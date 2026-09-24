from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class MiniScaleConfig:
    vocab_size: int = 260
    hidden_size: int = 256
    intermediate_size: int = 688
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    max_position_embeddings: int = 1024
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    @classmethod
    def smoke(cls) -> "MiniScaleConfig":
        return cls(
            hidden_size=32,
            intermediate_size=88,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=384,
        )

    @classmethod
    def small_64m(
        cls,
        vocab_size: int = 6400,
        max_position_embeddings: int = 512,
        *,
        num_hidden_layers: int = 20,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> "MiniScaleConfig":
        """The project base geometry, with about 63.6M parameters at the default 20 layers."""
        return cls(
            vocab_size=vocab_size,
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MiniScaleConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
