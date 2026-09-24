"""Compatibility imports; the canonical implementation lives in ``src/miniscale``."""

from miniscale.config import MiniScaleConfig
from miniscale.model import MiniScaleForCausalLM

__all__ = ["MiniScaleConfig", "MiniScaleForCausalLM"]
