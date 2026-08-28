import sys as _sys

from .config import MiniScaleConfig
from .data import agent as _agent_data
from .data import metrics as _data_metrics
from .data import preference as _preference_data
from .data import preference_audit as _dpo_data_audit
from .data import pretrain_audit as _data_audit
from .data import rl as _rl_data
from .data import sft as _sft_data
from .data import sft_audit as _sft_data_audit
from .data import sft_prepare as _sft_data_prepare
from .model import MiniScaleForCausalLM
from .tokenizer import ByteTokenizer, HuggingFaceTokenizer, SentencePieceTokenizer


_LEGACY_DATA_MODULES = {
    "agent_data": _agent_data,
    "data_audit": _data_audit,
    "data_metrics": _data_metrics,
    "dpo_data_audit": _dpo_data_audit,
    "preference_data": _preference_data,
    "rl_data": _rl_data,
    "sft_data": _sft_data,
    "sft_data_audit": _sft_data_audit,
    "sft_data_prepare": _sft_data_prepare,
}
for _legacy_name, _module in _LEGACY_DATA_MODULES.items():
    _sys.modules[f"{__name__}.{_legacy_name}"] = _module
    setattr(_sys.modules[__name__], _legacy_name, _module)

__all__ = [
    "ByteTokenizer",
    "HuggingFaceTokenizer",
    "MiniScaleConfig",
    "MiniScaleForCausalLM",
    "SentencePieceTokenizer",
]


def main() -> None:
    from .cli import main as cli_main

    cli_main()
