import sys as _sys

from .configs import dpo as _dpo_config
from .configs import pretrain as _pretrain_config
from .configs import rl as _rl_config
from .configs import sft as _sft_config
from .core import artifacts as _artifacts
from .core import checkpoint as _checkpoint
from .core import common as _common
from .core import runtime as _runtime
from .evaluators import dpo as _dpo_evaluation
from .evaluators import sft as _sft_evaluation
from .objectives import dpo as _dpo_objective
from .objectives import grpo as _grpo_objective
from .agent_rl import runner as _agent_rl
from .dpo import runner as _dpo
from .grpo import runner as _grpo
from .core import rl_runtime as _rl_runtime
from .pretrain import runner as _pretrain
from .sft import runner as _sft
from .dpo.config import DPOOptions
from .pretrain.config import PretrainOptions, SmokePretrainOptions
from .core.rl_config import AgentRLOptions, GRPOOptions
from .sft.config import SFTOptions, SmokeSFTOptions
from ..data.rl import RLTask
from .agent_rl import run_agent_grpo, run_agent_grpo_jsonl
from .dpo import run_dpo_jsonl
from .grpo import run_grpo, run_grpo_jsonl
from .pretrain import run_pretrain, run_pretrain_jsonl
from .sft import run_sft, run_sft_jsonl


_LEGACY_TRAINING_MODULES = {
    "artifacts": _artifacts,
    "checkpoint": _checkpoint,
    "common": _common,
    "dpo_config": _dpo_config,
    "dpo_evaluation": _dpo_evaluation,
    "dpo_objective": _dpo_objective,
    "grpo_objective": _grpo_objective,
    "pretrain_config": _pretrain_config,
    "rl_config": _rl_config,
    "rl_runtime": _rl_runtime,
    "runtime": _runtime,
    "sft_config": _sft_config,
    "sft_evaluation": _sft_evaluation,
}
for _legacy_name, _module in _LEGACY_TRAINING_MODULES.items():
    _sys.modules[f"{__name__}.{_legacy_name}"] = _module
    setattr(_sys.modules[__name__], _legacy_name, _module)

__all__ = [
    "AgentRLOptions",
    "DPOOptions",
    "GRPOOptions",
    "PretrainOptions",
    "SmokePretrainOptions",
    "RLTask",
    "SFTOptions",
    "SmokeSFTOptions",
    "run_agent_grpo",
    "run_agent_grpo_jsonl",
    "run_dpo_jsonl",
    "run_grpo",
    "run_grpo_jsonl",
    "run_pretrain",
    "run_pretrain_jsonl",
    "run_sft",
    "run_sft_jsonl",
]
