import sys as _sys

from .configs import dpo as _dpo_config
from .configs import pretrain as _pretrain_config
from .configs import rl as _rl_config
from .configs import sft as _sft_config
from .core import artifacts as _artifacts
from .core import checkpoint as _checkpoint
from .core import common as _common
from .core import rl_runtime as _rl_runtime
from .core import runtime as _runtime
from .evaluators import dpo as _dpo_evaluation
from .evaluators import sft as _sft_evaluation
from .objectives import dpo as _dpo_objective
from .objectives import grpo as _grpo_objective
from .stages import agent_rl as _agent_rl
from .stages import dpo as _dpo
from .stages import grpo as _grpo
from .stages import pretrain as _pretrain
from .stages import sft as _sft
from .configs.dpo import DPOOptions
from .configs.pretrain import PretrainOptions, SmokePretrainOptions
from .configs.rl import AgentRLOptions, GRPOOptions
from .configs.sft import SFTOptions, SmokeSFTOptions
from ..data.rl import RLTask
from .stages.agent_rl import run_agent_grpo, run_agent_grpo_jsonl
from .stages.dpo import run_dpo_jsonl
from .stages.grpo import run_grpo, run_grpo_jsonl
from .stages.pretrain import run_pretrain, run_pretrain_jsonl
from .stages.sft import run_sft, run_sft_jsonl


_LEGACY_TRAINING_MODULES = {
    "agent_rl": _agent_rl,
    "artifacts": _artifacts,
    "checkpoint": _checkpoint,
    "common": _common,
    "dpo": _dpo,
    "dpo_config": _dpo_config,
    "dpo_evaluation": _dpo_evaluation,
    "dpo_objective": _dpo_objective,
    "grpo": _grpo,
    "grpo_objective": _grpo_objective,
    "pretrain": _pretrain,
    "pretrain_config": _pretrain_config,
    "rl_config": _rl_config,
    "rl_runtime": _rl_runtime,
    "runtime": _runtime,
    "sft": _sft,
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
