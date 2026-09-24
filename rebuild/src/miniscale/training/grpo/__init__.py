"""Group relative policy optimization stage."""

from miniscale.data.rl import RLTask
from .config import GRPOOptions
from .objective import grpo_objective, normalize_group_rewards
from .rollout import collect_rollouts, math_reward
from .eval import evaluate_grpo
from .runner import run_grpo, run_grpo_jsonl
