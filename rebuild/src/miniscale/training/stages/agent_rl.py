"""Compatibility import; implementation lives in training.agent_rl.runner."""

import sys
from ..agent_rl.runner import __name__ as _target_name

sys.modules[__name__] = sys.modules[_target_name]
