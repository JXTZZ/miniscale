"""Compatibility import; implementation lives in training.grpo.runner."""

import sys
from ..grpo.runner import __name__ as _target_name

sys.modules[__name__] = sys.modules[_target_name]
