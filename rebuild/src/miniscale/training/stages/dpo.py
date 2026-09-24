"""Compatibility import; implementation lives in training.dpo.runner."""

import sys
from ..dpo.runner import __name__ as _target_name

sys.modules[__name__] = sys.modules[_target_name]
