"""Compatibility import; implementation lives in training.sft.runner."""

import sys
from ..sft.runner import __name__ as _target_name

sys.modules[__name__] = sys.modules[_target_name]
