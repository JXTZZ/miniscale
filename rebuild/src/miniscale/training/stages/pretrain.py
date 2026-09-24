"""Compatibility import; implementation lives in training.pretrain.runner."""

import sys
from ..pretrain.runner import __name__ as _target_name

sys.modules[__name__] = sys.modules[_target_name]
