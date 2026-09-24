"""Compatibility import; implementation lives in training.pretrain.resume."""

import sys
from ..pretrain.resume import __name__ as _target_name

sys.modules[__name__] = sys.modules[_target_name]
