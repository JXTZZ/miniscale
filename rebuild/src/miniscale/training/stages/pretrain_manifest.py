"""Compatibility import; implementation lives in training.pretrain.manifest."""

import sys
from ..pretrain.manifest import __name__ as _target_name

sys.modules[__name__] = sys.modules[_target_name]
