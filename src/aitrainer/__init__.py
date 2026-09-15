"""aiTrainer: a model-agnostic PyTorch training foundation.

``import aitrainer`` works with torch absent -- only validation, topology and
planning are reachable then, and everything that needs torch raises when called.
``tests/unit/test_torch_optional.py`` pins that property.
"""

from .api import *  # noqa: F401,F403
from .api import __all__ as _api_all

# Explicitly the curated list.  Deriving it from `globals()` also picked up the
# ~27 submodules the import system binds as package attributes (`aitrainer.mesh`,
# `aitrainer.precision`, ...), so `__all__` described the import machinery rather
# than the API.
__all__ = list(_api_all)
