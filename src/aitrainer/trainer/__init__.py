"""Training orchestration.

``Trainer`` is a facade: it owns the state and delegates the behaviour to
sibling modules, which take it as an argument.  See :mod:`aitrainer.trainer.step`
for why the state is not factored into a separate session object.
"""

from __future__ import annotations

from .facade import Trainer
from .state import StepOutput

__all__ = ["StepOutput", "Trainer"]
