"""Leaf layer: the primitives everything else is allowed to depend on.

Nothing in ``aitrainer.core`` imports anything else in ``aitrainer``, and torch
is optional and lazy throughout -- which is what keeps ``import aitrainer``
working on a machine with no torch installed.

Modules:
    batching    batch conventions: target keys, forward inputs, loss, microbatches
    dtypes      dtype naming and sizing
    lifecycle   async operation state machine and scheduler
    tensors     container traversal and tensor sizing
    torch       the single optional-torch policy
"""

from __future__ import annotations
