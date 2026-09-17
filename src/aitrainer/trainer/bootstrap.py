"""Turning a model or adapter into the pieces a ``Trainer`` needs.

Split out because the ordering here is load-bearing and easy to get wrong: the
runtime is created -- and therefore the seed applied -- BEFORE the model is
built.  Building first drew weight initialisation from the ambient unseeded RNG,
so ``seed`` never made initialisation reproducible, and the shipped example
produced five different losses across five runs.
"""

from __future__ import annotations

from typing import Any

from ..core.torch import require_torch
from ..runtime import Runtime


def prepare(model_or_adapter: Any, *, config: Any = None, optimizer: Any = None,
            optimizer_factory: Any = None, optimizer_cls: Any = None,
            optimizer_kwargs: Any = None, device: str | None = None,
            runtime: Runtime | None = None) -> tuple[Any, Any, Runtime | None]:
    """Build the model and optimizer, seeding before the model exists.

    Returns ``(model, optimizer, runtime)``.  Every rule that used to live inline
    in ``Trainer.from_model`` is here, unchanged:
    an adapter builds itself on CPU unless told otherwise, automatic parallel
    wrapping refuses a pre-built optimizer, and only one of
    optimizer/optimizer_factory/optimizer_cls may be supplied.
    """
    model = model_or_adapter
    runtime_obj = runtime
    requested_parallel = bool(config is not None and (
        config.fsdp.enabled or config.parallel.dp_size > 1 or
        config.parallel.tp_size > 1 or config.parallel.pp_size > 1))
    if runtime_obj is None and config is not None:
        runtime_obj = Runtime(device=device or config.device, seed=config.seed)
    if hasattr(model_or_adapter, "build") and not hasattr(model_or_adapter, "parameters"):
        build_device = device or (config.device if config is not None else "auto")
        if build_device == "auto":
            build_device = "cpu"
        model = model_or_adapter.build(device=build_device)
    if requested_parallel:
        if optimizer is not None:
            raise ValueError("pass an optimizer factory/class when automatic parallel wrapping is enabled")
        from ..parallelizer import parallelize
        # An adapter may declare how its model's forward is composed, for models
        # whose pipeline layers are not direct children.  Duck-typed rather than
        # added to the ModelAdapter protocol: requiring it would invalidate every
        # adapter already written against that protocol, and the ones that do not
        # need it -- anything with a flat forward -- should not have to say so.
        model = parallelize(model, config=config, runtime=runtime_obj,
                            execution_order=getattr(model_or_adapter, "execution_order", None))
    if optimizer is not None and (optimizer_factory is not None or optimizer_cls is not None):
        raise ValueError("pass only one of optimizer, optimizer_factory, or optimizer_cls")
    if optimizer is None and optimizer_factory is not None:
        optimizer = (optimizer_factory(model.parameters()) if callable(optimizer_factory)
                     else optimizer_factory.build(model.parameters()))
    if optimizer is None:
        torch = require_torch("aiTrainer training requires PyTorch; install the project's torch dependency")
        cls_optimizer = optimizer_cls or torch.optim.AdamW
        kwargs = dict(optimizer_kwargs or {})
        kwargs.setdefault("lr", 1e-3)
        optimizer = cls_optimizer(model.parameters(), **kwargs)
    return model, optimizer, runtime_obj
