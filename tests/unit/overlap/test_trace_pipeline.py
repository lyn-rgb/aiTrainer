from aitrainer.overlap import ParameterPrefetchCoordinator


def test_trace_invalidation_falls_back_to_fetch():
    calls = []
    coordinator = ParameterPrefetchCoordinator(fetch_fn=lambda name: calls.append(name) or name)
    coordinator.record("m", ("p",)); assert coordinator.finalize()
    op = coordinator.fetch("p"); assert op.wait() == "p"
    coordinator.invalidate(); assert coordinator.fetch("q") is None
    assert coordinator.wait("q") == "q"


def test_finalize_compares_against_the_digest_it_was_given():
    """``finalize(expected=...)`` has to be able to return True.

    ``fingerprint`` took a ``config`` argument defaulting to None, and
    ``finalize`` recomputed the digest without it -- so a caller that
    fingerprinted with a config always got False back, ``valid`` stayed False,
    and ``prefetch_async`` stayed on its synchronous branch forever.  Every test
    called ``finalize()`` with no argument, which skips the comparison.
    """
    from aitrainer.overlap.parameter import ParameterPrefetchCoordinator as Coordinator

    def recorded(order):
        coordinator = Coordinator()
        for name in order:
            coordinator.record(name, [f"{name}.weight"])
        return coordinator

    digest = recorded(["embed", "block0", "block1"]).fingerprint()
    assert recorded(["embed", "block0", "block1"]).finalize(expected=digest) is True, (
        "the same access order must validate against its own digest")
    for label, order in (("reordered", ["embed", "block1", "block0"]),
                         ("shorter", ["embed", "block0"]),
                         ("longer", ["embed", "block0", "block1", "head"])):
        assert recorded(order).finalize(expected=digest) is False, (
            f"a {label} access order was accepted, so prefetch would run against "
            "the wrong layer")


class _StubParameter:
    """A parameter-shaped object whose ``.data`` accepts any tensor.

    A real ``nn.Parameter`` refuses ``.data = <tensor of another type>``
    ("incompatible tensor type"), which is exactly what makes the copy path
    unobservable on a host with no accelerator: CPU is the only device here and
    a copy to CPU is a no-op.  The property under test is the ORDERING, not the
    bytes, so a stand-in that accepts the assignment is enough to observe it.
    """

    def __init__(self, shape=(4, 4)):
        import torch
        self._tensor = torch.zeros(shape)
        self.shape = shape
        self.device = "cpu"

    @property
    def data(self):
        return self._tensor

    @data.setter
    def data(self, value):
        self._tensor = value

    def detach(self):
        return self._tensor

    def numel(self):
        return self._tensor.numel()

    def element_size(self):
        return self._tensor.element_size()


class _StubModule:
    def __init__(self, count=2, shape=(4, 4)):
        self.params = [_StubParameter(shape) for _ in range(count)]

    def parameters(self):
        return iter(self.params)


def test_prefetch_issues_the_copy_before_the_wait():
    """The copy has to be in flight when the call returns, or nothing overlaps.

    ``prefetch_async`` used to store the fetch as a closure on ``op.handle`` with
    ``op._wait_fn = lambda fn: fn()``, so the copy ran when the consumer waited:
    a deferred fetch, not an early one, and it hid nothing.  Two things separate
    the two implementations from outside -- the parameter's ``data`` has been
    repointed at the destination by the time the call returns, and ``handle`` is
    a completion marker rather than a callable.

    ``meta`` is the target because it is the only device this host can accept
    that is not CPU, and it is enough for a question about ordering.
    """
    from aitrainer.offload.parameter import ParameterOffloader

    offloader = ParameterOffloader(pin_memory=False)
    module = _StubModule()
    offloader.register_module(module)
    offloader.trace.finalize()                      # open the gate
    assert str(module.params[0].data.device) == "cpu"

    handle = offloader.prefetch_async(module, device="meta")
    assert handle is not None
    assert str(module.params[0].data.device) == "meta", (
        "the copy had not been issued when prefetch_async returned -- it was "
        "deferred to wait(), which is the bug this replaced")
    assert not callable(handle.handle), (
        "handle is still a deferred closure rather than a completion marker")


def test_prefetch_falls_back_when_the_budget_refuses():
    """Refused for budget is not the same as "nothing to do".

    ``prefetch_async`` returned None without fetching when the coordinator said
    no, leaving the parameters on the host and raising nothing -- and a caller
    reads a declined prefetch as staged state.
    """
    from aitrainer.offload.parameter import ParameterOffloader

    offloader = ParameterOffloader(pin_memory=False)
    module = _StubModule()
    offloader.register_module(module)
    offloader.trace.max_prefetched_bytes = 1        # nothing fits
    offloader.trace.finalize()

    assert offloader.prefetch_async(module, device="meta") is None
    assert str(module.params[0].data.device) == "meta", (
        "the budget refused the prefetch and nothing fetched the parameters, so "
        "they are still on the host with no error raised")


def _digest_of(build):
    """The prefetch digest for a freshly built model, as a second run would see it."""
    import torch

    from aitrainer.offload.parameter import ParameterOffloader

    torch.manual_seed(0)
    offloader = ParameterOffloader(pin_memory=False)
    offloader.register_module(build())
    return offloader.trace.fingerprint().value


def _model(depth=2, extra=False):
    import torch

    model = torch.nn.Sequential()
    model.add_module("embed", torch.nn.Embedding(8, 4))
    for index in range(depth):
        block = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4))
        if extra and index == 0:
            block.add_module("extra", torch.nn.Linear(4, 4))
        model.add_module(f"b{index}", block)
    return model


def test_the_digest_is_stable_across_runs_of_the_same_model():
    """Or ``finalize(expected=<warmup digest>)`` can never match, and never has.

    The entries held ``str(id(p))``.  An id is a memory address, so two
    constructions of the same model fingerprinted differently -- measured, and
    enough on its own to keep the prefetch gate shut for every run.
    """
    assert _digest_of(_model) == _digest_of(_model), (
        "the same model fingerprinted differently on a second run, so a plan "
        "validated against a warmup digest can never be accepted")


def test_a_structural_change_moves_the_digest():
    """The other direction: a plan must not be reused for a different model."""
    assert _digest_of(_model) != _digest_of(lambda: _model(extra=True))


def test_one_block_of_each_kind_is_distinguishable():
    """Names have to reach the submodule, not stop at the child.

    With local names a block holding an extra projection and one without both
    record as ``norm``/``proj``/``attn`` in position order, so the two layouts
    digest the same.
    """
    import torch

    class Kind(torch.nn.Module):
        def __init__(self, extra):
            super().__init__()
            self.norm = torch.nn.LayerNorm(4)
            if extra:
                self.attn = torch.nn.Linear(4, 4)

    def build(first_has_extra):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b0 = Kind(extra=first_has_extra)
                self.b1 = Kind(extra=not first_has_extra)
        return Net()

    assert _digest_of(lambda: build(True)) != _digest_of(lambda: build(False)), (
        "two different child layouts digested the same, so the trace names are "
        "not fully qualified")


def test_the_trace_records_every_layer_not_the_whole_model():
    """A plan's granularity is the trace's granularity.

    Registering the whole model recorded ONE entry -- measured as a single
    ``TraceEntry(module='Blk', ...)`` for a two-layer model -- and a digest over
    one entry that names the container says nothing about layer order.
    """
    from aitrainer.offload.parameter import ParameterOffloader

    model = _model(depth=3)
    offloader = ParameterOffloader(pin_memory=False)
    offloader.register_module(model)
    recorded = [entry.module for entry in offloader.trace.trace]
    assert recorded == ["Sequential", "embed", "b0", "b0.0", "b0.1",
                        "b1", "b1.0", "b1.1", "b2", "b2.0", "b2.1"], recorded
