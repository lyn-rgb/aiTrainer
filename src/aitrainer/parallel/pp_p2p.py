"""Synchronous pipeline P2P protocol with explicit tensor metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pp_shapes import TensorSpec


class P2PError(RuntimeError):
    """Raised for invalid pipeline peer or message metadata."""


@dataclass
class P2PWork:
    work: Any
    description: str
    key: tuple[Any, ...] | None = None
    _waited: bool = False
    result: Any = None
    buffers: tuple[Any, ...] = ()

    def wait(self) -> None:
        if self._waited:
            return
        if callable(self.work):
            self.result = self.work()
        elif isinstance(self.work, (tuple, list)):
            for item in self.work:
                if item is not None and hasattr(item, "wait"):
                    item.wait()
        elif self.work is not None and hasattr(self.work, "wait"):
            self.work.wait()
        self._waited = True


class P2PCommunicator:
    """Point-to-point communicator for one non-interleaved pipeline stage."""

    def __init__(self, *, pp_rank: int, pp_size: int, pp_group: Any = None,
                 pp_ranks: tuple[int, ...] | None = None,
                 timeout_seconds: float = 1800.0) -> None:
        if not 0 <= pp_rank < pp_size:
            raise P2PError(f"pp_rank={pp_rank} outside [0, {pp_size})")
        if pp_ranks is not None and len(pp_ranks) != pp_size:
            raise P2PError("pp_ranks length must match pp_size")
        self.pp_rank, self.pp_size, self.pp_group = pp_rank, pp_size, pp_group
        self.pp_ranks = pp_ranks
        self.timeout_seconds = timeout_seconds
        self._pending: list[P2PWork] = []

    @property
    def is_first(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    def _dist(self) -> Any:
        try:
            import torch.distributed as dist
            if not dist.is_available() or not dist.is_initialized():
                raise P2PError("torch.distributed must be initialized before pipeline P2P")
            return dist
        except ImportError as exc:
            raise P2PError("PyTorch distributed is required for pipeline P2P") from exc

    def _peer(self, forward: bool) -> int:
        if forward:
            if self.is_last:
                raise P2PError("last pipeline stage has no forward peer")
            return self.pp_ranks[self.pp_rank + 1] if self.pp_ranks is not None else self.pp_rank + 1
        if self.is_first:
            raise P2PError("first pipeline stage has no backward peer")
        return self.pp_ranks[self.pp_rank - 1] if self.pp_ranks is not None else self.pp_rank - 1

    def recv_forward(self, spec: TensorSpec) -> Any:
        self._validate_message_spec(spec, forward=True)
        dist = self._dist()
        import torch
        self._receive_header(spec, forward=True)
        tensor = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        # An activation arrives from the upstream peer -- the same peer the
        # header came from (_receive_header uses _peer(not forward)).  Reading
        # the payload from _peer(True) mixed a header from one rank with a
        # payload from the other, hanging or corrupting every pp_size >= 2 run.
        dist.recv(tensor, src=self._peer(False), group=self.pp_group)
        spec.validate_tensor(tensor)
        return tensor.requires_grad_(True)

    def recv_forward_dynamic(self, *, device: Any) -> tuple[Any, TensorSpec]:
        """Receive metadata first and allocate a variable-shaped activation."""
        spec = self._receive_dynamic_spec(forward=True, device=device)
        import torch
        tensor = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        self._dist().recv(tensor, src=self._peer(False), group=self.pp_group)
        spec.validate_tensor(tensor)
        return tensor.requires_grad_(True), spec

    def send_forward(self, tensor: Any, *, spec: TensorSpec | None = None) -> None:
        dist = self._dist()
        if spec is None:
            raise P2PError("send_forward requires TensorSpec metadata")
        self._validate_message_spec(spec, forward=True)
        spec.validate_tensor(tensor)
        self._send_header(spec, tensor, forward=True)
        dist.send(tensor.detach().contiguous(), dst=self._peer(True), group=self.pp_group)

    def send_forward_async(self, tensor: Any, *, spec: TensorSpec) -> P2PWork:
        """Issue header and payload sends without blocking the next compute."""
        self._validate_message_spec(spec, forward=True)
        spec.validate_tensor(tensor)
        work = self._send_async(tensor, spec, forward=True)
        self._pending.append(work)
        return work

    def recv_backward(self, spec: TensorSpec) -> Any:
        self._validate_message_spec(spec, forward=False)
        dist = self._dist()
        import torch
        self._receive_header(spec, forward=False)
        grad = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        # A gradient arrives from the downstream peer; the header was read from
        # _peer(not forward) == _peer(True), so the payload must match it.
        dist.recv(grad, src=self._peer(True), group=self.pp_group)
        spec.validate_tensor(grad)
        return grad

    def recv_backward_dynamic(self, *, device: Any) -> tuple[Any, TensorSpec]:
        """Receive a variable-shaped backward gradient and its metadata."""
        spec = self._receive_dynamic_spec(forward=False, device=device)
        import torch
        grad = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        self._dist().recv(grad, src=self._peer(True), group=self.pp_group)
        spec.validate_tensor(grad)
        return grad, spec

    def send_backward(self, grad: Any, *, spec: TensorSpec | None = None) -> None:
        dist = self._dist()
        if spec is None:
            raise P2PError("send_backward requires TensorSpec metadata")
        self._validate_message_spec(spec, forward=False)
        spec.validate_tensor(grad)
        self._send_header(spec, grad, forward=False)
        dist.send(grad.detach().contiguous(), dst=self._peer(False), group=self.pp_group)

    def send_backward_async(self, grad: Any, *, spec: TensorSpec) -> P2PWork:
        """Issue a nonblocking backward header/payload send."""
        self._validate_message_spec(spec, forward=False)
        spec.validate_tensor(grad)
        work = self._send_async(grad, spec, forward=False)
        self._pending.append(work)
        return work

    def drain(self) -> None:
        errors: list[BaseException] = []
        for work in reversed(self._pending):
            try:
                work.wait()
            except BaseException as exc:
                errors.append(exc)
        self._pending.clear()
        if errors:
            raise P2PError(f"failed to drain {len(errors)} pipeline P2P handles") from errors[0]

    @staticmethod
    def _dtype_code(dtype: Any) -> int:
        name = str(dtype).replace("torch.", "")
        table = {"float32": 1, "float16": 2, "bfloat16": 3, "float64": 4,
                 "int64": 5, "int32": 6, "bool": 7}
        if name not in table:
            raise P2PError(f"unsupported pipeline dtype metadata {dtype!r}")
        return table[name]

    @staticmethod
    def _dtype_from_code(code: int) -> Any:
        import torch
        table = {1: torch.float32, 2: torch.float16, 3: torch.bfloat16, 4: torch.float64,
                 5: torch.int64, 6: torch.int32, 7: torch.bool}
        if code not in table:
            raise P2PError(f"unsupported pipeline dtype code {code}")
        return table[code]

    def _send_header(self, spec: TensorSpec, tensor: Any, *, forward: bool) -> None:
        import torch
        values = [1, 1 if forward else 2, self._dtype_code(spec.dtype), spec.stage,
                  spec.microbatch, len(spec.shape), *spec.shape]
        header = torch.tensor(values, dtype=torch.int64, device=tensor.device)
        dist = self._dist()
        dist.send(header[:6], dst=self._peer(forward), group=self.pp_group)
        dist.send(header[6:], dst=self._peer(forward), group=self.pp_group)

    def _send_async(self, tensor: Any, spec: TensorSpec, *, forward: bool) -> P2PWork:
        import torch
        values = [1, 1 if forward else 2, self._dtype_code(spec.dtype), spec.stage,
                  spec.microbatch, len(spec.shape), *spec.shape]
        header = torch.tensor(values, dtype=torch.int64, device=tensor.device)
        dist = self._dist()
        peer = self._peer(forward)
        payload = tensor.detach().contiguous()
        works = [dist.isend(header[:6], dst=peer, group=self.pp_group),
                 dist.isend(header[6:], dst=peer, group=self.pp_group),
                 dist.isend(payload, dst=peer, group=self.pp_group)]
        return P2PWork(works, "forward" if forward else "backward", buffers=(header, payload))

    def _receive_header(self, spec: TensorSpec, *, forward: bool) -> None:
        import torch
        header_size = torch.empty(6, dtype=torch.int64, device=spec.device)
        self._dist().recv(header_size, src=self._peer(not forward), group=self.pp_group)
        ndim = int(header_size[5].item())
        if ndim != len(spec.shape):
            raise P2PError(f"pipeline metadata rank mismatch: received ndim={ndim}, expected={len(spec.shape)}")
        header = torch.empty(6 + ndim, dtype=torch.int64, device=spec.device)
        header[:6].copy_(header_size)
        self._dist().recv(header[6:], src=self._peer(not forward), group=self.pp_group)
        received_shape = tuple(int(item) for item in header[6:].tolist())
        if received_shape != spec.shape:
            raise P2PError(f"pipeline metadata shape {received_shape} != expected {spec.shape}")
        if int(header[1].item()) != (1 if forward else 2):
            raise P2PError("pipeline forward/backward metadata tag mismatch")
        if int(header[2].item()) != self._dtype_code(spec.dtype):
            raise P2PError("pipeline dtype metadata mismatch")
        if int(header[3].item()) != spec.stage:
            raise P2PError("pipeline stage metadata mismatch")
        if int(header[4].item()) != spec.microbatch:
            raise P2PError("pipeline microbatch metadata mismatch")

    def _receive_dynamic_spec(self, *, forward: bool, device: Any) -> TensorSpec:
        import torch
        source = self._peer(False) if forward else self._peer(True)
        header_size = torch.empty(6, dtype=torch.int64, device=device)
        self._dist().recv(header_size, src=source, group=self.pp_group)
        ndim = int(header_size[5].item())
        if ndim < 1:
            raise P2PError(f"pipeline metadata ndim={ndim} is invalid")
        shape_header = torch.empty(ndim, dtype=torch.int64, device=device)
        self._dist().recv(shape_header, src=source, group=self.pp_group)
        tag_code = int(header_size[1].item())
        expected_tag = "forward" if forward else "backward"
        tag = {1: "forward", 2: "backward"}.get(tag_code)
        if tag != expected_tag:
            raise P2PError("pipeline forward/backward metadata tag mismatch")
        return TensorSpec(tuple(int(item) for item in shape_header.tolist()),
                          self._dtype_from_code(int(header_size[2].item())), device,
                          int(header_size[3].item()), int(header_size[4].item()), tag)

    @staticmethod
    def _validate_message_spec(spec: TensorSpec, *, forward: bool) -> None:
        if spec.tag not in {"forward", "backward"}:
            raise P2PError(f"unsupported pipeline message tag {spec.tag!r}")
        if forward and spec.tag != "forward":
            raise P2PError("forward P2P requires a forward message tag")
        if not forward and spec.tag != "backward":
            raise P2PError("backward P2P requires a backward message tag")
