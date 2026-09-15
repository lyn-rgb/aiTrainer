"""Deterministic, auditable topology and stage planning helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Sequence

from .config import ConfigurationError, FrameworkConfig
from .topology import RankMapping, Topology, make_rank_mapping
from .parallel.pp_shapes import StagePlan, plan_stages


class PlanningError(ConfigurationError):
    """Raised when an automatic plan cannot satisfy explicit constraints."""


@dataclass(frozen=True)
class PlanCandidate:
    dp_size: int
    tp_size: int
    pp_size: int
    mode: str
    score: float
    reasons: tuple[str, ...] = ()

    @property
    def world_size(self) -> int:
        return self.dp_size * self.tp_size * self.pp_size


@dataclass(frozen=True)
class PlanReport:
    enabled: bool
    world_size: int
    selected: PlanCandidate | None
    candidates: tuple[PlanCandidate, ...]
    stage_plans: tuple[StagePlan, ...] = ()
    fingerprint: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "world_size": self.world_size,
                "selected": _candidate_dict(self.selected),
                "candidates": [_candidate_dict(item) for item in self.candidates],
                "stage_plans": [plan.__dict__ for plan in self.stage_plans],
                "fingerprint": self.fingerprint, "reason": self.reason}


def _candidate_dict(candidate: PlanCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {"dp_size": candidate.dp_size, "tp_size": candidate.tp_size,
            "pp_size": candidate.pp_size, "mode": candidate.mode,
            "score": candidate.score, "reasons": list(candidate.reasons)}


def _layer_count(model: Any) -> int:
    if hasattr(model, "children"):
        return len(tuple(model.children()))
    if isinstance(model, Sequence):
        return len(model)
    return 0


def _parameter_count(model: Any) -> int:
    if hasattr(model, "parameters"):
        return sum(int(parameter.numel()) for parameter in model.parameters())
    return 0


def _fingerprint(config: FrameworkConfig, topology: Topology | None, model: Any) -> str:
    payload = repr((config.to_dict(), topology.rank_mapping if topology else None,
                    _layer_count(model), _parameter_count(model))).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _topology_penalty(topology: Topology | None, mode: str) -> float:
    if topology is None or not topology.peer_access:
        return 0.0
    links = [link for row in topology.peer_access for link in row]
    if not links:
        return 0.0
    inaccessible = sum(not link for link in links)
    return float(inaccessible) * {"tp": 2.0, "pp": 1.0, "fsdp": 0.5}.get(mode, 1.0)


def suggest_plan(model: Any, *, config: FrameworkConfig | None = None,
                 topology: Topology | None = None, world_size: int | None = None) -> PlanReport:
    """Return a deterministic recommendation without mutating model/config."""
    config = config or FrameworkConfig()
    world = int(world_size or (topology.world_size if topology else 1))
    config.validate(world_size=world if config.parallel.dp_size * config.parallel.tp_size * config.parallel.pp_size == world else
                    config.parallel.dp_size * config.parallel.tp_size * config.parallel.pp_size)
    if not config.planning.enabled:
        return PlanReport(False, world, None, (), fingerprint=_fingerprint(config, topology, model),
                          reason="automatic planning is disabled")
    if topology is None:
        # A topology has to describe `world` ranks.  Defaulting to pp=dp=tp=1
        # raised TopologyError for every world_size > 1, so an unconfigured
        # multi-rank world could never produce a plan.
        product = config.parallel.dp_size * config.parallel.tp_size * config.parallel.pp_size
        if product == world:
            dims = {"dp_size": config.parallel.dp_size, "tp_size": config.parallel.tp_size,
                    "pp_size": config.parallel.pp_size}
        else:
            dims = {"dp_size": world, "tp_size": 1, "pp_size": 1}
        topology = Topology("unknown", world, 0, 0, (), "unknown", make_rank_mapping(world, **dims))
    layers = tuple(model.children()) if hasattr(model, "children") else tuple(model) if isinstance(model, Sequence) else ()
    if not layers:
        raise PlanningError("planning requires an ordered model with at least one child layer")
    explicit = config.parallel
    explicit_non_default = any((
        (explicit.dp_size, explicit.tp_size, explicit.pp_size) != (1, 1, 1),
        explicit.sp_backend != "none",
        explicit.pp_schedule != "none",
    ))
    candidates: list[PlanCandidate] = []
    if explicit_non_default:
        mode = "tp" if explicit.tp_size > 1 else "pp" if explicit.pp_size > 1 else "fsdp" if config.fsdp.enabled else "eager"
        candidates.append(PlanCandidate(explicit.dp_size, explicit.tp_size, explicit.pp_size, mode,
                                         _topology_penalty(topology, mode), ("preserved explicit parallel config",)))
    else:
        if world == 1:
            candidates.append(PlanCandidate(1, 1, 1, "eager", 0.0, ("single-process baseline",)))
        if world > 1:
            candidates.append(PlanCandidate(world, 1, 1, "fsdp", _topology_penalty(topology, "fsdp"),
                                             ("data-parallel memory scaling",)))
            candidates.append(PlanCandidate(1, world, 1, "tp", _topology_penalty(topology, "tp"),
                                             ("tensor-parallel projection sharding",)))
            if world <= len(layers):
                candidates.append(PlanCandidate(1, 1, world, "pp", _topology_penalty(topology, "pp"),
                                                 ("pipeline stage capacity",)))
    candidates.sort(key=lambda item: (item.score, item.mode, item.dp_size, item.tp_size, item.pp_size))
    candidates = candidates[:config.planning.max_candidates]
    selected = candidates[0] if candidates else None
    stage_plans: tuple[StagePlan, ...] = ()
    if selected is not None and selected.pp_size > 1:
        stage_plans = plan_stages(layers, selected.pp_size, policy=config.planning.stage_policy)
    return PlanReport(True, world, selected, tuple(candidates), stage_plans,
                      _fingerprint(config, topology, model), "explicit dimensions preserved" if explicit_non_default else "generated candidates")


def apply_candidate(config: FrameworkConfig, candidate: PlanCandidate) -> FrameworkConfig:
    """Apply a report candidate only through an explicit caller action."""
    if not config.planning.allow_rewrite:
        raise PlanningError("planning.allow_rewrite=False; review and enable explicit plan application")
    if candidate.world_size < 1:
        raise PlanningError("candidate world size must be positive")
    schedule = "none" if candidate.pp_size == 1 else (
        "gpipe" if config.parallel.pp_schedule == "none" else config.parallel.pp_schedule)
    # `replace` keeps every field the candidate does not name — rebuilding the
    # dataclasses silently dropped sp_backend, pp_schedule and the whole FSDP
    # sub-config (mixed precision, prefetch, trace) while still validating.
    parallel = replace(config.parallel, dp_size=candidate.dp_size, tp_size=candidate.tp_size,
                       pp_size=candidate.pp_size, pp_schedule=schedule,
                       num_microbatches=max(config.parallel.num_microbatches, candidate.pp_size))
    updated = config.replace(parallel=parallel)
    if candidate.mode == "fsdp":
        updated = updated.replace(fsdp=replace(config.fsdp, enabled=True))
    return updated
