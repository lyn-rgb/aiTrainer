from aitrainer import FrameworkConfig, PlanCandidate, PlanningConfig, PlanningError, apply_candidate, suggest_plan


class Layer:
    def children(self):
        return iter((object(), object()))

    def parameters(self):
        return ()


def test_disabled_planner_is_read_only():
    report = suggest_plan(Layer(), config=FrameworkConfig(), world_size=1)
    assert not report.enabled and report.selected is None


def test_planner_apply_requires_explicit_opt_in():
    config = FrameworkConfig(planning=PlanningConfig(enabled=True))
    try:
        apply_candidate(config, PlanCandidate(1, 1, 1, "eager", 0.0))
    except PlanningError:
        return
    raise AssertionError("planner must not rewrite config without explicit allow_rewrite")
