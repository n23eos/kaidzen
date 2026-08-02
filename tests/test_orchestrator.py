import pytest

from kaidzen.candidate import LoopConfig
from kaidzen.orchestrator import (apply_judge_verdict, check_stop,
                                  select_assumptions)
from kaidzen.state import Assumption, JudgeResult, RunState, Version


def make_state(**kwargs) -> RunState:
    """Минимальный RunState; тесты переопределяют только нужные поля."""
    base = dict(run_id="r1", candidate_id="c1", config={}, original_idea="идея")
    base.update(kwargs)
    return RunState(**base)


def make_judge(*, total=10.0, delta=1.0, verdict="continue", critique=None):
    half = total / 2
    return JudgeResult(scores={"a": half, "b": half}, total=total,
                       delta_vs_previous=delta, verdict=verdict,
                       critique=critique if critique is not None else ["c"])


LOOP = LoopConfig(max_iterations=6, plateau_threshold=0.5,
                  assumptions_per_iteration=3)


# --- check_stop -------------------------------------------------------------

def test_check_stop_returns_none_mid_run():
    state = make_state(iteration=2,
                       assumptions=[Assumption(id="A1", text="x",
                                               criticality="high")])
    assert check_stop(state, LOOP) is None


def test_check_stop_max_iterations():
    assert check_stop(make_state(iteration=6), LOOP) == "max_iterations"


def test_check_stop_degrading():
    state = make_state(iteration=1, consecutive_rollbacks=2)
    assert check_stop(state, LOOP) == "degrading"


def test_check_stop_plateau():
    state = make_state(iteration=1, low_delta_streak=2)
    assert check_stop(state, LOOP) == "plateau"


def test_check_stop_assumptions_exhausted():
    state = make_state(iteration=1, assumptions=[
        Assumption(id="A1", text="x", criticality="high", status="confirmed"),
        Assumption(id="A2", text="y", criticality="high", status="untestable"),
        Assumption(id="A3", text="z", criticality="low", status="unverified"),
    ])
    assert check_stop(state, LOOP) == "assumptions_exhausted"


def test_check_stop_not_exhausted_when_high_assumption_partial():
    # partial — не закрытый статус: допущение можно уточнять дальше
    state = make_state(iteration=1, assumptions=[
        Assumption(id="A1", text="x", criticality="high", status="partial"),
    ])
    assert check_stop(state, LOOP) is None


def test_check_stop_ignores_empty_high_registry():
    state = make_state(iteration=1, assumptions=[
        Assumption(id="A1", text="x", criticality="low", status="confirmed"),
    ])
    assert check_stop(state, LOOP) is None


def test_check_stop_priority_max_iterations_over_plateau():
    state = make_state(iteration=6, low_delta_streak=5, consecutive_rollbacks=3)
    assert check_stop(state, LOOP) == "max_iterations"


def test_check_stop_priority_degrading_over_plateau():
    state = make_state(iteration=1, low_delta_streak=5, consecutive_rollbacks=2)
    assert check_stop(state, LOOP) == "degrading"


def test_check_stop_priority_plateau_over_exhausted():
    state = make_state(iteration=1, low_delta_streak=2, assumptions=[
        Assumption(id="A1", text="x", criticality="high", status="confirmed"),
    ])
    assert check_stop(state, LOOP) == "plateau"


# --- select_assumptions -----------------------------------------------------

def test_select_assumptions_orders_by_criticality_and_respects_limit():
    assumptions = [
        Assumption(id="A1", text="a", criticality="low"),
        Assumption(id="A2", text="b", criticality="medium"),
        Assumption(id="A3", text="c", criticality="high"),
        Assumption(id="A4", text="d", criticality="high"),
    ]
    picked = select_assumptions(assumptions, 3)
    assert [a.id for a in picked] == ["A3", "A4", "A2"]


def test_select_assumptions_skips_closed_ones():
    assumptions = [
        Assumption(id="A1", text="a", criticality="high", status="confirmed"),
        Assumption(id="A2", text="b", criticality="high", status="refuted"),
        Assumption(id="A3", text="c", criticality="high", status="partial"),
        Assumption(id="A4", text="d", criticality="medium"),
    ]
    picked = select_assumptions(assumptions, 5)
    # partial тоже не берём: выбираем только status == unverified
    assert [a.id for a in picked] == ["A4"]


def test_select_assumptions_returns_empty_when_nothing_left():
    assumptions = [Assumption(id="A1", text="a", criticality="high",
                              status="confirmed")]
    assert select_assumptions(assumptions, 3) == []


def test_select_assumptions_preserves_order_inside_criticality():
    assumptions = [Assumption(id=f"A{i}", text="x", criticality="high")
                   for i in range(5)]
    assert [a.id for a in select_assumptions(assumptions, 3)] == ["A0", "A1", "A2"]


# --- apply_judge_verdict ----------------------------------------------------

def make_state_with_version(**kwargs) -> RunState:
    state = make_state(**kwargs)
    state.versions.append(Version(n=1, idea_text="v1"))
    return state


def test_apply_judge_verdict_rollback():
    state = make_state_with_version(consecutive_rollbacks=1, rollbacks=1,
                                    low_delta_streak=1)
    apply_judge_verdict(state, make_judge(verdict="rollback", delta=-2.0), LOOP)
    version = state.versions[-1]
    assert version.rolled_back is True
    assert version.judge is None
    assert state.rollbacks == 2
    assert state.consecutive_rollbacks == 2
    # серия низких дельт при откате не трогается: версия просто выброшена
    assert state.low_delta_streak == 1


def test_apply_judge_verdict_continue_resets_rollbacks_and_streak():
    state = make_state_with_version(consecutive_rollbacks=1, low_delta_streak=1)
    judge = make_judge(verdict="continue", delta=2.0)
    apply_judge_verdict(state, judge, LOOP)
    assert state.versions[-1].judge == judge
    assert state.versions[-1].rolled_back is False
    assert state.consecutive_rollbacks == 0
    assert state.low_delta_streak == 0


def test_apply_judge_verdict_increments_low_delta_streak():
    state = make_state_with_version(low_delta_streak=1)
    apply_judge_verdict(state, make_judge(delta=0.2), LOOP)
    assert state.low_delta_streak == 2


def test_apply_judge_verdict_delta_equal_to_threshold_is_not_plateau():
    state = make_state_with_version(low_delta_streak=1)
    apply_judge_verdict(state, make_judge(delta=LOOP.plateau_threshold), LOOP)
    assert state.low_delta_streak == 0
