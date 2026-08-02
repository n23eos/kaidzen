import pytest

from kaidzen import orchestrator
from kaidzen.candidate import Candidate, LoopConfig
from kaidzen.orchestrator import (apply_judge_verdict, check_stop,
                                  run_pipeline, select_assumptions)
from kaidzen.roles.analyzer import AnalyzerOutput
from kaidzen.roles.refiner import RefinerOutput
from kaidzen.roles.researcher import ResearchFinding, ResearcherOutput
from kaidzen.state import (ApiUsage, Assumption, Fact, JudgeResult, RunState,
                           Version, load_state)
from tests.conftest import FakeLLM


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


# --- run_pipeline -----------------------------------------------------------

FACT = Fact(claim="рынок растёт", source_url="https://example.com/report")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Ретраи не должны замедлять тесты."""
    monkeypatch.setattr(orchestrator.time, "sleep", lambda *_: None)


def with_loop(candidate: Candidate, **loop_fields) -> Candidate:
    """Копия кандидата с изменёнными параметрами цикла."""
    loop = candidate.config.loop.model_copy(update=loop_fields)
    config = candidate.config.model_copy(update={"loop": loop})
    return candidate.model_copy(update={"config": config})


def analyzer_out(*assumptions: Assumption) -> AnalyzerOutput:
    return AnalyzerOutput(problem="p", audience="a", mechanism="m",
                          assumptions=list(assumptions), unknowns=["u"])


def research_out(*pairs) -> ResearcherOutput:
    return ResearcherOutput(findings=[
        ResearchFinding(assumption_id=aid, verdict=verdict, facts=[FACT],
                        notes="n")
        for aid, verdict in pairs])


def refiner_out(text: str) -> RefinerOutput:
    return RefinerOutput(idea_text=text, changelog=[])


def high(aid: str) -> Assumption:
    return Assumption(id=aid, text=f"допущение {aid}", criticality="high")


def medium(aid: str) -> Assumption:
    return Assumption(id=aid, text=f"допущение {aid}", criticality="medium")


def test_pipeline_runs_one_iteration_and_stops_on_exhausted(candidate, tmp_path):
    run_dir = tmp_path / "run-1"
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(llm, candidate, idea_text="сырая идея",
                         run_dir=run_dir, resume=False)

    assert state.run_id == "run-1"
    assert state.candidate_id == candidate.candidate_id
    assert state.original_idea == "сырая идея"
    assert state.stop_reason == "assumptions_exhausted"
    assert state.iteration == 1
    assert state.analysis.problem == "p"
    assert state.assumptions[0].status == "confirmed"
    assert state.assumptions[0].facts == [FACT]
    assert state.current_idea_text() == "версия 1"
    assert state.versions[0].n == 1
    assert state.versions[0].judge is not None
    assert state.pending_findings is None
    assert load_state(run_dir) == state


def test_pipeline_passes_previous_idea_and_critique_to_roles(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(medium("A1"), medium("A2")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0, critique=["слабый рынок"]),
                   research_out(("A2", "confirmed")),
                   refiner_out("версия 2"),
                   make_judge(delta=2.0)])

    run_pipeline(llm, candidate, idea_text="сырая идея",
                 run_dir=tmp_path / "run-1",
                 resume=False)

    second_refiner = llm.calls[5]["user"]
    assert "версия 1" in second_refiner          # вход = текущая идея
    assert "слабый рынок" in second_refiner      # критика прошлого Judge
    second_judge = llm.calls[6]["user"]
    assert "версия 2" in second_judge and "версия 1" in second_judge


def test_pipeline_stops_when_no_unverified_assumptions_left(candidate, tmp_path):
    # все допущения medium: остановки по «критичные закрыты» не будет,
    # но Researcher'у уже нечего проверять
    llm = FakeLLM([analyzer_out(medium("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(llm, candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.stop_reason == "assumptions_exhausted"
    assert llm.outputs == []


def test_pipeline_stops_on_max_iterations(candidate, tmp_path):
    limited = with_loop(candidate, max_iterations=1)
    llm = FakeLLM([analyzer_out(medium("A1"), medium("A2")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(llm, limited, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.stop_reason == "max_iterations"
    assert state.iteration == 1


def test_pipeline_stops_on_plateau(candidate, tmp_path):
    loop_candidate = with_loop(candidate, assumptions_per_iteration=1)
    llm = FakeLLM([analyzer_out(medium("A1"), medium("A2"), medium("A3")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=0.1),
                   research_out(("A2", "confirmed")),
                   refiner_out("версия 2"),
                   make_judge(delta=0.1)])

    state = run_pipeline(llm, loop_candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.stop_reason == "plateau"
    assert state.iteration == 2
    assert state.low_delta_streak == 2


def test_pipeline_rollback_marks_version_and_keeps_previous_text(candidate,
                                                                 tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("плохая версия"),
                   make_judge(verdict="rollback", delta=-3.0)])

    state = run_pipeline(llm, candidate, idea_text="сырая идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.versions[0].rolled_back is True
    assert state.versions[0].judge is None
    assert state.rollbacks == 1
    assert state.current_idea_text() == "сырая идея"
    assert state.stop_reason == "assumptions_exhausted"


def test_pipeline_copies_usage_into_state(candidate, tmp_path):
    run_dir = tmp_path / "run-1"
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])
    llm.usage = ApiUsage(input_tokens=120, output_tokens=45, web_searches=3)

    state = run_pipeline(llm, candidate, idea_text="идея", run_dir=run_dir,
                         resume=False)

    assert state.api_usage == ApiUsage(input_tokens=120, output_tokens=45,
                                       web_searches=3)
    assert load_state(run_dir).api_usage.input_tokens == 120


def test_pipeline_retries_transient_errors(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   RuntimeError("503"),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(llm, candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.stop_reason == "assumptions_exhausted"
    assert llm.outputs == []


def test_pipeline_resumes_after_researcher_step(candidate, tmp_path):
    run_dir = tmp_path / "run-1"
    crashing = FakeLLM([analyzer_out(high("A1")),
                        research_out(("A1", "confirmed")),
                        RuntimeError("API упал"),
                        RuntimeError("API упал"),
                        RuntimeError("API упал")])

    with pytest.raises(RuntimeError, match="API упал"):
        run_pipeline(crashing, candidate, idea_text="сырая идея",
                     run_dir=run_dir, resume=False)

    saved = load_state(run_dir)
    assert saved.last_completed_step == "researcher"
    assert saved.assumptions[0].status == "confirmed"
    assert saved.pending_findings is not None
    assert saved.versions == []

    resumed_llm = FakeLLM([refiner_out("версия 1"), make_judge(delta=2.0)])
    state = run_pipeline(resumed_llm, candidate, idea_text="сырая идея",
                         run_dir=run_dir, resume=True)

    assert len(resumed_llm.calls) == 2          # Analyzer/Researcher не повторились
    assert state.stop_reason == "assumptions_exhausted"
    assert state.current_idea_text() == "версия 1"
    assert state.iteration == 1
    # находки прошлого шага дошли до Refiner
    assert "рынок растёт" in resumed_llm.calls[0]["user"]


def test_pipeline_resumes_after_refiner_step(candidate, tmp_path):
    run_dir = tmp_path / "run-1"
    crashing = FakeLLM([analyzer_out(high("A1")),
                        research_out(("A1", "confirmed")),
                        refiner_out("версия 1")]
                       + [RuntimeError("Judge упал")] * 3)

    with pytest.raises(RuntimeError, match="Judge упал"):
        run_pipeline(crashing, candidate, idea_text="сырая идея",
                     run_dir=run_dir, resume=False)

    assert load_state(run_dir).last_completed_step == "refiner"

    resumed_llm = FakeLLM([make_judge(delta=2.0)])
    state = run_pipeline(resumed_llm, candidate, idea_text="сырая идея",
                         run_dir=run_dir, resume=True)

    assert len(resumed_llm.calls) == 1
    assert len(state.versions) == 1             # версия не задвоилась
    assert state.versions[0].judge is not None
    assert state.stop_reason == "assumptions_exhausted"
    # Judge получил предыдущий текст, а не собственную новую версию
    assert "сырая идея" in resumed_llm.calls[0]["user"]


def test_pipeline_resumes_after_analyzer_step(candidate, tmp_path):
    run_dir = tmp_path / "run-1"
    crashing = FakeLLM([analyzer_out(high("A1"))]
                       + [RuntimeError("Researcher упал")] * 3)

    with pytest.raises(RuntimeError):
        run_pipeline(crashing, candidate, idea_text="идея", run_dir=run_dir,
                     resume=False)

    assert load_state(run_dir).last_completed_step == "analyzer"

    resumed_llm = FakeLLM([research_out(("A1", "confirmed")),
                           refiner_out("версия 1"), make_judge(delta=2.0)])
    state = run_pipeline(resumed_llm, candidate, idea_text="идея",
                         run_dir=run_dir, resume=True)

    assert len(resumed_llm.calls) == 3
    assert state.iteration == 1
