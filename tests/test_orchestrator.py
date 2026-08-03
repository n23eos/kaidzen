import anthropic
import httpx
import pytest

from kaidzen import orchestrator
from kaidzen.candidate import Candidate, LoopConfig
from kaidzen.llm import SchemaValidationFailure, SearchNotPerformed
from kaidzen.orchestrator import (apply_judge_verdict, check_stop,
                                  run_pipeline, select_assumptions)
from kaidzen.roles.analyzer import AnalyzerOutput
from kaidzen.roles.refiner import RefinerOutput
from kaidzen.roles.researcher import ResearchFinding, ResearcherOutput
from kaidzen.state import (ApiUsage, Assumption, ChangelogEntry, Fact,
                           JudgeResult, RunState, Version, load_state,
                           save_state)
from tests.conftest import FakeLLM, as_backends


def transient() -> Exception:
    """Настоящая транзиентная ошибка SDK — только такие имеет смысл повторять."""
    return anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def make_state(**kwargs) -> RunState:
    """Минимальный RunState; тесты переопределяют только нужные поля."""
    base = dict(run_id="r1", candidate_id="c1", config={}, original_idea="идея")
    base.update(kwargs)
    return RunState(**base)


# оси рубрики кандидата из фикстуры: Judge обязан вернуть ровно их
RUBRIC_AXES = ("novelty", "feasibility", "groundedness", "potential", "clarity")


def make_judge(*, total=10.0, delta=1.0, verdict="continue", critique=None):
    scores = {axis: total / len(RUBRIC_AXES) for axis in RUBRIC_AXES}
    return JudgeResult(scores=scores, total=sum(scores.values()),
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


def refiner_out(text: str, *grounded_in: str) -> RefinerOutput:
    """Валидный ответ Refiner: изменения всегда объяснены (см. проверку grounding)."""
    return RefinerOutput(idea_text=text, changelog=[
        ChangelogEntry(change=f"переписали под {text}", reason="находки",
                       grounded_in=list(grounded_in))])


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

    state = run_pipeline(as_backends(llm), candidate, idea_text="сырая идея",
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


def test_pipeline_records_candidate_dir_in_config_snapshot(candidate, tmp_path):
    """resume должен грузить промпты из ТОГО ЖЕ каталога, с которым прогон
    стартовал (см. __main__._resume_candidate_dir) — путь обязан попасть в
    state.config уже на первом checkpoint'е (после analyzer), а не только
    в конце прогона, иначе прерванный Ctrl+C прогон его не сохранит."""
    run_dir = tmp_path / "run-1"
    candidate_dir = tmp_path / "candidates" / "my-candidate"
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(as_backends(llm), candidate, idea_text="сырая идея",
                         run_dir=run_dir, resume=False,
                         candidate_dir=candidate_dir)

    assert state.config["candidate_dir"] == str(candidate_dir)


def test_pipeline_passes_previous_idea_and_critique_to_roles(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(medium("A1"), medium("A2")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0, critique=["слабый рынок"]),
                   research_out(("A2", "confirmed")),
                   refiner_out("версия 2"),
                   make_judge(delta=2.0)])

    run_pipeline(as_backends(llm), candidate, idea_text="сырая идея",
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

    state = run_pipeline(as_backends(llm), candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.stop_reason == "assumptions_exhausted"
    assert llm.outputs == []


def test_pipeline_stops_on_max_iterations(candidate, tmp_path):
    limited = with_loop(candidate, max_iterations=1)
    llm = FakeLLM([analyzer_out(medium("A1"), medium("A2")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(as_backends(llm), limited, idea_text="идея",
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

    state = run_pipeline(as_backends(llm), loop_candidate, idea_text="идея",
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

    state = run_pipeline(as_backends(llm), candidate, idea_text="сырая идея",
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

    state = run_pipeline(as_backends(llm), candidate, idea_text="идея", run_dir=run_dir,
                         resume=False)

    assert state.api_usage == ApiUsage(input_tokens=120, output_tokens=45,
                                       web_searches=3)
    assert load_state(run_dir).api_usage.input_tokens == 120


def test_pipeline_retries_transient_errors(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   transient(),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(as_backends(llm), candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.stop_reason == "assumptions_exhausted"
    assert llm.outputs == []


def test_pipeline_resumes_after_researcher_step(candidate, tmp_path):
    run_dir = tmp_path / "run-1"
    crashing = FakeLLM([analyzer_out(high("A1")),
                        research_out(("A1", "confirmed")),
                        RuntimeError("API упал")])

    with pytest.raises(RuntimeError, match="API упал"):
        run_pipeline(as_backends(crashing), candidate, idea_text="сырая идея",
                     run_dir=run_dir, resume=False)

    saved = load_state(run_dir)
    assert saved.last_completed_step == "researcher"
    assert saved.assumptions[0].status == "confirmed"
    assert saved.pending_findings is not None
    assert saved.versions == []

    resumed_llm = FakeLLM([refiner_out("версия 1"), make_judge(delta=2.0)])
    state = run_pipeline(as_backends(resumed_llm), candidate, idea_text="сырая идея",
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
                        refiner_out("версия 1"),
                        RuntimeError("Judge упал")])

    with pytest.raises(RuntimeError, match="Judge упал"):
        run_pipeline(as_backends(crashing), candidate, idea_text="сырая идея",
                     run_dir=run_dir, resume=False)

    assert load_state(run_dir).last_completed_step == "refiner"

    resumed_llm = FakeLLM([make_judge(delta=2.0)])
    state = run_pipeline(as_backends(resumed_llm), candidate, idea_text="сырая идея",
                         run_dir=run_dir, resume=True)

    assert len(resumed_llm.calls) == 1
    assert len(state.versions) == 1             # версия не задвоилась
    assert state.versions[0].judge is not None
    assert state.stop_reason == "assumptions_exhausted"
    # Judge получил предыдущий текст, а не собственную новую версию
    assert "сырая идея" in resumed_llm.calls[0]["user"]


def test_pipeline_resumes_after_analyzer_step(candidate, tmp_path):
    run_dir = tmp_path / "run-1"
    crashing = FakeLLM([analyzer_out(high("A1")),
                        RuntimeError("Researcher упал")])

    with pytest.raises(RuntimeError):
        run_pipeline(as_backends(crashing), candidate, idea_text="идея", run_dir=run_dir,
                     resume=False)

    assert load_state(run_dir).last_completed_step == "analyzer"

    resumed_llm = FakeLLM([research_out(("A1", "confirmed")),
                           refiner_out("версия 1"), make_judge(delta=2.0)])
    state = run_pipeline(as_backends(resumed_llm), candidate, idea_text="идея",
                         run_dir=run_dir, resume=True)

    assert len(resumed_llm.calls) == 3
    assert state.iteration == 1


# --- on_step callback --------------------------------------------------------

def test_pipeline_without_callback_behaves_as_before(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(as_backends(llm), candidate, idea_text="сырая идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.stop_reason == "assumptions_exhausted"


def test_pipeline_calls_on_step_once_per_step_in_order(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])
    calls: list[str] = []

    run_pipeline(as_backends(llm), candidate, idea_text="сырая идея",
                 run_dir=tmp_path / "run-1", resume=False,
                 on_step=lambda step, state: calls.append(step))

    assert calls == ["analyzer", "researcher", "refiner", "judge"]


def test_pipeline_on_step_sees_state_already_reflecting_the_step(candidate,
                                                                  tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])
    seen: dict[str, int] = {}

    def on_step(step, state):
        if step == "analyzer":
            seen["assumptions"] = len(state.assumptions)

    run_pipeline(as_backends(llm), candidate, idea_text="сырая идея",
                 run_dir=tmp_path / "run-1", resume=False, on_step=on_step)

    assert seen["assumptions"] == 1


def test_pipeline_survives_a_raising_callback(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1"),
                   make_judge(delta=2.0)])

    def boom(step, state):
        raise RuntimeError("проблема с выводом прогресса")

    state = run_pipeline(as_backends(llm), candidate, idea_text="сырая идея",
                         run_dir=tmp_path / "run-1", resume=False,
                         on_step=boom)

    assert state.stop_reason == "assumptions_exhausted"


# --- находки с чужими id допущений (FIX 1) ----------------------------------

def test_pipeline_fails_when_every_finding_has_unknown_id(candidate, tmp_path):
    # весь вызов Researcher оказался мусором: молча продолжать нельзя,
    # иначе прогон докрутит бюджет до конца и не проверит ничего
    llm = FakeLLM([analyzer_out(high("A1"))]
                  + [research_out(("A-1", "confirmed"))] * 3)

    with pytest.raises(ValueError, match="A-1"):
        run_pipeline(as_backends(llm), candidate, idea_text="идея",
                     run_dir=tmp_path / "run-1", resume=False)


def test_pipeline_retries_researcher_after_unknown_ids(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("1", "confirmed")),      # выдуманный id
                   research_out(("A1", "confirmed")),     # повтор удался
                   refiner_out("версия 1", "A1"),
                   make_judge(delta=2.0)])

    state = run_pipeline(as_backends(llm), candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.assumptions[0].status == "confirmed"
    assert state.stop_reason == "assumptions_exhausted"


def test_pipeline_applies_known_ids_and_reports_unknown_ones(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(high("A1"), medium("A2")),
                   research_out(("A1", "confirmed"), ("A99", "confirmed")),
                   refiner_out("версия 1", "A1"),
                   make_judge(delta=2.0)])
    steps: list[str] = []

    state = run_pipeline(as_backends(llm), candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False,
                         on_step=lambda step, _state: steps.append(step))

    assert state.assumptions[0].status == "confirmed"     # A1 применён
    assert state.assumptions[1].status == "unverified"    # A2 не тронут
    warnings = [s for s in steps if s.startswith(orchestrator.STEP_WARNING)]
    assert len(warnings) == 1 and "A99" in warnings[0]
    # предупреждение не должно подменять последний завершённый шаг
    assert load_state(tmp_path / "run-1").last_completed_step == "judge"


# --- resume берёт параметры цикла из state (FIX 2) ---------------------------

def crash_after_researcher(candidate, tmp_path, run_dir) -> None:
    """Роняет прогон сразу после шага Researcher, оставляя state на диске."""
    crashing = FakeLLM([analyzer_out(medium("A1"), medium("A2"), medium("A3")),
                        research_out(("A1", "confirmed")),
                        RuntimeError("API упал")])
    with pytest.raises(RuntimeError):
        run_pipeline(as_backends(crashing), candidate, idea_text="идея", run_dir=run_dir,
                     resume=False)


def test_resume_uses_loop_config_from_state_not_from_candidate(candidate,
                                                               tmp_path):
    run_dir = tmp_path / "run-1"
    started_with = with_loop(candidate, max_iterations=1,
                             assumptions_per_iteration=1)
    crash_after_researcher(started_with, tmp_path, run_dir)
    assert load_state(run_dir).config["loop"]["max_iterations"] == 1

    # кандидат «с диска» разрешает 6 итераций — но прогон стартовал с 1
    resumed_llm = FakeLLM([refiner_out("версия 1", "A1"), make_judge(delta=2.0)])
    state = run_pipeline(as_backends(resumed_llm), candidate, idea_text="идея",
                         run_dir=run_dir, resume=True)

    assert state.stop_reason == "max_iterations"
    assert state.iteration == 1
    assert resumed_llm.outputs == []


def test_resume_fails_loudly_when_state_has_no_loop_snapshot(tmp_path,
                                                             candidate):
    run_dir = tmp_path / "run-1"
    save_state(RunState(run_id="run-1", candidate_id="c", config={},
                        original_idea="идея", last_completed_step="judge"),
               run_dir)

    with pytest.raises(ValueError, match="loop"):
        run_pipeline(as_backends(FakeLLM([])), candidate, idea_text="идея", run_dir=run_dir,
                     resume=True)


# --- Refiner обязан обосновывать изменения (FIX 3) ---------------------------

def test_pipeline_rejects_refiner_without_changelog(candidate, tmp_path):
    empty = RefinerOutput(idea_text="версия 1", changelog=[])
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed"))] + [empty] * 3)

    with pytest.raises(ValueError, match="changelog"):
        run_pipeline(as_backends(llm), candidate, idea_text="идея",
                     run_dir=tmp_path / "run-1", resume=False)


def test_pipeline_rejects_changelog_grounded_in_unknown_assumption(candidate,
                                                                    tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed"))]
                  + [refiner_out("версия 1", "A7")] * 3)

    with pytest.raises(ValueError, match="A7"):
        run_pipeline(as_backends(llm), candidate, idea_text="идея",
                     run_dir=tmp_path / "run-1", resume=False)


def test_pipeline_retries_refiner_after_bad_grounding(candidate, tmp_path):
    llm = FakeLLM([analyzer_out(high("A1")),
                   research_out(("A1", "confirmed")),
                   refiner_out("версия 1", "A7"),     # чужой id
                   refiner_out("версия 1", "A1"),     # повтор удался
                   make_judge(delta=2.0)])

    state = run_pipeline(as_backends(llm), candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert state.stop_reason == "assumptions_exhausted"
    assert state.versions[0].changelog[0].grounded_in == ["A1"]


# --- delta считается по сохранённым оценкам, а не со слов Judge (FIX 4) ------

def make_state_with_scored_previous(previous_total: float, **kwargs) -> RunState:
    """Версия v1 с оценкой previous_total и свежая неоценённая v2."""
    state = make_state(**kwargs)
    state.versions.append(Version(n=1, idea_text="v1",
                                  judge=make_judge(total=previous_total,
                                                   delta=previous_total)))
    state.versions.append(Version(n=2, idea_text="v2"))
    return state


def test_apply_judge_verdict_recomputes_delta_and_sees_plateau():
    # модель рапортует бодрый прирост, а по факту total почти не вырос
    state = make_state_with_scored_previous(8.0, low_delta_streak=0)
    apply_judge_verdict(state, make_judge(total=8.2, delta=5.0), LOOP)
    assert state.low_delta_streak == 1


def test_apply_judge_verdict_keeps_reported_delta_for_display():
    state = make_state_with_scored_previous(8.0)
    apply_judge_verdict(state, make_judge(total=8.2, delta=5.0), LOOP)
    assert state.versions[-1].judge.delta_vs_previous == 5.0


def test_apply_judge_verdict_recomputed_growth_beats_pessimistic_report():
    state = make_state_with_scored_previous(4.0, low_delta_streak=1)
    apply_judge_verdict(state, make_judge(total=9.0, delta=0.0), LOOP)
    assert state.low_delta_streak == 0


def test_apply_judge_verdict_rolls_back_when_recomputed_delta_is_negative():
    # Judge поставил меньше, чем прошлой версии, но объявил verdict=continue
    state = make_state_with_scored_previous(9.0)
    apply_judge_verdict(state, make_judge(total=6.0, delta=3.0,
                                          verdict="continue"), LOOP)
    assert state.versions[-1].rolled_back is True
    assert state.versions[-1].judge is None
    assert state.consecutive_rollbacks == 1


def test_apply_judge_verdict_trusts_reported_delta_without_previous_score():
    # сравнивать не с чем: первая версия оценивается относительно сырой идеи
    state = make_state_with_version()
    apply_judge_verdict(state, make_judge(total=2.0, delta=2.0), LOOP)
    assert state.low_delta_streak == 0
    assert state.versions[-1].rolled_back is False


def test_apply_judge_verdict_ignores_rolled_back_version_as_baseline():
    state = make_state_with_scored_previous(8.0)
    state.versions[0].rolled_back = True
    state.versions[0].judge = None
    apply_judge_verdict(state, make_judge(total=1.0, delta=0.9), LOOP)
    # базы для пересчёта нет — работаем по заявленной дельте, отката нет
    assert state.versions[-1].rolled_back is False
    assert state.low_delta_streak == 0


# --- критика откаченной версии доходит до следующего Refiner (FIX 6) ---------

def test_apply_judge_verdict_stores_critique_even_on_rollback():
    state = make_state_with_version()
    apply_judge_verdict(state, make_judge(verdict="rollback", delta=-2.0,
                                          critique=["слишком общо"]), LOOP)
    assert state.last_critique == ["слишком общо"]


def test_pipeline_feeds_rolled_back_critique_to_next_refiner(candidate,
                                                             tmp_path):
    loop_candidate = with_loop(candidate, assumptions_per_iteration=1)
    llm = FakeLLM([analyzer_out(medium("A1"), medium("A2")),
                   research_out(("A1", "confirmed")),
                   refiner_out("плохая версия", "A1"),
                   make_judge(verdict="rollback", delta=-3.0,
                              critique=["выкинуты цифры из фактов"]),
                   research_out(("A2", "confirmed")),
                   refiner_out("версия 2", "A2"),
                   make_judge(delta=2.0)])

    run_pipeline(as_backends(llm), loop_candidate, idea_text="сырая идея",
                 run_dir=tmp_path / "run-1", resume=False)

    second_refiner = llm.calls[5]["user"]
    assert "выкинуты цифры из фактов" in second_refiner


# --- ретраятся только транзиентные ошибки (FIX 5) ----------------------------

@pytest.mark.parametrize("error", [
    SchemaValidationFailure("схема не сошлась"),
    RuntimeError("что-то совсем другое"),
])
def test_pipeline_does_not_retry_deterministic_failures(candidate, tmp_path,
                                                        error):
    llm = FakeLLM([analyzer_out(high("A1")), error])

    with pytest.raises(type(error)):
        run_pipeline(as_backends(llm), candidate, idea_text="идея",
                     run_dir=tmp_path / "run-1", resume=False)

    assert len(llm.calls) == 2      # ни одного лишнего платного вызова


# --- у каждой роли свой бэкенд (мультибэкенд) --------------------------------

def test_pipeline_calls_the_backend_of_each_role(candidate, tmp_path):
    """Роль обязана ходить в СВОЙ бэкенд: иначе «дешёвый Judge на DeepSeek»
    молча оплачивался бы ключом соседней роли."""
    # Arrange
    analyzer = FakeLLM([analyzer_out(high("A1"))])
    researcher = FakeLLM([research_out(("A1", "confirmed"))])
    refiner = FakeLLM([refiner_out("версия 1", "A1")])
    judge = FakeLLM([make_judge(delta=2.0)])
    backends = as_backends(FakeLLM([]), analyzer=analyzer, researcher=researcher,
                           refiner=refiner, judge=judge)

    # Act
    run_pipeline(backends, candidate, idea_text="идея",
                 run_dir=tmp_path / "run-1", resume=False)

    # Assert
    assert len(analyzer.calls) == 1
    assert analyzer.calls[0]["schema"] is AnalyzerOutput
    assert len(researcher.calls) == 1
    assert researcher.calls[0]["web_search"] is True
    assert len(refiner.calls) == 1
    assert len(judge.calls) == 1


def test_pipeline_sums_usage_of_all_backends(candidate, tmp_path):
    """Счётчик у каждого бэкенда свой — в state должна лечь общая сумма."""
    # Arrange
    run_dir = tmp_path / "run-1"
    cheap = FakeLLM([analyzer_out(high("A1")), refiner_out("версия 1", "A1"),
                     make_judge(delta=2.0)],
                    usage=ApiUsage(input_tokens=100, output_tokens=10,
                                   web_searches=0))
    searching = FakeLLM([research_out(("A1", "confirmed"))],
                        usage=ApiUsage(input_tokens=20, output_tokens=5,
                                       web_searches=3))
    backends = as_backends(cheap, researcher=searching)

    # Act
    state = run_pipeline(backends, candidate, idea_text="идея",
                         run_dir=run_dir, resume=False)

    # Assert
    assert state.api_usage == ApiUsage(input_tokens=120, output_tokens=15,
                                       web_searches=3)
    assert load_state(run_dir).api_usage.web_searches == 3


# --- Researcher обязан реально искать (ТЗ §4) --------------------------------

def test_pipeline_retries_researcher_when_no_search_was_performed(candidate,
                                                                  tmp_path):
    searching = FakeLLM([SearchNotPerformed("поиск не выполнен"),
                         research_out(("A1", "confirmed"))])
    backends = as_backends(FakeLLM([analyzer_out(high("A1")),
                                    refiner_out("версия 1", "A1"),
                                    make_judge(delta=2.0)]),
                           researcher=searching)

    state = run_pipeline(backends, candidate, idea_text="идея",
                         run_dir=tmp_path / "run-1", resume=False)

    assert len(searching.calls) == 2
    assert state.assumptions[0].status == "confirmed"


def test_pipeline_fails_step_when_search_never_happens(candidate, tmp_path):
    """Исчерпав попытки, шаг падает: выдуманный источник в отчёте хуже отказа."""
    searching = FakeLLM([SearchNotPerformed("поиск не выполнен")] * 3)
    backends = as_backends(FakeLLM([analyzer_out(high("A1"))]),
                           researcher=searching)

    with pytest.raises(SearchNotPerformed):
        run_pipeline(backends, candidate, idea_text="идея",
                     run_dir=tmp_path / "run-1", resume=False)

    assert len(searching.calls) == 3      # ровно RETRY_ATTEMPTS попыток


# --- снапшот конфига старой формы (models:) ----------------------------------

def test_resume_refuses_old_shape_config_snapshot(candidate, tmp_path):
    """Прогон, начатый до сменных бэкендов, продолжать нечем: неизвестно,
    какими бэкендами он шёл. Явный отказ вместо тихого продолжения."""
    run_dir = tmp_path / "run-1"
    save_state(RunState(run_id="run-1", candidate_id="c",
                        config={"loop": LOOP.model_dump(),
                                "models": {"analyzer": "claude-sonnet-5"}},
                        original_idea="идея", last_completed_step="judge"),
               run_dir)

    with pytest.raises(ValueError) as exc:
        run_pipeline(as_backends(FakeLLM([])), candidate, idea_text="идея",
                     run_dir=run_dir, resume=True)

    assert "report" in str(exc.value)
