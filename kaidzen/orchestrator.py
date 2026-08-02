"""Оркестратор: критерии остановки, отбор допущений и сам цикл с возобновлением."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

from kaidzen.candidate import Candidate, LoopConfig
from kaidzen.roles.analyzer import run_analyzer
from kaidzen.roles.judge import run_judge
from kaidzen.roles.refiner import run_refiner
from kaidzen.roles.researcher import run_researcher
from kaidzen.state import (Analysis, ApiUsage, Assumption, JudgeResult,
                           RunState, Version, load_state, save_state)

T = TypeVar("T")

# два подряд отката = идея деградирует, дальше крутить бессмысленно
ROLLBACK_LIMIT = 2
# два подряд слабых прироста = плато
PLATEAU_STREAK = 2

# статусы, после которых допущение больше не проверяется;
# partial сюда НЕ входит — его ещё можно доуточнить
CLOSED_STATUSES = frozenset({"confirmed", "refuted", "untestable"})
OPEN_STATUS = "unverified"

# порядок отбора: сначала самые критичные допущения
CRITICALITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# ретраи транзиентных ошибок API вокруг каждого вызова роли
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2.0
RETRY_BACKOFF_FACTOR = 2.0

STEP_ANALYZER = "analyzer"
STEP_RESEARCHER = "researcher"
STEP_REFINER = "refiner"
STEP_JUDGE = "judge"
# шаги, на которых возобновление входит в середину итерации;
# остальные значения last_completed_step означают границу итерации
MID_ITERATION_STEPS = (STEP_RESEARCHER, STEP_REFINER)

STOP_MAX_ITERATIONS = "max_iterations"
STOP_DEGRADING = "degrading"
STOP_PLATEAU = "plateau"
STOP_ASSUMPTIONS_EXHAUSTED = "assumptions_exhausted"


def check_stop(state: RunState, loop: LoopConfig) -> str | None:
    """Причина остановки цикла или None. Порядок приоритетов важен:
    жёсткий лимит итераций перекрывает всё остальное, деградация — плато.
    """
    if state.iteration >= loop.max_iterations:
        return STOP_MAX_ITERATIONS
    if state.consecutive_rollbacks >= ROLLBACK_LIMIT:
        return STOP_DEGRADING
    if state.low_delta_streak >= PLATEAU_STREAK:
        return STOP_PLATEAU
    if _high_assumptions_closed(state.assumptions):
        return STOP_ASSUMPTIONS_EXHAUSTED
    return None


def _high_assumptions_closed(assumptions: list[Assumption]) -> bool:
    """Все критичные допущения закрыты. Пустой реестр критичных не считается."""
    high = [a for a in assumptions if a.criticality == "high"]
    if not high:
        return False
    return all(a.status in CLOSED_STATUSES for a in high)


def select_assumptions(assumptions: list[Assumption],
                       limit: int) -> list[Assumption]:
    """До limit непроверенных допущений: сначала high, затем medium, затем low.
    Внутри одной критичности порядок реестра сохраняется (sorted стабилен).
    """
    open_ones = [a for a in assumptions if a.status == OPEN_STATUS]
    ordered = sorted(open_ones, key=lambda a: CRITICALITY_ORDER[a.criticality])
    return ordered[:limit]


def apply_judge_verdict(state: RunState, judge: JudgeResult,
                        loop: LoopConfig) -> None:
    """Применяет вердикт к последней версии и обновляет счётчики остановки.

    При откате результат Judge к версии НЕ прикрепляется: версия выброшена,
    и её оценка не должна попадать в отчёт как оценка идеи.
    """
    version = state.versions[-1]
    if judge.verdict == "rollback":
        version.rolled_back = True
        state.rollbacks += 1
        state.consecutive_rollbacks += 1
        return

    version.judge = judge
    state.consecutive_rollbacks = 0
    if judge.delta_vs_previous < loop.plateau_threshold:
        state.low_delta_streak += 1
    else:
        state.low_delta_streak = 0


def run_pipeline(llm, candidate: Candidate, *, idea_text: str, run_dir: Path,
                 resume: bool = False,
                 on_step: Callable[[str, RunState], None] | None = None
                 ) -> RunState:
    """Полный прогон: Analyzer один раз, затем цикл Researcher→Refiner→Judge.

    Состояние сохраняется после каждого завершённого шага, поэтому прогон,
    убитый Ctrl+C или падением API, продолжается с того же места (resume=True)
    и не переплачивает за уже сделанные шаги.

    on_step, если задан, вызывается после каждого завершённого шага (уже
    сохранённого в state.json) с именем шага и текущим состоянием — это
    единственный способ показать пользователю живой прогресс во время
    платного многоминутного прогона.
    """
    state = load_state(run_dir) if resume else _new_state(candidate, idea_text,
                                                          run_dir)
    loop = candidate.config.loop
    if state.analysis is None:
        _step_analyzer(llm, candidate, state, run_dir, on_step)

    # с какого шага продолжаем прерванную итерацию (None = с её начала)
    entry = state.last_completed_step if resume else None
    entry = entry if entry in MID_ITERATION_STEPS else None
    while True:
        if entry is None:
            reason = check_stop(state, loop)
            if reason:
                return _finish(state, run_dir, llm, reason)
            if not _step_researcher(llm, candidate, state, run_dir, loop, on_step):
                return _finish(state, run_dir, llm, STOP_ASSUMPTIONS_EXHAUSTED)
        if entry in (None, STEP_RESEARCHER):
            _step_refiner(llm, candidate, state, run_dir, on_step)
        _step_judge(llm, candidate, state, run_dir, loop, on_step)
        entry = None


def _new_state(candidate: Candidate, idea_text: str, run_dir: Path) -> RunState:
    return RunState(run_id=run_dir.name, candidate_id=candidate.candidate_id,
                    config=candidate.config.model_dump(),
                    original_idea=idea_text)


def _step_analyzer(llm, candidate: Candidate, state: RunState,
                   run_dir: Path,
                   on_step: Callable[[str, RunState], None] | None = None
                   ) -> None:
    """Раскладывает сырую идею и заполняет реестр допущений."""
    out = _with_retry(lambda: run_analyzer(llm, candidate,
                                           idea_text=state.original_idea))
    state.analysis = Analysis(problem=out.problem, audience=out.audience,
                              mechanism=out.mechanism, unknowns=out.unknowns)
    state.assumptions = list(out.assumptions)
    _checkpoint(state, run_dir, llm, STEP_ANALYZER, on_step)


def _step_researcher(llm, candidate: Candidate, state: RunState, run_dir: Path,
                     loop: LoopConfig,
                     on_step: Callable[[str, RunState], None] | None = None
                     ) -> bool:
    """Проверяет самые рисковые допущения. False = проверять больше нечего."""
    selected = select_assumptions(state.assumptions,
                                  loop.assumptions_per_iteration)
    if not selected:
        return False
    out = _with_retry(lambda: run_researcher(
        llm, candidate, idea_text=state.current_idea_text(),
        assumptions=selected))
    _apply_findings(state, out)
    # находки нужны Refiner'у целиком; в реестр попадают только статус и факты,
    # поэтому сырой JSON кладём в состояние — иначе возобновление их потеряет
    state.pending_findings = out.model_dump_json()
    _checkpoint(state, run_dir, llm, STEP_RESEARCHER, on_step)
    return True


def _apply_findings(state: RunState, researcher_output) -> None:
    """Переносит вердикты и факты в реестр допущений."""
    by_id = {a.id: a for a in state.assumptions}
    for finding in researcher_output.findings:
        assumption = by_id.get(finding.assumption_id)
        if assumption is None:
            continue  # модель придумала id — молча игнорируем
        assumption.status = finding.verdict
        assumption.facts = assumption.facts + list(finding.facts)


def _step_refiner(llm, candidate: Candidate, state: RunState,
                  run_dir: Path,
                  on_step: Callable[[str, RunState], None] | None = None
                  ) -> None:
    """Переписывает идею по находкам и критике прошлого Judge."""
    current = state.current_version()
    critique = current.judge.critique if current and current.judge else []
    out = _with_retry(lambda: run_refiner(
        llm, candidate, idea_text=state.current_idea_text(),
        findings_json=state.pending_findings or "", critique=critique))
    state.versions.append(Version(n=len(state.versions) + 1,
                                  idea_text=out.idea_text,
                                  changelog=out.changelog))
    _checkpoint(state, run_dir, llm, STEP_REFINER, on_step)


def _step_judge(llm, candidate: Candidate, state: RunState, run_dir: Path,
                loop: LoopConfig,
                on_step: Callable[[str, RunState], None] | None = None
                ) -> None:
    """Оценивает новую версию и применяет вердикт."""
    new_version = state.versions[-1]
    judge = _with_retry(lambda: run_judge(
        llm, candidate, new_idea=new_version.idea_text,
        previous_idea=_idea_before_last_version(state),
        assumptions=state.assumptions))
    apply_judge_verdict(state, judge, loop)
    state.iteration += 1
    state.pending_findings = None
    _checkpoint(state, run_dir, llm, STEP_JUDGE, on_step)


def _idea_before_last_version(state: RunState) -> str:
    """Текст идеи до последней версии — вход Refiner'а этой итерации.

    Считается по состоянию, а не запоминается в переменной, чтобы Judge получил
    тот же текст и при возобновлении после шага refiner.
    """
    for version in reversed(state.versions[:-1]):
        if not version.rolled_back:
            return version.idea_text
    return state.original_idea


def _with_retry(call: Callable[[], T]) -> T:
    """Три попытки с экспоненциальной паузой: сеть и 5xx лечатся повтором.

    Если не помогло — исключение уходит наверх, состояние на диске при этом
    уже отражает последний завершённый шаг.
    """
    delay = RETRY_BASE_DELAY_SECONDS
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return call()
        except Exception:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= RETRY_BACKOFF_FACTOR
    raise AssertionError("недостижимо")  # pragma: no cover


def _checkpoint(state: RunState, run_dir: Path, llm, step: str,
                on_step: Callable[[str, RunState], None] | None = None
                ) -> None:
    state.last_completed_step = step
    _save(state, run_dir, llm)
    if on_step is None:
        return
    try:
        on_step(step, state)
    except Exception:
        # прогресс в stdout — украшение, а не часть платного прогона:
        # упавший колбэк не должен убивать уже сделанную (и оплаченную) работу
        pass


def _finish(state: RunState, run_dir: Path, llm, reason: str) -> RunState:
    state.stop_reason = reason
    _save(state, run_dir, llm)
    return state


def _save(state: RunState, run_dir: Path, llm) -> None:
    """Атомарная запись состояния вместе со свежим расходом токенов."""
    usage: Optional[ApiUsage] = getattr(llm, "usage", None)
    if usage is not None:
        state.api_usage = ApiUsage.model_validate(usage, from_attributes=True)
    save_state(state, run_dir)
