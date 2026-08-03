"""Оркестратор: критерии остановки, отбор допущений и сам цикл с возобновлением."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, TypeVar

import anthropic

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

# статусы, после которых допущение больше не проверяется циклом остановки;
# partial сюда НЕ входит: пока критичное допущение подтверждено лишь частично,
# «допущения исчерпаны» не наступает и цикл продолжает работать над идеей
CLOSED_STATUSES = frozenset({"confirmed", "refuted", "untestable"})
# Researcher получает только допущения в этом статусе: каждое допущение
# исследуется ровно один раз, а бюджет итераций уходит на ещё не тронутые.
# partial повторно НЕ исследуется — это и гарантирует остановку цикла.
OPEN_STATUS = "unverified"

# порядок отбора: сначала самые критичные допущения
CRITICALITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# ретраи транзиентных ошибок API вокруг каждого вызова роли
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 2.0
RETRY_BACKOFF_FACTOR = 2.0
# Повторяем только то, что реально может пройти со второй попытки: сеть, 429,
# 5xx — и наши собственные ValueError, которыми роли отвергают бессмысленный
# ответ модели (повтор = новый промпт, у него есть шанс). Всё остальное
# (обрезанный ответ, провал схемы, 400) детерминировано и при повторе просто
# оплачивается втройне, поэтому уходит наверх сразу.
RETRYABLE_ERRORS = (anthropic.APIConnectionError, anthropic.RateLimitError,
                    anthropic.InternalServerError, ValueError)

STEP_ANALYZER = "analyzer"
STEP_RESEARCHER = "researcher"
STEP_REFINER = "refiner"
STEP_JUDGE = "judge"
# префикс «шага», через который оркестратор сообщает наверх о проблеме,
# не отменяющей прогон: у оркестратора нет вывода, есть только on_step
STEP_WARNING = "warning"
# шаги, на которых возобновление входит в середину итерации;
# остальные значения last_completed_step означают границу итерации
MID_ITERATION_STEPS = (STEP_RESEARCHER, STEP_REFINER)

# ключи снапшота конфига: по ним отличается прогон, начатый до сменных бэкендов
ROLES_KEY = "roles"
LEGACY_MODELS_KEY = "models"

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

    delta_vs_previous, присланная моделью, на управление циклом НЕ влияет:
    Judge сам себе выставляет оценку и сам же сообщает прирост, а на этом
    числе висит остановка по плато. Прирост пересчитывается по сохранённому
    total предыдущей оценённой версии; заявленное значение остаётся в
    JudgeResult только для отчёта. Версия, набравшая меньше предыдущей, — это
    деградация независимо от вердикта, поэтому она откатывается.

    При откате результат Judge к версии НЕ прикрепляется: версия выброшена,
    и её оценка не должна попадать в отчёт как оценка идеи. Критика при этом
    сохраняется в state.last_critique — именно она объясняет следующему
    Refiner'у, почему прошлая попытка оказалась хуже.
    """
    version = state.versions[-1]
    state.last_critique = list(judge.critique)
    previous_total = _previous_scored_total(state)
    delta = (judge.total - previous_total if previous_total is not None
             else judge.delta_vs_previous)
    is_worse = previous_total is not None and delta < 0

    if judge.verdict == "rollback" or is_worse:
        version.rolled_back = True
        state.rollbacks += 1
        state.consecutive_rollbacks += 1
        return

    version.judge = judge
    state.consecutive_rollbacks = 0
    if delta < loop.plateau_threshold:
        state.low_delta_streak += 1
    else:
        state.low_delta_streak = 0


def _previous_scored_total(state: RunState) -> float | None:
    """total последней НЕ откаченной оценённой версии — база для прироста.

    Откаченные версии пропускаются: Refiner переписывал идею не от них,
    сравнивать новую версию с выброшенной было бы сравнением не с тем текстом.
    None означает, что сравнивать пока не с чем (первая оценка в прогоне).
    """
    for version in reversed(state.versions[:-1]):
        if not version.rolled_back and version.judge is not None:
            return version.judge.total
    return None


def run_pipeline(backends, candidate: Candidate, *, idea_text: str, run_dir: Path,
                 resume: bool = False,
                 candidate_dir: Path | None = None,
                 on_step: Callable[[str, RunState], None] | None = None
                 ) -> RunState:
    """Полный прогон: Analyzer один раз, затем цикл Researcher→Refiner→Judge.

    backends — отображение «имя роли → её бэкенд» (candidate.backends_by_role).
    У каждой роли свой транспорт; логика цикла о нём ничего не знает и лишь
    суммирует расход всех бэкендов в state.api_usage.

    Состояние сохраняется после каждого завершённого шага, поэтому прогон,
    убитый Ctrl+C или падением API, продолжается с того же места (resume=True)
    и не переплачивает за уже сделанные шаги.

    on_step, если задан, вызывается после каждого завершённого шага (уже
    сохранённого в state.json) с именем шага и текущим состоянием — это
    единственный способ показать пользователю живой прогресс во время
    платного многоминутного прогона. Тем же колбэком приходят предупреждения:
    шаг с именем, начинающимся на STEP_WARNING, — это не завершённый шаг, а
    сообщение о проблеме, которая не отменяет прогон.

    Контракт resume: параметры цикла (max_iterations, plateau_threshold,
    assumptions_per_iteration) берутся ИЗ СНАПШОТА state.config (ТЗ §5), а не
    из переданного кандидата. Кандидат при возобновлении нужен только ради
    промптов и имён моделей. Иначе прогон, запущенный с --max-iter 2, при
    продолжении получил бы лимит из config.yaml кандидата и потратил чужой
    бюджет; вызывающему коду достаточно загрузить кандидата по
    state.candidate_id, ничего не подмешивая в его конфиг.
    """
    state = load_state(run_dir) if resume else _new_state(candidate, idea_text,
                                                          run_dir, candidate_dir)
    if resume:
        _check_snapshot_shape(state)
    loop = _loop_from_state(state) if resume else candidate.config.loop
    if state.analysis is None:
        _step_analyzer(backends, candidate, state, run_dir, on_step)

    # с какого шага продолжаем прерванную итерацию (None = с её начала)
    entry = state.last_completed_step if resume else None
    entry = entry if entry in MID_ITERATION_STEPS else None
    while True:
        if entry is None:
            reason = check_stop(state, loop)
            if reason:
                return _finish(state, run_dir, backends, reason)
            if not _step_researcher(backends, candidate, state, run_dir, loop, on_step):
                return _finish(state, run_dir, backends, STOP_ASSUMPTIONS_EXHAUSTED)
        if entry in (None, STEP_RESEARCHER):
            _step_refiner(backends, candidate, state, run_dir, on_step)
        _step_judge(backends, candidate, state, run_dir, loop, on_step)
        entry = None


def _check_snapshot_shape(state: RunState) -> None:
    """Снапшот конфига должен быть в текущей форме (backends + roles).

    Прогоны до перехода на сменные бэкенды записали в state.config секцию
    models: тогда весь цикл шёл через один Anthropic API, и по такому снапшоту
    неизвестно, на каком транспорте продолжать. Молча продолжить — значит
    дописать в чужой прогон шаги, сделанные другими моделями, поэтому здесь
    явный отказ. Отчёт по такому прогону по-прежнему собирается: команда
    report снапшот не читает.
    """
    if LEGACY_MODELS_KEY in state.config and ROLES_KEY not in state.config:
        raise ValueError(
            f"прогон {state.run_id} запущен до перехода на сменные бэкенды "
            f"(в state.config есть '{LEGACY_MODELS_KEY}', нет '{ROLES_KEY}') — "
            f"продолжить его нельзя: неизвестно, какими бэкендами он шёл. "
            f"Отчёт по нему собирается: python -m kaidzen report <каталог>")


def _loop_from_state(state: RunState) -> LoopConfig:
    """Параметры цикла из снапшота конфига прогона.

    Отсутствие секции loop — это битый или чужой state.json: тихо подставить
    дефолты значило бы продолжить прогон с неизвестным пользователю бюджетом.
    """
    loop_data = state.config.get("loop")
    if loop_data is None:
        raise ValueError(
            "в state.config нет секции loop — продолжить прогон нельзя: "
            "неизвестно, с какими параметрами цикла он был запущен")
    return LoopConfig.model_validate(loop_data)


def _new_state(candidate: Candidate, idea_text: str, run_dir: Path,
               candidate_dir: Path | None = None) -> RunState:
    """candidate_dir попадает в снапшот конфига, чтобы resume (см.
    __main__._resume_candidate_dir) грузил промпты из того же каталога, с
    которым прогон был запущен, а не из candidates/<candidate_id> — при
    --candidate вне CANDIDATES_ROOT это может быть другой кандидат.
    """
    config = candidate.config.model_dump()
    if candidate_dir is not None:
        config["candidate_dir"] = str(candidate_dir)
    return RunState(run_id=run_dir.name, candidate_id=candidate.candidate_id,
                    config=config, original_idea=idea_text)


def _step_analyzer(backends, candidate: Candidate, state: RunState,
                   run_dir: Path,
                   on_step: Callable[[str, RunState], None] | None = None
                   ) -> None:
    """Раскладывает сырую идею и заполняет реестр допущений."""
    out = _with_retry(lambda: run_analyzer(backends[STEP_ANALYZER], candidate,
                                           idea_text=state.original_idea))
    state.analysis = Analysis(problem=out.problem, audience=out.audience,
                              mechanism=out.mechanism, unknowns=out.unknowns)
    state.assumptions = list(out.assumptions)
    _checkpoint(state, run_dir, backends, STEP_ANALYZER, on_step)


def _step_researcher(backends, candidate: Candidate, state: RunState, run_dir: Path,
                     loop: LoopConfig,
                     on_step: Callable[[str, RunState], None] | None = None
                     ) -> bool:
    """Проверяет самые рисковые допущения. False = проверять больше нечего."""
    selected = select_assumptions(state.assumptions,
                                  loop.assumptions_per_iteration)
    if not selected:
        return False
    out, unknown_ids = _with_retry(
        lambda: _research_and_apply(backends, candidate, state, selected))
    # находки нужны Refiner'у целиком; в реестр попадают только статус и факты,
    # поэтому сырой JSON кладём в состояние — иначе возобновление их потеряет
    state.pending_findings = out.model_dump_json()
    _checkpoint(state, run_dir, backends, STEP_RESEARCHER, on_step)
    if unknown_ids:
        _notify(on_step,
                f"{STEP_WARNING}: Researcher вернул находки по неизвестным "
                f"допущениям (отброшены): {', '.join(unknown_ids)}", state)
    return True


def _research_and_apply(backends, candidate: Candidate, state: RunState,
                        selected: list[Assumption]):
    """Один вызов Researcher вместе с разбором ответа — целиком под ретраем."""
    out = run_researcher(backends[STEP_RESEARCHER], candidate, idea_text=state.current_idea_text(),
                         assumptions=selected)
    return out, _apply_findings(state, out)


def _apply_findings(state: RunState, researcher_output) -> list[str]:
    """Переносит вердикты и факты в реестр. Возвращает id, которых там нет.

    Если в реестр не легла НИ ОДНА находка, вызов бесполезен целиком: допущения
    остаются unverified, следующая итерация исследует их заново, и так до конца
    бюджета — прогон стоит денег и не проверяет ничего. Поэтому здесь ValueError:
    шаг уйдёт на повтор, и модель получит промпт заново.
    """
    by_id = {a.id: a for a in state.assumptions}
    findings = researcher_output.findings
    unknown_ids = [f.assumption_id for f in findings if f.assumption_id not in by_id]
    if findings and len(unknown_ids) == len(findings):
        raise ValueError(
            "Researcher не вернул ни одной находки по существующим допущениям; "
            f"неизвестные id: {', '.join(unknown_ids)}")

    # реестр меняем только после проверки: неудачная попытка не должна
    # оставлять после себя половину применённых находок
    for finding in findings:
        assumption = by_id.get(finding.assumption_id)
        if assumption is None:
            continue
        assumption.status = finding.verdict
        assumption.facts = assumption.facts + list(finding.facts)
    return unknown_ids


def _step_refiner(backends, candidate: Candidate, state: RunState,
                  run_dir: Path,
                  on_step: Callable[[str, RunState], None] | None = None
                  ) -> None:
    """Переписывает идею по находкам и критике прошлого Judge.

    Критика берётся из state.last_critique, а не из текущей версии: после
    отката текущая версия — это позапрошлый текст с позапрошлой критикой, и
    Refiner повторил бы ровно то изменение, из-за которого случился откат.
    """
    out = _with_retry(lambda: _refine_and_check(backends, candidate, state))
    state.versions.append(Version(n=len(state.versions) + 1,
                                  idea_text=out.idea_text,
                                  changelog=out.changelog))
    _checkpoint(state, run_dir, backends, STEP_REFINER, on_step)


def _refine_and_check(backends, candidate: Candidate, state: RunState):
    """Один вызов Refiner вместе с проверкой обоснованности — под ретраем."""
    out = run_refiner(backends[STEP_REFINER], candidate, idea_text=state.current_idea_text(),
                      findings_json=state.pending_findings or "",
                      critique=state.last_critique)
    _check_grounding(state, out)
    return out


def _check_grounding(state: RunState, refiner_output) -> None:
    """Изменения идеи должны опираться на проверенные допущения.

    Вся система строится на том, что итерация закрывает допущения фактами, а
    не полирует текст. Пустой changelog при полученных находках и ссылка на
    несуществующее допущение — это ровно «переписал как захотелось»: такой
    ответ не принимаем, шаг уходит на повтор.
    """
    if state.pending_findings and not refiner_output.changelog:
        raise ValueError(
            "Refiner вернул пустой changelog, хотя получил находки Researcher: "
            "изменения идеи не обоснованы")
    known = {a.id for a in state.assumptions}
    unknown = [gid for entry in refiner_output.changelog
               for gid in entry.grounded_in if gid not in known]
    if unknown:
        raise ValueError(
            "changelog ссылается на несуществующие допущения: "
            f"{', '.join(unknown)}")


def _step_judge(backends, candidate: Candidate, state: RunState, run_dir: Path,
                loop: LoopConfig,
                on_step: Callable[[str, RunState], None] | None = None
                ) -> None:
    """Оценивает новую версию и применяет вердикт."""
    new_version = state.versions[-1]
    judge = _with_retry(lambda: run_judge(
        backends[STEP_JUDGE], candidate, new_idea=new_version.idea_text,
        previous_idea=_idea_before_last_version(state),
        assumptions=state.assumptions))
    apply_judge_verdict(state, judge, loop)
    state.iteration += 1
    state.pending_findings = None
    _checkpoint(state, run_dir, backends, STEP_JUDGE, on_step)


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
    """Три попытки с экспоненциальной паузой для ошибок из RETRYABLE_ERRORS.

    Остальное уходит наверх с первого раза: повтор детерминированной ошибки
    (обрезанный ответ, провал схемы, 400) не может закончиться иначе, зато
    стоит втрое. Состояние на диске при этом уже отражает последний
    завершённый шаг, так что прогон продолжается через resume.
    """
    delay = RETRY_BASE_DELAY_SECONDS
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return call()
        except RETRYABLE_ERRORS:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay *= RETRY_BACKOFF_FACTOR
    raise AssertionError("недостижимо")  # pragma: no cover


def _checkpoint(state: RunState, run_dir: Path, backends, step: str,
                on_step: Callable[[str, RunState], None] | None = None
                ) -> None:
    state.last_completed_step = step
    _save(state, run_dir, backends)
    _notify(on_step, step, state)


def _notify(on_step: Callable[[str, RunState], None] | None, step: str,
            state: RunState) -> None:
    if on_step is None:
        return
    try:
        on_step(step, state)
    except Exception:
        # прогресс в stdout — украшение, а не часть платного прогона:
        # упавший колбэк не должен убивать уже сделанную (и оплаченную) работу
        pass


def _finish(state: RunState, run_dir: Path, backends, reason: str) -> RunState:
    state.stop_reason = reason
    _save(state, run_dir, backends)
    return state


def _save(state: RunState, run_dir: Path, backends: dict) -> None:
    """Атомарная запись состояния вместе со свежим расходом токенов."""
    state.api_usage = total_usage(backends)
    save_state(state, run_dir)


def total_usage(backends: dict) -> ApiUsage:
    """Суммарный расход всех бэкендов прогона.

    Счётчик у каждого бэкенда свой, а в отчёте нужна общая цифра. Один и тот
    же экземпляр обслуживает несколько ролей (обычная схема: всё на подписке),
    поэтому считаем каждый экземпляр ровно один раз.
    """
    usages = [u for u in (getattr(b, "usage", None)
                          for b in _unique_backends(backends)) if u is not None]
    return ApiUsage(
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        web_searches=sum(u.web_searches for u in usages))


def _unique_backends(backends: dict) -> list:
    """Экземпляры без повторов, в порядке ролей: dict по объектам не построить."""
    seen: set[int] = set()
    unique = []
    for backend in backends.values():
        if id(backend) not in seen:
            seen.add(id(backend))
            unique.append(backend)
    return unique
