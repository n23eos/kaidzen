"""Оркестратор поколений мета-лупа.

Одно поколение: диагноз слабостей чемпиона → две мутации → eval челленджеров
на train-идеях настоящим Уровнем 1 → слепые попарки отчётов → Gate → возможная
смена чемпиона.

Три свойства держат всю конструкцию и ломаются незаметно:

1. **Слепота.** Судья получает только два текста отчётов, из которых вычищены
   имена прогона и кандидата, и каждая пара сравнивается дважды с перестановкой
   позиций. Совпали — победа, разошлись — ничья. Позиционная предвзятость судьи
   иначе становится самым дешёвым способом «выиграть», ничего не улучшив.
2. **Gate главнее судьи.** Победа в попарках без просадки объективных метрик —
   два независимых условия, и второе решает. Обходных путей у промоции нет.
3. **Прогоны дорогие по времени, а не по деньгам.** Один прогон Уровня 1 — 2–4
   минуты, поколение — около десятка прогонов. Отсюда пул на eval, кэш прогонов
   чемпиона и урезанный лимит итераций; отсюда же чекпоинт состояния после
   каждой стадии: прерывание в любой точке не должно стоить повторных прогонов.

`evolve` никогда не запускается сам (ТЗ §4.5): ни демона, ни крона, ни таймера.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_lowercase
from typing import Callable

from pydantic import BaseModel, Field

from kaidzen.backends.registry import build_backends
from kaidzen.benchmark import Benchmark
from kaidzen.candidate import (CHAMPION_PREFIX, Candidate, backends_by_role,
                               load_candidate)
from kaidzen.checkpoint import (CHECKPOINT_EVERY, STATUS_REJECTED as CHECKPOINT_REJECTED,
                                ChampionRef, CheckpointRecord, HoldoutResult,
                                make_checkpoint, pending_checkpoint)
from kaidzen.checkpoint import approve as approve_state
from kaidzen.checkpoint import reject as reject_state
from kaidzen.gate import decide
from kaidzen.metrics import RunMetrics, aggregate, run_metrics
from kaidzen.mutation import set_status, write_candidate
from kaidzen.orchestrator import run_pipeline
from kaidzen.report import build_report
from kaidzen.roles.meta import MetaConfig
from kaidzen.roles.meta.diagnostician import Diagnosis, run_diagnostician
from kaidzen.roles.meta.meta_judge import run_meta_judge
from kaidzen.roles.meta.mutator import run_mutator, to_patch
from kaidzen.state import RunState, load_state, write_atomically

MAX_GENERATIONS = 5
CHALLENGERS_PER_GENERATION = 2
# два поколения подряд без промоции = плато (ТЗ §4.4)
PLATEAU_GENERATIONS = 2
# Пул eval-прогонов. Двойка не от скромности: у подписки есть лимиты, и
# агрессивный пул упрётся в них — тогда поколение не ускорится, а развалится.
EVAL_CONCURRENCY = 2
# Урезанный лимит итераций на eval (ТЗ §4.6): поколение из десятка прогонов
# иначе идёт часами. Лимит только опускается, поднять его мутацией нельзя.
# Три вместо четырёх: на подписке платит квота пользователя, и цикл отладки
# самого мета-лупа не должен её выедать.
EVAL_MAX_ITERATIONS = 3
RUN_STAMP_LENGTH = 6      # метка прогона в имени потомка: хвост времени из evolve_id
# столько упавших прогонов — и кандидат считается нестабильным
FAILURE_LIMIT = 2
# меньше этого числа сравнимых идей — результат попарок ничего не значит,
# и об этом нужно сказать вслух: «побед 0 из 1» выглядит как полноценный итог
MIN_COMPARABLE_IDEAS = 2

STATE_FILE = "state.json"
SUMMARY_FILE = "summary.md"
RUNS_DIR = "runs"
REPORT_FILE = "report.md"
EVAL_SUMMARY_NOTE = ("_Прогон мета-лупа: executive summary не генерировался, чтобы не тратить\nвызов модели на каждый eval._")
# Мягкая остановка — отдельный файл, а не поле в state.json: evolve-stop пишется
# ЧУЖИМ процессом, пока оркестратор работает, и его правку state.json первое же
# сохранение молча затрёт. Файл переживает любые сохранения.
STOP_FILE = "STOP"

# чем заменяются имена прогона и кандидата в отчёте, который увидит судья
ANON = "—"

STAGE_STARTED = "started"
STAGE_DIAGNOSED = "diagnosed"
STAGE_MUTATED = "mutated"
STAGE_EVALUATED = "evaluated"
STAGE_COMPARED = "compared"
STAGE_GATED = "gated"

STATUS_PENDING = "pending"
STATUS_EVALUATED = "evaluated"
STATUS_UNSTABLE = "unstable"
STATUS_REJECTED = "rejected"
STATUS_PROMOTED = "promoted"

STOP_MAX_GENERATIONS = "max_generations"
STOP_PLATEAU = "plateau"
STOP_REQUESTED = "stop_requested"
STOP_CHECKPOINT_PENDING = "checkpoint_pending"
STOP_CHECKPOINT_REJECTED = "checkpoint_rejected"

CHAMPION_META_STATUS = "champion"


# --- состояние -------------------------------------------------------------

class EvalRun(BaseModel):
    """Один прогон Уровня 1: кандидат × идея."""

    candidate_id: str
    idea: str
    run_dir: str
    ok: bool = True
    error: str = ""
    metrics: RunMetrics | None = None


class CandidateRecord(BaseModel):
    candidate_id: str
    candidate_dir: str
    rationale: str = ""
    status: str = STATUS_PENDING
    runs: list[EvalRun] = Field(default_factory=list)
    metrics: RunMetrics | None = None
    win_rate: float | None = None
    gate_reason: str = ""


class GenerationRecord(BaseModel):
    number: int
    champion_id: str
    stage: str = STAGE_STARTED
    weaknesses: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    champion: CandidateRecord | None = None
    challengers: list[CandidateRecord] = Field(default_factory=list)
    # попытки мутатора, не доехавшие до диска (невалидный или слишком широкий
    # патч): поколение о них помнит, чтобы resume не повторял их вслепую
    discarded: list[str] = Field(default_factory=list)
    attempts_done: int = 0
    promoted_id: str = ""


class EvolveState(BaseModel):
    evolve_id: str
    domain: str
    champion_id: str
    champion_dir: str
    previous_champion_id: str = ""
    previous_champion_dir: str = ""
    generation: int = 0
    max_generations: int = MAX_GENERATIONS
    checkpoint_every: int = CHECKPOINT_EVERY
    generations: list[GenerationRecord] = Field(default_factory=list)
    checkpoints: list[CheckpointRecord] = Field(default_factory=list)
    # метрики кандидата на train — их сравнивает чекпоинт с holdout-метриками
    train_metrics: dict[str, RunMetrics] = Field(default_factory=dict)
    # (кандидат, идея) → готовый прогон: чемпион не гоняется дважды по одной
    # идее, а resume не переплачивает временем за уже сделанное
    cache: dict[str, EvalRun] = Field(default_factory=dict)
    stop_requested: bool = False
    stop_reason: str | None = None
    no_promotion_streak: int = 0


def save_evolve_state(state: EvolveState, evolve_dir: Path) -> None:
    write_atomically(evolve_dir / STATE_FILE, state.model_dump_json(indent=2))


def load_evolve_state(evolve_dir: Path) -> EvolveState:
    path = evolve_dir / STATE_FILE
    if not path.exists():
        raise FileNotFoundError(f"state.json не найден в {evolve_dir}")
    return EvolveState.model_validate_json(path.read_text(encoding="utf-8"))


# --- прогон Уровня 1 как зависимость ---------------------------------------

def default_pipeline(candidate: Candidate, *, candidate_dir: Path,
                     idea_text: str, run_dir: Path) -> RunState:
    """Настоящий прогон Уровня 1 со своими бэкендами."""
    backends = backends_by_role(candidate,
                               build_backends(candidate.config.model_dump()))
    return run_pipeline(backends, candidate, idea_text=idea_text,
                        run_dir=run_dir, candidate_dir=candidate_dir)


@dataclass(frozen=True)
class EvolveContext:
    """Внешние зависимости прогона: пути, бенчмарк, модели, транспорт.

    Собраны в один объект, чтобы стадии не таскали по восемь параметров, а
    тесты подменяли pipeline и мета-бэкенд без обезьяньих патчей.
    """

    evolve_dir: Path
    candidates_root: Path
    benchmark: Benchmark
    meta_backend: object
    meta: MetaConfig = field(default_factory=MetaConfig)
    pipeline: Callable[..., RunState] = default_pipeline
    concurrency: int = EVAL_CONCURRENCY
    on_event: Callable[[str], None] | None = None


# --- публичный вход --------------------------------------------------------

def run_evolve(ctx: EvolveContext, *, resume: bool = False,
               champion_dir: Path | None = None, evolve_id: str = "",
               max_generations: int = MAX_GENERATIONS,
               checkpoint_every: int = CHECKPOINT_EVERY) -> EvolveState:
    """Гоняет поколения до первого критерия остановки и возвращает состояние.

    Возврат — это всегда остановка с причиной в `state.stop_reason`, а не
    авария: незакрытый чекпоинт, исчерпанные поколения, плато, просьба
    остановиться. Продолжение — повторный вызов с resume=True.

    При resume лимиты (max_generations, checkpoint_every) берутся ИЗ СОСТОЯНИЯ,
    а не из аргументов: контракт тот же, что у run_pipeline (ТЗ §5) — прогон,
    начатый на двух поколениях, при продолжении не должен получить пять.
    """
    state = (load_evolve_state(ctx.evolve_dir) if resume
             else _new_state(ctx, champion_dir, evolve_id, max_generations,
                             checkpoint_every))
    _refuse_while_checkpoint_pending(state)
    state.stop_reason = None
    while True:
        reason = _check_stop(ctx, state)
        if reason:
            return _finish(ctx, state, reason)
        _run_generation(ctx, state)
        if _checkpoint_due(state):
            _open_checkpoint(ctx, state)
            return _finish(ctx, state, STOP_CHECKPOINT_PENDING)


def request_stop(evolve_dir: Path) -> None:
    """Мягкая остановка: поколение в работе дозавершается, новое не начнётся."""
    write_atomically(evolve_dir / STOP_FILE, "stop\n")


def approve_checkpoint(evolve_dir: Path) -> CheckpointRecord:
    state = load_evolve_state(evolve_dir)
    record = approve_state(state)
    save_evolve_state(state, evolve_dir)
    return record


def reject_checkpoint(evolve_dir: Path) -> CheckpointRecord:
    """Отклонение человека откатывает чемпиона и заканчивает evolve-прогон."""
    state = load_evolve_state(evolve_dir)
    record = reject_state(state)
    write_champion_pointer(state)      # откат должен дойти и до обычной работы
    state.stop_reason = STOP_CHECKPOINT_REJECTED
    save_evolve_state(state, evolve_dir)
    return record


def _new_state(ctx: EvolveContext, champion_dir: Path | None, evolve_id: str,
               max_generations: int, checkpoint_every: int) -> EvolveState:
    if champion_dir is None:
        raise ValueError("для нового evolve-прогона нужен champion_dir")
    return EvolveState(
        evolve_id=evolve_id or ctx.evolve_dir.name, domain=ctx.benchmark.domain,
        champion_id=champion_dir.name, champion_dir=str(champion_dir),
        max_generations=max_generations, checkpoint_every=checkpoint_every)


def _finish(ctx: EvolveContext, state: EvolveState, reason: str) -> EvolveState:
    state.stop_reason = reason
    _save(ctx, state)
    _event(ctx, f"остановка: {reason}")
    return state


def _save(ctx: EvolveContext, state: EvolveState) -> None:
    save_evolve_state(state, ctx.evolve_dir)
    write_atomically(ctx.evolve_dir / SUMMARY_FILE, render_summary(state))


def _event(ctx: EvolveContext, message: str) -> None:
    if ctx.on_event is None:
        return
    try:
        ctx.on_event(message)
    except Exception:
        # прогресс в stdout — украшение: упавший колбэк не должен ронять
        # поколение, за которое уже заплачено стенными часами
        pass


# --- критерии остановки ----------------------------------------------------

def _check_stop(ctx: EvolveContext, state: EvolveState) -> str | None:
    """Критерии остановки проверяются только на границе поколений.

    Недоделанное поколение сначала доводится до конца: его eval-прогоны уже
    оплачены стенными часами, а лимит поколений успел учесть начатое.
    """
    if _generation_in_flight(state):
        return None
    if _last_checkpoint_rejected(state):
        return STOP_CHECKPOINT_REJECTED
    if _stop_requested(ctx, state):
        return STOP_REQUESTED
    if state.generation >= state.max_generations:
        return STOP_MAX_GENERATIONS
    if state.no_promotion_streak >= PLATEAU_GENERATIONS:
        return STOP_PLATEAU
    return None


def _generation_in_flight(state: EvolveState) -> bool:
    return bool(state.generations) and state.generations[-1].stage != STAGE_GATED


def _stop_requested(ctx: EvolveContext, state: EvolveState) -> bool:
    """Флаг проверяется только на границе поколений (ТЗ §4.5).

    Бросать поколение на полпути нельзя: eval-прогоны уже оплачены временем,
    и выброшенные результаты пришлось бы получать заново.
    """
    if (ctx.evolve_dir / STOP_FILE).exists():
        state.stop_requested = True
    return state.stop_requested


def _last_checkpoint_rejected(state: EvolveState) -> bool:
    return (bool(state.checkpoints)
            and state.checkpoints[-1].status == CHECKPOINT_REJECTED)


def _refuse_while_checkpoint_pending(state: EvolveState) -> None:
    record = pending_checkpoint(state)
    if record is None:
        return
    raise ValueError(
        f"поколение {record.generation} ждёт решения человека: {record.path}. "
        f"Продолжить можно после checkpoint --approve или --reject")


# --- поколение -------------------------------------------------------------

def _run_generation(ctx: EvolveContext, state: EvolveState) -> None:
    """Стадии по порядку; каждая начинается с того места, где оборвалась."""
    gen = _current_generation(ctx, state)
    _event(ctx, f"поколение {gen.number}: чемпион {gen.champion_id}")
    if gen.stage == STAGE_STARTED:
        _diagnose(ctx, state, gen)
    if gen.stage == STAGE_DIAGNOSED:
        _mutate(ctx, state, gen)
    if gen.stage == STAGE_MUTATED:
        _evaluate(ctx, state, gen)
    if gen.stage == STAGE_EVALUATED:
        _compare(ctx, state, gen)
    if gen.stage == STAGE_COMPARED:
        _gate(ctx, state, gen)


def _current_generation(ctx: EvolveContext,
                        state: EvolveState) -> GenerationRecord:
    """Незавершённое поколение (resume) либо новое."""
    if state.generations and state.generations[-1].stage != STAGE_GATED:
        return state.generations[-1]
    state.generation += 1
    gen = GenerationRecord(number=state.generation, champion_id=state.champion_id)
    state.generations.append(gen)
    _save(ctx, state)
    return gen


def _advance(ctx: EvolveContext, state: EvolveState, gen: GenerationRecord,
             stage: str) -> None:
    gen.stage = stage
    _save(ctx, state)


def _diagnose(ctx: EvolveContext, state: EvolveState,
              gen: GenerationRecord) -> None:
    """Чемпион на train-идеях (обычно из кэша) → метрики и отчёты → диагноз."""
    record = gen.champion or CandidateRecord(candidate_id=state.champion_id,
                                             candidate_dir=state.champion_dir)
    gen.champion = record
    # чемпион идёт по всем идеям без досрочной отбраковки: он база сравнения
    _evaluate_candidate(ctx, state, record, ctx.benchmark.train,
                        stop_on_failures=False)
    reports = _reports_of(record)
    if not reports:
        raise ValueError(
            f"чемпион {record.candidate_id} не дал ни одного успешного прогона "
            f"на train — эволюционировать не от чего")
    diagnosis = run_diagnostician(ctx.meta_backend, ctx.meta,
                                  metrics=record.metrics, reports=reports)
    gen.weaknesses = list(diagnosis.weaknesses)
    gen.hypotheses = list(diagnosis.hypotheses)
    _event(ctx, f"диагноз: {len(gen.hypotheses)} гипотез")
    _advance(ctx, state, gen, STAGE_DIAGNOSED)


def _mutate(ctx: EvolveContext, state: EvolveState,
            gen: GenerationRecord) -> None:
    """Две попытки мутатора. Непринятая попытка пропускается, а не чинится."""
    champion = load_candidate(Path(state.champion_dir))
    diagnosis = _diagnosis_of(gen)
    for attempt in range(CHALLENGERS_PER_GENERATION):
        if attempt < gen.attempts_done:
            continue        # эту попытку уже сделали до прерывания
        proposal = run_mutator(ctx.meta_backend, ctx.meta, champion,
                               diagnosis=diagnosis, attempt=attempt)
        candidate_id = _candidate_id(gen.number, attempt, state.evolve_id)
        try:
            target = write_candidate(parent_dir=Path(state.champion_dir),
                                     root=ctx.candidates_root,
                                     candidate_id=candidate_id,
                                     patch=to_patch(proposal))
        except (ValueError, FileNotFoundError) as e:
            gen.discarded.append(f"{candidate_id}: {e}")
            _event(ctx, f"мутация {candidate_id} отброшена: {e}")
        else:
            gen.challengers.append(CandidateRecord(
                candidate_id=candidate_id, candidate_dir=str(target),
                rationale=proposal.rationale))
            _event(ctx, f"челленджер {candidate_id} записан")
        gen.attempts_done = attempt + 1
        _save(ctx, state)
    _advance(ctx, state, gen, STAGE_MUTATED)


def _diagnosis_of(gen: GenerationRecord) -> Diagnosis:
    """Диагноз восстанавливается из записи поколения, а не хранится отдельно:
    после прерывания между стадиями мутатор обязан получить тот же вход."""
    return Diagnosis(weaknesses=gen.weaknesses, hypotheses=gen.hypotheses)


def _evaluate(ctx: EvolveContext, state: EvolveState,
              gen: GenerationRecord) -> None:
    for record in gen.challengers:
        if record.status == STATUS_PENDING:
            _evaluate_candidate(ctx, state, record, ctx.benchmark.train)
    _advance(ctx, state, gen, STAGE_EVALUATED)


def _compare(ctx: EvolveContext, state: EvolveState,
             gen: GenerationRecord) -> None:
    """Слепые попарки отчётов чемпиона и каждого челленджера по каждой идее."""
    champion_reports = {run.idea: _blind_report(run)
                        for run in gen.champion.runs if run.ok}
    for record in gen.challengers:
        if record.status != STATUS_EVALUATED or record.win_rate is not None:
            continue
        pairs = [(champion_reports[run.idea], _blind_report(run))
                 for run in record.runs if run.ok and run.idea in champion_reports]
        wins = sum(1 for champion, challenger in pairs
                   if _challenger_wins(ctx, champion, challenger))
        record.win_rate = wins / len(pairs) if pairs else 0.0
        total = len(ctx.benchmark.train)
        note = (f"{record.candidate_id}: побед {wins} из {len(pairs)} "
                f"(сравнимых идей {len(pairs)} из {total})")
        if len(pairs) < MIN_COMPARABLE_IDEAS:
            note += " — выборка слишком мала, результату верить нельзя"
        _event(ctx, note)
        _save(ctx, state)
    _advance(ctx, state, gen, STAGE_COMPARED)


def _challenger_wins(ctx: EvolveContext, champion_report: str,
                     challenger_report: str) -> bool:
    """Две прогонки с перестановкой позиций; расхождение — ничья, не победа.

    Судья предвзят к позиции, и без перестановки эта предвзятость — самый
    дешёвый способ для челленджера «выиграть», не став лучше.
    """
    first = run_meta_judge(ctx.meta_backend, ctx.meta,
                           report_a=champion_report, report_b=challenger_report)
    second = run_meta_judge(ctx.meta_backend, ctx.meta,
                            report_a=challenger_report, report_b=champion_report)
    return first.winner == "B" and second.winner == "A"


def _candidate_id(generation: int, attempt: int, evolve_id: str) -> str:
    """Имя потомка, уникальное между прогонами и стабильное внутри одного.

    Без метки прогона второй запуск evolve упирается в кандидатов, созданных
    первым, и отбрасывает обе мутации — поколение проходит впустую, потратив
    прогоны чемпиона. Поймано на первом полном прогоне бенчмарка.

    Метка берётся из evolve_id детерминированно, иначе resume не попал бы в то
    же имя и создал бы дубль потомка.
    """
    digits = "".join(c for c in evolve_id if c.isdigit())
    stamp = digits[-RUN_STAMP_LENGTH:] if digits else "x"
    return f"gen{generation:03d}-{ascii_lowercase[attempt]}-{stamp}"


def _gate(ctx: EvolveContext, state: EvolveState,
          gen: GenerationRecord) -> None:
    """Решение Gate по каждому челленджеру и смена чемпиона, если есть кому."""
    winner: CandidateRecord | None = None
    for record in gen.challengers:
        if record.status != STATUS_EVALUATED:
            continue
        decision = decide(champion=gen.champion.metrics,
                          challenger=record.metrics, win_rate=record.win_rate)
        record.gate_reason = decision.reason
        record.status = STATUS_PROMOTED if decision.promote else STATUS_REJECTED
        _event(ctx, f"{record.candidate_id}: {decision.reason}")
        if decision.promote and (winner is None
                                 or record.win_rate > winner.win_rate):
            winner = record
    _apply_gate(ctx, state, gen, winner)
    _advance(ctx, state, gen, STAGE_GATED)


def _apply_gate(ctx: EvolveContext, state: EvolveState, gen: GenerationRecord,
                winner: CandidateRecord | None) -> None:
    """Смена чемпиона — единственный способ, которым мутация доходит до работы."""
    for record in gen.challengers:
        if record is not winner and record.status == STATUS_PROMOTED:
            # прошёл Gate, но поколение продвигает одного: иначе непонятно,
            # чей вклад измеряет следующий диагноз
            record.status = STATUS_REJECTED
            record.gate_reason += "; другой челленджер выиграл больше попарок"
        _write_meta_status(record)
    if winner is None:
        state.no_promotion_streak += 1
        return
    state.previous_champion_id = state.champion_id
    state.previous_champion_dir = state.champion_dir
    state.champion_id = winner.candidate_id
    state.champion_dir = winner.candidate_dir
    gen.promoted_id = winner.candidate_id
    state.no_promotion_streak = 0
    _write_meta_status(winner, CHAMPION_META_STATUS)
    write_champion_pointer(state)


def write_champion_pointer(state: EvolveState) -> None:
    """Указатель candidates/CHAMPION-<domain> на нового чемпиона.

    Смена этого файла — единственный способ, которым результат эволюции
    попадает в обычную работу (ТЗ §4.5): `run` читает именно его.
    """
    root = Path(state.champion_dir).parent
    write_atomically(root / f"{CHAMPION_PREFIX}{state.domain}",
                     f"{state.champion_id}\n")


def _write_meta_status(record: CandidateRecord, status: str = "") -> None:
    """Статус и метрики в meta.json кандидата: линия предков читается с диска."""
    set_status(Path(record.candidate_dir), status or record.status,
               eval_data={"win_rate": record.win_rate,
                          "gate_reason": record.gate_reason,
                          "metrics": record.metrics.model_dump()
                          if record.metrics else None})


# --- eval ------------------------------------------------------------------

def _cache_key(candidate_id: str, idea: Path) -> str:
    return f"{candidate_id}|{idea}"


def _evaluate_candidate(ctx: EvolveContext, state: EvolveState,
                        record: CandidateRecord, ideas: list[Path], *,
                        is_train: bool = True,
                        stop_on_failures: bool = True) -> None:
    """Прогоняет кандидата по идеям: готовое берёт из кэша, остальное — пулом.

    stop_on_failures=False — для чемпиона: он база сравнения, а не кандидат на
    оценку. Оборвав его прогоны, мы сужаем пересечение с челленджерами, и
    сравнивать становится не на чем. Поймано на полном бенчмарке: два падения
    чемпиона схлопнули пять идей до одного сравнения.
    """
    done = {run.idea for run in record.runs}
    pending = []
    for idea in ideas:
        if str(idea) in done:
            continue
        cached = state.cache.get(_cache_key(record.candidate_id, idea))
        if cached is not None:
            record.runs.append(cached)
        else:
            pending.append(idea)
    _save(ctx, state)
    if pending:
        _run_pool(ctx, state, record, pending,
                  stop_on_failures=stop_on_failures)
    _finish_candidate(ctx, state, record, is_train=is_train,
                      mark_unstable=stop_on_failures)


def _run_pool(ctx: EvolveContext, state: EvolveState, record: CandidateRecord,
              ideas: list[Path], *, stop_on_failures: bool = True) -> None:
    """Пул прогонов. Состояние трогает только главный поток — по мере готовности.

    Рабочие потоки возвращают готовый EvalRun и ничего не сохраняют: иначе
    атомарная запись state.json из двух потоков перетирала бы саму себя.
    """
    with ThreadPoolExecutor(max_workers=max(1, ctx.concurrency)) as pool:
        futures: list[Future] = [pool.submit(_one_eval, ctx, record, idea)
                                 for idea in ideas]
        for future in as_completed(futures):
            run = future.result()
            record.runs.append(run)
            state.cache[_cache_key(record.candidate_id, Path(run.idea))] = run
            _event(ctx, f"{record.candidate_id} / {Path(run.idea).name}: "
                        f"{'готово' if run.ok else 'ошибка ' + run.error}")
            _save(ctx, state)
            if stop_on_failures and _failures(record) >= FAILURE_LIMIT:
                for rest in futures:
                    rest.cancel()   # уже начатые дойдут сами, новые не стартуют
                break


def _one_eval(ctx: EvolveContext, record: CandidateRecord,
              idea: Path) -> EvalRun:
    """Один прогон Уровня 1 в рабочем потоке. Падение — это данные, не авария."""
    run_dir = ctx.evolve_dir / RUNS_DIR / record.candidate_id / idea.stem
    base = EvalRun(candidate_id=record.candidate_id, idea=str(idea),
                   run_dir=str(run_dir))
    try:
        candidate = _eval_candidate(Path(record.candidate_dir))
        state = ctx.pipeline(candidate, candidate_dir=Path(record.candidate_dir),
                             idea_text=idea.read_text(encoding="utf-8"),
                             run_dir=run_dir)
    except Exception as e:
        return base.model_copy(update={"ok": False, "error": str(e)})
    _write_eval_report(state, run_dir)
    return base.model_copy(update={"metrics": run_metrics(state)})


def _write_eval_report(state: RunState, run_dir: Path) -> None:
    """Читаемый отчёт рядом с состоянием.

    Сборка отчёта не требует модели, поэтому делать её здесь ничего не стоит,
    а без неё результат поколения приходится выковыривать из state.json руками.
    Executive summary не генерируем: это единственная часть, требующая вызова
    модели, и ради неё гонять её на каждый eval-прогон не стоит.
    """
    report = build_report(state, summary_text=EVAL_SUMMARY_NOTE)
    (run_dir / REPORT_FILE).write_text(report, encoding="utf-8")


def _eval_candidate(candidate_dir: Path) -> Candidate:
    """Кандидат с урезанным лимитом итераций: eval не должен идти часами."""
    candidate = load_candidate(candidate_dir)
    limit = min(candidate.config.loop.max_iterations, EVAL_MAX_ITERATIONS)
    loop = candidate.config.loop.model_copy(update={"max_iterations": limit})
    config = candidate.config.model_copy(update={"loop": loop})
    return candidate.model_copy(update={"config": config})


def _failures(record: CandidateRecord) -> int:
    return sum(1 for run in record.runs if not run.ok)


def _finish_candidate(ctx: EvolveContext, state: EvolveState,
                      record: CandidateRecord, *, is_train: bool,
                      mark_unstable: bool = True) -> None:
    """Итог кандидата: агрегированные метрики и статус.

    Два упавших прогона — это нестабильный кандидат, а не досадная случайность:
    дальше по поколению он не идёт, чтобы не сравнивать чемпиона с обломками.
    """
    record.metrics = aggregate([run.metrics for run in record.runs
                                if run.ok and run.metrics is not None])
    unstable = mark_unstable and _failures(record) >= FAILURE_LIMIT
    record.status = STATUS_UNSTABLE if unstable else STATUS_EVALUATED
    if is_train:
        # holdout-метрики сюда попасть не должны: на чекпоинте именно их
        # сравнивают с train-метриками, и совпадение обессмыслило бы проверку
        state.train_metrics[record.candidate_id] = record.metrics
    _save(ctx, state)


def _reports_of(record: CandidateRecord) -> list[str]:
    return [_blind_report(run) for run in record.runs if run.ok]


def _blind_report(run: EvalRun) -> str:
    """Отчёт без имён прогона и кандидата.

    Судья сравнивает два таких текста, и «gen003-a» в шапке одного из них
    сразу подсказал бы ему, кто здесь новичок, а кто действующий чемпион.
    """
    state = load_state(Path(run.run_dir))
    return build_report(state.model_copy(update={"run_id": ANON,
                                                 "candidate_id": ANON}))


# --- чекпоинты -------------------------------------------------------------

def _checkpoint_due(state: EvolveState) -> bool:
    """Ноль отключает чекпоинты совсем: иначе их не отключить."""
    if state.checkpoint_every <= 0:
        return False
    return state.generation % state.checkpoint_every == 0


def _open_checkpoint(ctx: EvolveContext, state: EvolveState) -> None:
    """Пауза с прогоном обоих чемпионов на holdout и сводкой для человека."""
    champion = ChampionRef(candidate_id=state.champion_id,
                           candidate_dir=state.champion_dir)
    previous = (ChampionRef(candidate_id=state.previous_champion_id,
                            candidate_dir=state.previous_champion_dir)
                if state.previous_champion_id else None)
    record = make_checkpoint(
        evolve_dir=ctx.evolve_dir, generation=state.generation,
        champion=champion, previous=previous,
        evaluate=lambda ref: _holdout(ctx, state, ref),
        train_metrics=state.train_metrics)
    state.checkpoints.append(record)
    _event(ctx, f"чекпоинт поколения {record.generation}: {record.path}")


def _holdout(ctx: EvolveContext, state: EvolveState,
             ref: ChampionRef) -> HoldoutResult:
    """Прогон кандидата на holdout-идеях — тех, что не видели ни диагност,
    ни мутатор. Кэш общий с train: одну и ту же пару дважды не гоняем."""
    record = CandidateRecord(candidate_id=ref.candidate_id,
                             candidate_dir=ref.candidate_dir)
    _evaluate_candidate(ctx, state, record, ctx.benchmark.holdout,
                        is_train=False)
    return HoldoutResult(metrics=record.metrics, reports=_reports_of(record))


# --- сводка для человека ---------------------------------------------------

def render_summary(state: EvolveState) -> str:
    """Линия эволюции: что предлагалось, что прошло Gate и почему."""
    lines = [f"# Эволюция {state.evolve_id} (домен {state.domain})", "",
             f"- Текущий чемпион: {state.champion_id}",
             f"- Поколений пройдено: {state.generation}",
             f"- Остановка: {state.stop_reason or ANON}", ""]
    for gen in state.generations:
        lines.append(f"## Поколение {gen.number} (стадия {gen.stage})")
        lines.append("")
        lines.append(f"Чемпион: {gen.champion_id}")
        lines.append("")
        for record in gen.challengers:
            win = ANON if record.win_rate is None else f"{record.win_rate:.0%}"
            lines.append(f"- **{record.candidate_id}** — {record.status}, "
                         f"побед в попарках {win}. {record.gate_reason}")
            lines.append(f"  - замысел: {record.rationale}")
        for note in gen.discarded:
            lines.append(f"- отброшено: {note}")
        lines.append("")
    return "\n".join(lines) + "\n"
