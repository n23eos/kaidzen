"""CLI Kaidzen: run / resume / report + evolve / evolve-resume / evolve-stop /
checkpoint.

Точка входа для человека за терминалом: прогон стоит денег и идёт минутами,
поэтому команды печатают ход дела и валятся с понятным текстом, а не трейсбеком.

Мета-луп (Уровень 2) добавляет к этому вторую шкалу времени: поколение — это
около десятка прогонов Уровня 1, то есть десятки минут. Поэтому у evolve те же
два свойства, что и у run: живой прогресс в stdout и понятная финальная сводка.
Сам он никогда не стартует: ни демона, ни крона — только эти команды (ТЗ §4.5).
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from kaidzen.backends.base import BackendError
from kaidzen.backends.registry import build_backends
from kaidzen.benchmark import Benchmark, BenchmarkEmpty, load_benchmark
from kaidzen.candidate import (CHAMPION_PREFIX, REPORTER_ROLE, Candidate,
                               LoopConfig, backends_by_role, load_candidate)
from kaidzen.checkpoint import NO_DATA, pending_checkpoint
from kaidzen.evolve import (EVAL_CONCURRENCY, MAX_GENERATIONS,
                            STOP_CHECKPOINT_REJECTED, SUMMARY_FILE,
                            EvolveContext, EvolveState, approve_checkpoint,
                            load_evolve_state, reject_checkpoint, request_stop,
                            run_evolve)
from kaidzen.orchestrator import (STEP_ANALYZER, STEP_JUDGE, STEP_REFINER,
                                  STEP_RESEARCHER, STEP_WARNING,
                                  _loop_from_state, run_pipeline, total_usage)
from kaidzen.report import build_report
from kaidzen.roles.meta import MetaConfig, build_meta_backend
from kaidzen.state import RunState, load_state, save_state

# пересказ готового текста без домысливания
SUMMARY_EFFORT = "low"
SUMMARY_SYSTEM = (
    "Ты пишешь executive summary по финальной версии идеи. "
    "Дай 3–5 предложений: что за идея, для кого, как работает и что "
    "проверено. Пиши по-русски, без воды и без маркетинговых прилагательных."
)

RUN_DIR_TIMESTAMP_FORMAT = "%Y-%m-%d-%H%M%S"
CANDIDATES_ROOT = Path("candidates")
RUNS_ROOT = Path("runs")
BENCHMARK_ROOT = Path("benchmark")
EVOLVE_ROOT = Path("evolve")
# префикс строк прогресса мета-лупа: сообщения о поколении не должны путаться
# со строками ролей Уровня 1, которые печатает вложенный прогон
EVOLVE_PREFIX = "evolve"
REPORT_FILENAME = "report.md"
DOMAINS = ("generic", "business", "games")
DEFAULT_DOMAIN = "generic"
FALLBACK_SLUG = "idea"
REBUILT_SUMMARY = ("_Отчёт пересобран из состояния прогона без обращения к API: "
                   "executive summary не генерировался._")


class SummaryOutput(BaseModel):
    summary: str


# --- чистые хелперы --------------------------------------------------------

def resolve_candidate_dir(*, candidates_root: Path, domain: str) -> Path:
    """Каталог кандидата-чемпиона для домена по файлу-указателю CHAMPION-<domain>."""
    pointer = candidates_root / f"{CHAMPION_PREFIX}{domain}"
    if not pointer.exists():
        raise FileNotFoundError(f"нет файла-указателя чемпиона: {pointer}")
    name = pointer.read_text(encoding="utf-8").strip()
    if not name:
        raise ValueError(f"файл-указатель пуст: {pointer}")
    # указатель пишет write_champion_pointer, и там всегда имя папки; путь в
    # нём означает правку руками или порчу файла, и уводит прогон за пределы
    # каталога кандидатов
    if name != Path(name).name:
        raise ValueError(
            f"в {pointer} должно быть имя каталога кандидата, а не путь: {name!r}")
    candidate_dir = candidates_root / name
    if not candidate_dir.is_dir():
        raise FileNotFoundError(
            f"чемпион '{name}' из {pointer} не найден: {candidate_dir}")
    return candidate_dir


def make_run_dir(*, runs_root: Path, idea_path: Path, now_str: str) -> Path:
    """Путь каталога прогона: <runs_root>/<время>-<слаг имени файла идеи>.

    Каталог не создаётся — функция чистая. Буквы кириллицы сохраняются:
    пользователь пишет идеи по-русски и должен узнавать свой прогон в списке.
    """
    return runs_root / f"{now_str}-{slugify(idea_path.stem)}"


def make_evolve_dir(*, evolve_root: Path, domain: str, now_str: str) -> Path:
    """Путь каталога evolve-прогона: <evolve_root>/<время>-<домен>.

    Каталог не создаётся — функция чистая, как и make_run_dir. Домен в имени,
    а не слаг идеи: evolve-прогон гоняет весь бенчмарк домена, а не одну идею.
    """
    return evolve_root / f"{now_str}-{domain}"


def positive_int(value: str) -> int:
    """Тип argparse для лимитов, у которых ноль означает опечатку.

    Без проверки `--generations 0` тихо завершал команду, ничего не сделав, а
    `--concurrency 0` так же тихо подменялся единицей внутри пула: пользователь
    видел не отказ, а странно отработавший прогон.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"нужно целое число, получено {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"нужно число больше нуля, получено {number}")
    return number


def slugify(text: str) -> str:
    """Нижний регистр, любые не-буквенно-цифровые серии — в один дефис."""
    slug = re.sub(r"[\W_]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    return slug or FALLBACK_SLUG


def _resume_candidate_dir(saved: RunState) -> Path:
    """Каталог кандидата, с которым прогон был запущен.

    Свежие прогоны кладут абсолютный путь в state.config["candidate_dir"]
    (см. orchestrator._new_state) — resume обязан грузить промпты именно
    оттуда: --candidate мог указывать на каталог вне CANDIDATES_ROOT, и
    candidates/<candidate_id> при этом либо не существует, либо содержит
    совсем другого кандидата с тем же именем. Старые state.json без этого
    ключа возвращаются к прежнему поведению.
    """
    recorded = saved.config.get("candidate_dir")
    if recorded is None:
        return CANDIDATES_ROOT / saved.candidate_id
    path = Path(recorded)
    if not path.is_dir():
        raise FileNotFoundError(
            f"каталог кандидата, с которым был запущен прогон {saved.run_id}, "
            f"не найден: {path}")
    return path


def apply_max_iter(candidate: Candidate, max_iter: int | None) -> Candidate:
    """Новый кандидат с изменённым лимитом итераций (исходный не меняем).

    Лимит собирается через валидацию, а не через model_copy: model_copy границы
    поля не проверяет, и --max-iter 0 или --max-iter 999 молча прошёл бы мимо
    LoopConfig. Прогон при этом стартует, снапшот с чужим числом уезжает в
    state.json, а resume падает на _loop_from_state — то есть оплаченный прогон
    оказывается непродолжаемым. Отказ на старте дешевле.
    """
    if max_iter is None:
        return candidate
    loop = LoopConfig.model_validate(
        {**candidate.config.loop.model_dump(), "max_iterations": max_iter})
    config = candidate.config.model_copy(update={"loop": loop})
    return candidate.model_copy(update={"config": config})


# --- вывод хода прогона ----------------------------------------------------

def _print_start(run_dir: Path, candidate: Candidate,
                 max_iterations: int | None = None) -> None:
    """max_iterations — лимит, действующий на этот прогон.

    По умолчанию берётся из candidate.config.loop (свежий run). При resume
    вызывающий код обязан передать лимит из снапшота state.config
    (orchestrator._loop_from_state) — именно он, а не конфиг кандидата,
    управляет остановкой цикла в run_pipeline (ТЗ §5).
    """
    limit = (max_iterations if max_iterations is not None
             else candidate.config.loop.max_iterations)
    print(f"Прогон: {run_dir.name}")
    print(f"Кандидат: {candidate.candidate_id} "
          f"(домен {candidate.config.domain}, "
          f"до {limit} итераций)")
    _print_roles(candidate)


def _print_roles(candidate: Candidate) -> None:
    """Кто на каком бэкенде: пользователь должен видеть, что идёт по подписке,
    а не жжёт платный ключ. Ключи не печатаются — только имя бэкенда и модель.
    """
    for role, cfg in candidate.config.roles.items():
        print(f"  {role}: {cfg.backend} / {cfg.model}")


class ProgressPrinter:
    """Печатает ход прогона в stdout по мере готовности каждого шага.

    Передаётся в run_pipeline как on_step, поэтому пользователь видит номер
    итерации, проверяемые допущения, вердикты и оценки Judge не постфактум,
    а в реальном времени — иначе платный многоминутный прогон неотличим
    от зависшего терминала.

    Для шага Researcher колбэк получает всё состояние целиком, а не только
    свежие находки, поэтому принтер сам хранит статусы допущений с прошлого
    вызова и печатает только то, что изменилось именно сейчас.
    """

    def __init__(self) -> None:
        self._last_statuses: dict[str, str] = {}

    def __call__(self, step: str, state: RunState) -> None:
        if step == STEP_ANALYZER:
            self._print_analyzer(state)
        elif step == STEP_RESEARCHER:
            self._print_researcher(state)
        elif step == STEP_REFINER:
            self._print_refiner(state)
        elif step == STEP_JUDGE:
            self._print_judge(state)
        elif step.startswith(STEP_WARNING):
            self._print_warning(step)

    def _print_analyzer(self, state: RunState) -> None:
        high = sum(1 for a in state.assumptions if a.criticality == "high")
        print(f"[analyzer] допущений найдено: {len(state.assumptions)}, "
              f"из них критичных: {high}")
        self._remember(state)

    def _print_researcher(self, state: RunState) -> None:
        checked = [a for a in state.assumptions
                  if self._last_statuses.get(a.id) != a.status]
        verdicts = ", ".join(f"{a.id}={a.status}" for a in checked)
        print(f"[researcher] проверены допущения: {verdicts or 'нет'}")
        self._remember(state)

    def _print_refiner(self, state: RunState) -> None:
        version = state.versions[-1]
        print(f"[refiner] готова новая версия идеи: v{version.n}")

    def _print_judge(self, state: RunState) -> None:
        version = state.versions[-1]
        if version.rolled_back:
            print(f"[judge] итерация {state.iteration}: версия откачена")
            return
        judge = version.judge
        print(f"[judge] итерация {state.iteration}: оценка {judge.total:.1f} "
              f"(дельта {judge.delta_vs_previous:+.1f})")

    def _print_warning(self, step: str) -> None:
        # step имеет вид "warning: <текст>" (см. orchestrator._notify) —
        # печатаем сам текст, не задваивая префикс
        message = step[len(STEP_WARNING) + 1:].strip()
        print(f"[{STEP_WARNING}] {message}")

    def _remember(self, state: RunState) -> None:
        self._last_statuses = {a.id: a.status for a in state.assumptions}


def _print_finish(state: RunState, report_path: Path) -> None:
    usage = state.api_usage
    print(f"Остановка: {state.stop_reason}")
    print(f"Итераций: {state.iteration}")
    print(f"Токены: {usage.input_tokens} входных, {usage.output_tokens} выходных; "
          f"веб-поисков: {usage.web_searches}")
    print(f"Отчёт: {report_path}")


# --- шаги команд -----------------------------------------------------------

def _build_role_backends(candidate: Candidate) -> dict:
    """Бэкенды ролей — один раз на старте, до первого платного вызова.

    Ключи требуются только те, которые объявили бэкенды этого кандидата:
    прогон целиком на подписке не требует ни одной переменной окружения.
    Отсутствующий ключ — понятный выход, а не трейсбек на пятой минуте.
    """
    try:
        built = build_backends(candidate.config.model_dump())
    except BackendError as e:
        sys.exit(f"Не удалось подготовить бэкенды кандидата "
                 f"{candidate.candidate_id}: {e}")
    return backends_by_role(candidate, built)


def _read_idea(idea_path: Path) -> str:
    if not idea_path.is_file():
        sys.exit(f"Файл идеи не найден: {idea_path}")
    return idea_path.read_text(encoding="utf-8")


def _generate_summary(backends: dict, candidate: Candidate,
                      state: RunState) -> str:
    """Единственный вызов LLM вне цикла: короткое резюме финальной идеи."""
    out = backends[REPORTER_ROLE].structured(
        model=_reporter_model(candidate), system=SUMMARY_SYSTEM,
        user=state.current_idea_text(), schema=SummaryOutput,
        effort=SUMMARY_EFFORT)
    state.api_usage = total_usage(backends)
    return out.summary


def _write_report(state: RunState, run_dir: Path, summary_text: str) -> Path:
    report_path = run_dir / REPORT_FILENAME
    report_path.write_text(build_report(state, summary_text=summary_text),
                           encoding="utf-8")
    return report_path


def _reporter_model(candidate: Candidate) -> str:
    """Модель для executive summary.

    Наличие роли reporter гарантирует валидация кандидата (candidate.py),
    поэтому здесь проверки нет: она бы дублировала загрузчик конфига.
    """
    return candidate.config.roles[REPORTER_ROLE].model


def _finish_run(backends: dict, candidate: Candidate, state: RunState,
                run_dir: Path) -> None:
    """Общий хвост run и resume: резюме, отчёт, итоговая сводка.

    Ход итераций пользователь уже видел вживую через ProgressPrinter — здесь
    печатается только финальная сводка, без повторного прохода по versions.
    """
    summary = _safe_generate_summary(backends, candidate, state)
    save_state(state, run_dir)  # расход на резюме тоже должен попасть в state
    _print_finish(state, _write_report(state, run_dir, summary))


def _safe_generate_summary(backends: dict, candidate: Candidate,
                           state: RunState) -> str:
    """Резюме без риска потерять уже оплаченный прогон.

    _generate_summary — единственный вызов LLM без ретрая run_pipeline; цикл к
    этому моменту полностью оплачен и завершён, поэтому падение здесь не
    должно оставить пользователя без report.md и с голым трейсбеком. При
    ошибке подставляем ту же заглушку, что и офлайн-пересборка (REBUILT_SUMMARY),
    и явно говорим об этом в stdout — отчёт всё равно пишется.
    """
    try:
        return _generate_summary(backends, candidate, state)
    except Exception as e:
        print(f"Не удалось сгенерировать резюме ({e}); report.md записан без него.")
        return REBUILT_SUMMARY


# --- команды ---------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    idea_path = Path(args.idea)
    idea_text = _read_idea(idea_path)
    candidate_dir = (Path(args.candidate) if args.candidate
                     else resolve_candidate_dir(candidates_root=CANDIDATES_ROOT,
                                                domain=args.domain))
    candidate = apply_max_iter(load_candidate(candidate_dir), args.max_iter)
    run_dir = make_run_dir(runs_root=RUNS_ROOT, idea_path=idea_path,
                           now_str=datetime.now().strftime(RUN_DIR_TIMESTAMP_FORMAT))
    if (run_dir / "state.json").exists():
        # секунды в имени каталога сужают окно коллизии, но не закрывают его;
        # настоящая защита от затирания оплаченного прогона — этот отказ
        sys.exit(f"Каталог прогона уже существует и содержит state.json: "
                 f"{run_dir}. Возможно, прогон для этой идеи уже запущен "
                 f"в эту секунду — подождите и повторите, чтобы не затереть "
                 f"чужой state.json.")
    run_dir.mkdir(parents=True, exist_ok=True)
    _print_start(run_dir, candidate)
    backends = _build_role_backends(candidate)
    state = run_pipeline(backends, candidate, idea_text=idea_text,
                         run_dir=run_dir,
                         candidate_dir=candidate_dir.resolve(),
                         on_step=ProgressPrinter())
    _finish_run(backends, candidate, state, run_dir)


def cmd_resume(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    saved = load_state(run_dir)
    if saved.stop_reason:
        sys.exit(f"Прогон {saved.run_id} уже завершён "
                 f"(причина: {saved.stop_reason}). Чтобы пересобрать отчёт: "
                 f"python -m kaidzen report {run_dir}")
    candidate = load_candidate(_resume_candidate_dir(saved))
    _print_start(run_dir, candidate, max_iterations=_loop_from_state(saved).max_iterations)
    backends = _build_role_backends(candidate)
    state = run_pipeline(backends, candidate, idea_text=saved.original_idea,
                         run_dir=run_dir, resume=True,
                         on_step=ProgressPrinter())
    _finish_run(backends, candidate, state, run_dir)


def cmd_report(args: argparse.Namespace) -> None:
    """Оффлайн-пересборка отчёта: ни одного обращения к API."""
    run_dir = Path(args.run_dir)
    state = load_state(run_dir)
    print(f"Отчёт пересобран: {_write_report(state, run_dir, REBUILT_SUMMARY)}")


# --- вывод хода эволюции ---------------------------------------------------

def print_evolve_event(message: str) -> None:
    """Один шаг поколения в stdout.

    Передаётся в EvolveContext как on_event. Тексты собирает сам оркестратор —
    он единственный знает, что именно сейчас решилось: какой кандидат на какой
    идее, сколько попарок выиграно, что и почему сказал Gate. CLI их только
    печатает, не переформулируя, иначе причина решения Gate потерялась бы
    по дороге. Поколение идёт десятки минут, и молчание в терминале
    неотличимо от зависания.
    """
    print(f"[{EVOLVE_PREFIX}] {message}")


def _print_evolve_start(evolve_dir: Path, benchmark: Benchmark, meta: MetaConfig,
                        *, champion_id: str, generations: int,
                        concurrency: int) -> None:
    print(f"Эволюция: {evolve_dir.name}")
    print(f"Домен: {benchmark.domain}, чемпион: {champion_id}")
    print(f"Бенчмарк: {len(benchmark.train)} идей train, "
          f"{len(benchmark.holdout)} holdout")
    _print_meta_backend(meta)
    print(f"Поколений: до {generations}, параллельных прогонов: {concurrency}")


def _print_meta_backend(meta: MetaConfig) -> None:
    """На чём ходят мета-роли. Ключи не печатаются — только тип и модели."""
    print(f"  мета-бэкенд: {meta.backend.get('type')}")
    print(f"  диагност и мутатор: {meta.deep_model}")
    print(f"  слепой судья: {meta.judge_model}")


def _print_evolve_finish(state: EvolveState, evolve_dir: Path) -> None:
    promoted = [gen.promoted_id for gen in state.generations if gen.promoted_id]
    print(f"Остановка: {state.stop_reason}")
    print(f"Поколений пройдено: {state.generation}")
    print(f"Промоций: {len(promoted)}"
          + (f" ({', '.join(promoted)})" if promoted else ""))
    print(f"Чемпион: {state.champion_id}")
    print(f"Сводка: {evolve_dir / SUMMARY_FILE}")
    _print_pending_hint(state, evolve_dir)


def _print_pending_hint(state: EvolveState, evolve_dir: Path) -> None:
    """Незакрытый чекпоинт — не авария, а ожидание человека: скажем, что делать."""
    record = pending_checkpoint(state)
    if record is None:
        return
    print(f"Ждёт решения человека: {record.path}")
    print(f"Прочитать и решить: python -m kaidzen checkpoint {evolve_dir}")


# --- шаги команд эволюции --------------------------------------------------

def _load_benchmark(domain: str) -> Benchmark:
    try:
        return load_benchmark(BENCHMARK_ROOT, domain=domain)
    except BenchmarkEmpty as e:
        sys.exit(f"Эволюционировать не на чем: {e}. Положите туда несколько "
                 f"markdown-файлов с идеями и повторите.")


def _evolve_context(evolve_dir: Path, benchmark: Benchmark, meta: MetaConfig,
                    concurrency: int) -> EvolveContext:
    """Контекст поколения. Бэкенд мета-уровня строится здесь, до первого вызова:
    ключи требуются только те, что объявил сам мета-конфиг, а на подписке — ни
    одного."""
    try:
        backend = build_meta_backend(meta)
    except BackendError as e:
        sys.exit(f"Не удалось подготовить бэкенд мета-уровня: {e}")
    return EvolveContext(evolve_dir=evolve_dir, candidates_root=CANDIDATES_ROOT,
                         benchmark=benchmark, meta_backend=backend, meta=meta,
                         concurrency=concurrency, on_event=print_evolve_event)


def _refuse_resume_while_checkpoint_pending(state: EvolveState,
                                            evolve_dir: Path) -> None:
    """Пока чекпоинт висит, поколения не идут — это решает человек, не CLI."""
    record = pending_checkpoint(state)
    if record is None:
        return
    sys.exit(f"Поколение {record.generation} ждёт решения человека: "
             f"{record.path}. Сначала прочитайте сводку и решите: "
             f"python -m kaidzen checkpoint {evolve_dir} --approve | --reject")


# --- команды эволюции ------------------------------------------------------

def cmd_evolve(args: argparse.Namespace) -> None:
    benchmark = _load_benchmark(args.domain)
    champion_dir = resolve_candidate_dir(candidates_root=CANDIDATES_ROOT,
                                         domain=args.domain).resolve()
    evolve_dir = make_evolve_dir(
        evolve_root=EVOLVE_ROOT, domain=args.domain,
        now_str=datetime.now().strftime(RUN_DIR_TIMESTAMP_FORMAT))
    evolve_dir.mkdir(parents=True, exist_ok=True)
    meta = MetaConfig()
    ctx = _evolve_context(evolve_dir, benchmark, meta, args.concurrency)
    _print_evolve_start(evolve_dir, benchmark, meta,
                        champion_id=champion_dir.name,
                        generations=args.generations,
                        concurrency=args.concurrency)
    state = run_evolve(ctx, champion_dir=champion_dir,
                       max_generations=args.generations)
    _print_evolve_finish(state, evolve_dir)


def cmd_evolve_resume(args: argparse.Namespace) -> None:
    """Продолжение evolve-прогона.

    Лимиты берутся из состояния, а не из флагов — тот же контракт, что у
    resume Уровня 1: прогон, начатый на двух поколениях, не должен получить
    пять только потому, что его продолжили другой командой.
    """
    evolve_dir = Path(args.evolve_dir)
    saved = load_evolve_state(evolve_dir)
    _refuse_resume_while_checkpoint_pending(saved, evolve_dir)
    benchmark = _load_benchmark(saved.domain)
    meta = MetaConfig()
    ctx = _evolve_context(evolve_dir, benchmark, meta, EVAL_CONCURRENCY)
    _print_evolve_start(evolve_dir, benchmark, meta,
                        champion_id=saved.champion_id,
                        generations=saved.max_generations,
                        concurrency=EVAL_CONCURRENCY)
    state = run_evolve(ctx, resume=True)
    _print_evolve_finish(state, evolve_dir)


def cmd_evolve_stop(args: argparse.Namespace) -> None:
    """Мягкая остановка: флаг на диске, который оркестратор читает на границе
    поколений. Жёстко бросать поколение нельзя — его eval-прогоны уже оплачены
    стенными часами."""
    evolve_dir = Path(args.evolve_dir)
    state = load_evolve_state(evolve_dir)
    request_stop(evolve_dir)
    print(f"Мягкая остановка запрошена: {state.evolve_id}")
    print("Поколение в работе дозавершится — уже сделанные прогоны не пропадут; "
          "новое не начнётся.")
    print(f"Продолжить позже: python -m kaidzen evolve-resume {evolve_dir}")


def cmd_checkpoint(args: argparse.Namespace) -> None:
    evolve_dir = Path(args.evolve_dir)
    if args.approve:
        _approve_checkpoint(evolve_dir)
    elif args.reject:
        _reject_checkpoint(evolve_dir)
    else:
        _show_checkpoint(evolve_dir)


def _show_checkpoint(evolve_dir: Path) -> None:
    """Без флага решение не принимается: сначала человек читает сводку."""
    state = load_evolve_state(evolve_dir)
    record = pending_checkpoint(state)
    if record is None:
        print(f"Незакрытых чекпоинтов нет. Чемпион: {state.champion_id}")
        return
    previous = record.previous_champion_id or NO_DATA
    print(f"Чекпоинт поколения {record.generation} ждёт решения.")
    print(f"Сравниваются на holdout-идеях: {previous} (предыдущий чемпион) "
          f"и {record.champion_id} (текущий).")
    if record.overfit:
        print("Похоже на подгонку под бенчмарк: на train лучше, на holdout — нет.")
    print(f"Сводка: {record.path}")
    print(f"Решение: python -m kaidzen checkpoint {evolve_dir} "
          f"--approve | --reject")


def _approve_checkpoint(evolve_dir: Path) -> None:
    record = approve_checkpoint(evolve_dir)
    print(f"Чекпоинт поколения {record.generation} принят: "
          f"чемпион {record.champion_id} остаётся.")
    print(f"Продолжить: python -m kaidzen evolve-resume {evolve_dir}")


def _reject_checkpoint(evolve_dir: Path) -> None:
    """Отклонение откатывает чемпиона и заканчивает evolve-прогон."""
    record = reject_checkpoint(evolve_dir)
    state = load_evolve_state(evolve_dir)
    print(f"Чекпоинт поколения {record.generation} отклонён: "
          f"чемпион откачен на {state.champion_id}.")
    print(f"Evolve-прогон завершён (причина: {STOP_CHECKPOINT_REJECTED}).")


COMMANDS = {"run": cmd_run, "resume": cmd_resume, "report": cmd_report,
            "evolve": cmd_evolve, "evolve-resume": cmd_evolve_resume,
            "evolve-stop": cmd_evolve_stop, "checkpoint": cmd_checkpoint}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaidzen",
        description="Доводка сырой идеи циклом Analyzer→Researcher→Refiner→Judge.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Прогнать идею через полный цикл.")
    run.add_argument("idea", help="Путь к markdown-файлу с сырой идеей.")
    run.add_argument("--domain", choices=DOMAINS, default=DEFAULT_DOMAIN,
                     help="Домен: из него берётся кандидат-чемпион.")
    run.add_argument("--candidate", default=None,
                     help="Путь к каталогу кандидата (важнее, чем --domain).")
    run.add_argument("--max-iter", type=int, default=None,
                     help="Переопределить лимит итераций кандидата.")

    resume = sub.add_parser("resume", help="Продолжить прерванный прогон.")
    resume.add_argument("run_dir", help="Каталог прогона со state.json.")

    report = sub.add_parser("report",
                            help="Пересобрать report.md без обращения к API.")
    report.add_argument("run_dir", help="Каталог прогона со state.json.")
    _add_evolve_commands(sub)
    return parser


def _add_evolve_commands(sub) -> None:
    """Команды мета-лупа. Ни одна из них не запускает эволюцию сама по себе:
    evolve отрабатывает заказанные поколения и выходит (ТЗ §4.5)."""
    evolve = sub.add_parser(
        "evolve", help="Эволюционировать кандидатов домена на бенчмарке.")
    evolve.add_argument("--domain", choices=DOMAINS, default=DEFAULT_DOMAIN,
                        help="Домен: из него берутся чемпион и идеи бенчмарка.")
    evolve.add_argument("--generations", type=positive_int,
                        default=MAX_GENERATIONS,
                        help="Сколько поколений прогнать максимум.")
    evolve.add_argument("--concurrency", type=positive_int,
                        default=EVAL_CONCURRENCY,
                        help="Сколько eval-прогонов идут параллельно.")

    resume = sub.add_parser("evolve-resume",
                            help="Продолжить прерванный evolve-прогон.")
    resume.add_argument("evolve_dir", help="Каталог evolve-прогона со state.json.")

    stop = sub.add_parser(
        "evolve-stop",
        help="Попросить остановиться: поколение в работе дозавершится.")
    stop.add_argument("evolve_dir", help="Каталог evolve-прогона со state.json.")

    checkpoint = sub.add_parser(
        "checkpoint", help="Показать или закрыть чекпоинт evolve-прогона.")
    checkpoint.add_argument("evolve_dir",
                            help="Каталог evolve-прогона со state.json.")
    decision = checkpoint.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true",
                          help="Согласиться с текущим чемпионом.")
    decision.add_argument("--reject", action="store_true",
                          help="Откатить чемпиона на предыдущего.")


def main(argv: list[str] | None = None) -> None:
    # под пайпом Python буферизует stdout блоками, и прогресс появляется
    # только в самом конце — на прогоне в десятки минут это выглядит как
    # зависание. Построчный режим возвращает смысл всему прогресс-выводу.
    sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except (FileNotFoundError, ValueError) as e:
        # ожидаемые ошибки пользователя: нет файла, битый конфиг, пустой указатель
        sys.exit(str(e))


if __name__ == "__main__":
    main()
