"""CLI Kaidzen: run / resume / report.

Точка входа для человека за терминалом: прогон стоит денег и идёт минутами,
поэтому команды печатают ход дела и валятся с понятным текстом, а не трейсбеком.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from kaidzen.candidate import Candidate, load_candidate
from kaidzen.llm import LLMClient
from kaidzen.orchestrator import run_pipeline
from kaidzen.report import build_report
from kaidzen.state import ApiUsage, RunState, load_state, save_state

SUMMARY_MODEL = "claude-sonnet-5"
SUMMARY_TEMPERATURE = 0.2
SUMMARY_SYSTEM = (
    "Ты пишешь executive summary по финальной версии идеи. "
    "Дай 3–5 предложений: что за идея, для кого, как работает и что "
    "проверено. Пиши по-русски, без воды и без маркетинговых прилагательных."
)

RUN_DIR_TIMESTAMP_FORMAT = "%Y-%m-%d-%H%M"
CHAMPION_PREFIX = "CHAMPION-"
CANDIDATES_ROOT = Path("candidates")
RUNS_ROOT = Path("runs")
REPORT_FILENAME = "report.md"
API_KEY_ENV = "ANTHROPIC_API_KEY"
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


def slugify(text: str) -> str:
    """Нижний регистр, любые не-буквенно-цифровые серии — в один дефис."""
    slug = re.sub(r"[\W_]+", "-", text.lower(), flags=re.UNICODE).strip("-")
    return slug or FALLBACK_SLUG


def apply_max_iter(candidate: Candidate, max_iter: int | None) -> Candidate:
    """Новый кандидат с изменённым лимитом итераций (исходный не меняем)."""
    if max_iter is None:
        return candidate
    loop = candidate.config.loop.model_copy(update={"max_iterations": max_iter})
    config = candidate.config.model_copy(update={"loop": loop})
    return candidate.model_copy(update={"config": config})


# --- вывод хода прогона ----------------------------------------------------

def _print_start(run_dir: Path, candidate: Candidate) -> None:
    print(f"Прогон: {run_dir.name}")
    print(f"Кандидат: {candidate.candidate_id} "
          f"(домен {candidate.config.domain}, "
          f"до {candidate.config.loop.max_iterations} итераций)")


def _print_iterations(state: RunState) -> None:
    """Итерации печатаются постфактум по state.versions.

    Оркестратор не отдаёт колбэки по ходу цикла, а трогать его ради этого
    не стали: детали итераций всё равно целиком лежат в состоянии.
    """
    for version in state.versions:
        if version.rolled_back:
            print(f"  итерация {version.n}: версия откачена судьёй")
        elif version.judge is not None:
            print(f"  итерация {version.n}: оценка {version.judge.total:.1f} "
                  f"(дельта {version.judge.delta_vs_previous:+.1f})")
        else:
            print(f"  итерация {version.n}: без оценки")


def _print_finish(state: RunState, report_path: Path) -> None:
    usage = state.api_usage
    print(f"Остановка: {state.stop_reason}")
    print(f"Итераций: {state.iteration}")
    print(f"Токены: {usage.input_tokens} входных, {usage.output_tokens} выходных; "
          f"веб-поисков: {usage.web_searches}")
    print(f"Отчёт: {report_path}")


# --- шаги команд -----------------------------------------------------------

def _require_api_key() -> None:
    """Ключ проверяем до любой работы, чтобы не падать после чтения файлов."""
    if not os.environ.get(API_KEY_ENV):
        sys.exit(f"Не задана переменная окружения {API_KEY_ENV}. "
                 f"Экспортируйте ключ Anthropic API и повторите.")


def _read_idea(idea_path: Path) -> str:
    if not idea_path.is_file():
        sys.exit(f"Файл идеи не найден: {idea_path}")
    return idea_path.read_text(encoding="utf-8")


def _generate_summary(llm, state: RunState) -> str:
    """Единственный вызов LLM вне цикла: короткое резюме финальной идеи."""
    out = llm.structured(model=SUMMARY_MODEL, system=SUMMARY_SYSTEM,
                         user=state.current_idea_text(), schema=SummaryOutput,
                         temperature=SUMMARY_TEMPERATURE)
    state.api_usage = ApiUsage.model_validate(llm.usage, from_attributes=True)
    return out.summary


def _write_report(state: RunState, run_dir: Path, summary_text: str) -> Path:
    report_path = run_dir / REPORT_FILENAME
    report_path.write_text(build_report(state, summary_text=summary_text),
                           encoding="utf-8")
    return report_path


def _finish_run(llm, state: RunState, run_dir: Path) -> None:
    """Общий хвост run и resume: резюме, отчёт, итоговая сводка."""
    _print_iterations(state)
    summary = _generate_summary(llm, state)
    save_state(state, run_dir)  # расход на резюме тоже должен попасть в state
    _print_finish(state, _write_report(state, run_dir, summary))


# --- команды ---------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    _require_api_key()
    idea_path = Path(args.idea)
    idea_text = _read_idea(idea_path)
    candidate_dir = (Path(args.candidate) if args.candidate
                     else resolve_candidate_dir(candidates_root=CANDIDATES_ROOT,
                                                domain=args.domain))
    candidate = apply_max_iter(load_candidate(candidate_dir), args.max_iter)
    run_dir = make_run_dir(runs_root=RUNS_ROOT, idea_path=idea_path,
                           now_str=datetime.now().strftime(RUN_DIR_TIMESTAMP_FORMAT))
    run_dir.mkdir(parents=True, exist_ok=True)
    _print_start(run_dir, candidate)
    llm = LLMClient()
    state = run_pipeline(llm, candidate, idea_text=idea_text, run_dir=run_dir)
    _finish_run(llm, state, run_dir)


def cmd_resume(args: argparse.Namespace) -> None:
    _require_api_key()
    run_dir = Path(args.run_dir)
    saved = load_state(run_dir)
    if saved.stop_reason:
        sys.exit(f"Прогон {saved.run_id} уже завершён "
                 f"(причина: {saved.stop_reason}). Чтобы пересобрать отчёт: "
                 f"python -m kaidzen report {run_dir}")
    candidate = load_candidate(CANDIDATES_ROOT / saved.candidate_id)
    _print_start(run_dir, candidate)
    llm = LLMClient()
    state = run_pipeline(llm, candidate, idea_text=saved.original_idea,
                         run_dir=run_dir, resume=True)
    _finish_run(llm, state, run_dir)


def cmd_report(args: argparse.Namespace) -> None:
    """Оффлайн-пересборка отчёта: ни одного обращения к API."""
    run_dir = Path(args.run_dir)
    state = load_state(run_dir)
    print(f"Отчёт пересобран: {_write_report(state, run_dir, REBUILT_SUMMARY)}")


COMMANDS = {"run": cmd_run, "resume": cmd_resume, "report": cmd_report}


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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        COMMANDS[args.command](args)
    except (FileNotFoundError, ValueError) as e:
        # ожидаемые ошибки пользователя: нет файла, битый конфиг, пустой указатель
        sys.exit(str(e))


if __name__ == "__main__":
    main()
