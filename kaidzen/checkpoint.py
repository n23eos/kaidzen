"""Чекпоинты человека: пауза мета-лупа каждые K поколений.

Третья защита от Гудхарта после объективных метрик и слепых попарок. Каждые
K поколений текущий и предыдущий чемпион прогоняются на holdout-идеях —
на тех, которых не видели ни диагност, ни мутатор. Расхождение «лучше на
train, не лучше на holdout» означает подгонку под бенчмарк, и сводка обязана
показать его явно: иначе holdout существует зря.

Модуль ничего не знает про оркестратор: прогоны ему подаёт вызывающий код
колбэком `evaluate`. Обратная зависимость (evolve → checkpoint) — односторонняя.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel, Field

from kaidzen.metrics import RunMetrics
from kaidzen.state import write_atomically

if TYPE_CHECKING:  # pragma: no cover - только для аннотаций, без цикла импортов
    from kaidzen.evolve import EvolveState

# каждые сколько поколений человек смотрит глазами
CHECKPOINT_EVERY = 3
CHECKPOINTS_DIR = "checkpoints"

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

# Метрика, по которой ловится подгонка: именно доля закрытых допущений —
# суть работы системы, а не производная от неё.
OVERFIT_METRIC = "assumptions_closed_rate"
# метрики сводки: числа, которые человек сравнивает глазами
SUMMARY_METRICS = ("assumptions_closed_rate", "partial_rate",
                   "grounded_changelog_rate", "iterations", "rollbacks")

NO_DATA = "—"
OVERFIT_NOTE = (
    "**Похоже на подгонку под бенчмарк:** на train чемпион лучше предыдущего, "
    "на holdout — нет. Holdout не участвовал в мутациях, поэтому верить надо "
    "ему, а не train.")
OK_NOTE = "Расхождения train/holdout не видно."


class ChampionRef(BaseModel):
    """Кого прогонять на holdout: id и каталог кандидата."""

    candidate_id: str
    candidate_dir: str


class HoldoutResult(BaseModel):
    """Что вернул прогон чемпиона на holdout-идеях."""

    metrics: RunMetrics | None = None
    reports: list[str] = Field(default_factory=list)


class CheckpointRecord(BaseModel):
    generation: int
    champion_id: str
    previous_champion_id: str = ""
    status: str = STATUS_PENDING
    path: str = ""
    overfit: bool = False


# evaluate(ref) -> HoldoutResult: прогон кандидата на holdout-идеях
Evaluator = Callable[[ChampionRef], HoldoutResult]


def checkpoint_path(evolve_dir: Path, generation: int) -> Path:
    return evolve_dir / CHECKPOINTS_DIR / f"gen{generation:03d}.md"


def make_checkpoint(*, evolve_dir: Path, generation: int,
                    champion: ChampionRef, previous: ChampionRef | None,
                    evaluate: Evaluator,
                    train_metrics: dict[str, RunMetrics]) -> CheckpointRecord:
    """Гоняет обоих чемпионов на holdout и пишет сводку для человека."""
    current = evaluate(champion)
    before = evaluate(previous) if previous is not None else HoldoutResult()
    overfit = _is_overfit(
        train_before=train_metrics.get(previous.candidate_id) if previous else None,
        train_after=train_metrics.get(champion.candidate_id),
        holdout_before=before.metrics, holdout_after=current.metrics)
    path = checkpoint_path(evolve_dir, generation)
    write_atomically(path, render_summary(
        generation=generation, champion=champion, previous=previous,
        current=current, before=before, overfit=overfit,
        train_metrics=train_metrics))
    return CheckpointRecord(
        generation=generation, champion_id=champion.candidate_id,
        previous_champion_id=previous.candidate_id if previous else "",
        path=str(path), overfit=overfit)


def _value(metrics: RunMetrics | None, name: str) -> float | None:
    return getattr(metrics, name) if metrics is not None else None


def _delta(before: RunMetrics | None, after: RunMetrics | None,
           name: str = OVERFIT_METRIC) -> float | None:
    """Прирост метрики. None, если сравнивать не с чем."""
    old, new = _value(before, name), _value(after, name)
    if old is None or new is None:
        return None
    return new - old


def _is_overfit(*, train_before: RunMetrics | None, train_after: RunMetrics | None,
                holdout_before: RunMetrics | None,
                holdout_after: RunMetrics | None) -> bool:
    """Вырос на train, не вырос на holdout — подгонка."""
    train = _delta(train_before, train_after)
    holdout = _delta(holdout_before, holdout_after)
    if train is None or holdout is None:
        return False
    return train > 0 and holdout <= 0


# --- сводка ---------------------------------------------------------------

def render_summary(*, generation: int, champion: ChampionRef,
                   previous: ChampionRef | None, current: HoldoutResult,
                   before: HoldoutResult, overfit: bool,
                   train_metrics: dict[str, RunMetrics]) -> str:
    """Markdown-сводка чекпоинта: цифры сверху, отчёты целиком снизу."""
    previous_id = previous.candidate_id if previous else NO_DATA
    parts = [
        f"# Чекпоинт поколения {generation}",
        "",
        f"- Текущий чемпион: {champion.candidate_id}",
        f"- Предыдущий чемпион: {previous_id}",
        "",
        OVERFIT_NOTE if overfit else OK_NOTE,
        "",
        _render_table("Метрики на holdout", before.metrics, current.metrics),
        "",
        _render_table("Метрики на train",
                      train_metrics.get(previous.candidate_id) if previous else None,
                      train_metrics.get(champion.candidate_id)),
        "",
        _render_reports(f"Отчёты holdout: {previous_id}", before.reports),
        "",
        _render_reports(f"Отчёты holdout: {champion.candidate_id}",
                        current.reports),
        "",
        "Решение: `checkpoint --approve` или `checkpoint --reject`.",
    ]
    return "\n".join(parts) + "\n"


def _render_table(title: str, before: RunMetrics | None,
                  after: RunMetrics | None) -> str:
    lines = [f"## {title}", "", "| Метрика | Предыдущий | Текущий | Дельта |",
             "|---|---|---|---|"]
    for name in SUMMARY_METRICS:
        old, new = _value(before, name), _value(after, name)
        delta = _delta(before, after, name)
        lines.append(f"| {name} | {_fmt(old)} | {_fmt(new)} | {_fmt(delta, sign=True)} |")
    return "\n".join(lines)


def _fmt(value: float | None, sign: bool = False) -> str:
    if value is None:
        return NO_DATA
    return f"{value:+.2f}" if sign else f"{value:.2f}"


def _render_reports(title: str, reports: list[str]) -> str:
    if not reports:
        return f"## {title}\n\n{NO_DATA}"
    body = "\n\n---\n\n".join(reports)
    return f"## {title}\n\n{body}"


# --- решение человека -----------------------------------------------------

def pending_checkpoint(state: "EvolveState") -> CheckpointRecord | None:
    """Незакрытый чекпоинт: пока он есть, поколения не идут."""
    for record in state.checkpoints:
        if record.status == STATUS_PENDING:
            return record
    return None


def approve(state: "EvolveState") -> CheckpointRecord:
    """Человек согласен с текущим чемпионом — поколения продолжаются."""
    record = _require_pending(state)
    record.status = STATUS_APPROVED
    return record


def reject(state: "EvolveState") -> CheckpointRecord:
    """Человек не согласен: чемпион откатывается к предыдущему.

    Откат тривиален, потому что кандидаты иммутабельны — предыдущий чемпион
    никуда не делся, достаточно вернуть на него указатель.
    """
    record = _require_pending(state)
    record.status = STATUS_REJECTED
    if state.previous_champion_id:
        state.champion_id = state.previous_champion_id
        state.champion_dir = state.previous_champion_dir
        state.previous_champion_id = ""
        state.previous_champion_dir = ""
    return record


def _require_pending(state: "EvolveState") -> CheckpointRecord:
    record = pending_checkpoint(state)
    if record is None:
        raise ValueError(
            f"у evolve-прогона {state.evolve_id} нет незакрытого чекпоинта")
    return record
