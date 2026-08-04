"""Журнал эволюции: что уже пробовали на этом домене и чем это кончилось.

Один файл на домен, `candidates/EVOLUTION-<domain>.json`, и он лежит в git —
в отличие от `evolve/`, который в .gitignore. Причина простая: каталог прогона
это артефакт, а журнал — накопленное знание. Без него каждый evolve-прогон
стартует с нуля: мутатор второй раз предлагает уже отклонённую правку, а
удачную находку прошлого прогона переписывает поверх, не зная, что она удачная.

Дозапись, а не перезапись. Источник истины — файл на диске: перед добавлением
записи он перечитывается целиком, новая запись кладётся в хвост, и результат
пишется атомарно (`write_atomically`). Поэтому обрыв прогона между поколениями
не стоит журналу ничего, а состояние процесса в памяти никогда не может
затереть чужую запись.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from kaidzen.state import write_atomically

LOG_PREFIX = "EVOLUTION-"
LOG_SUFFIX = ".json"
JSON_INDENT = 2

OUTCOME_PROMOTED = "promoted"
OUTCOME_REJECTED = "rejected"
OUTCOME_UNSTABLE = "unstable"
OUTCOME_DISCARDED = "discarded"

Outcome = Literal["promoted", "rejected", "unstable", "discarded"]

# Сколько НЕпринятых записей уходит в промпт. Журнал растёт линейно по числу
# мутаций, а промпт не резиновый; принятые записи не усекаются никогда — это
# та самая память об удачах, ради которой журнал и заводился.
DIGEST_RECENT_LIMIT = 10

# что именно из метрик показываем в выжимке, в каком порядке и как называем;
# токены записаны на один прогон, и в тексте это должно быть видно, иначе
# число читается как расход всего поколения
DELTA_LABELS = {
    "assumptions_closed_rate": "assumptions_closed_rate",
    "partial_rate": "partial_rate",
    "high_closed": "high_closed",
    "output_tokens": "output_tokens (на прогон)",
}

OUTCOME_TITLES = {
    OUTCOME_PROMOTED: "ПРИНЯТО",
    OUTCOME_REJECTED: "ОТКЛОНЕНО",
    OUTCOME_UNSTABLE: "НЕСТАБИЛЬНО",
    OUTCOME_DISCARDED: "ОТБРОШЕНО (патч не записался)",
}


class EvolutionRecord(BaseModel):
    """Одна попытка мутации: замысел, что тронуто и чем кончилось."""

    evolve_id: str
    generation: int
    candidate_id: str
    parent_id: str = ""
    hypothesis: str = ""
    roles_touched: list[str] = Field(default_factory=list)
    outcome: Outcome
    gate_reason: str = ""
    win_rate: float | None = None
    metrics_delta: dict[str, float] = Field(default_factory=dict)
    comparable_ideas: int = 0

    @property
    def key(self) -> tuple[str, str]:
        """Одна попытка одного прогона. По ней ловится повторная запись."""
        return (self.evolve_id, self.candidate_id)


# --- файл ------------------------------------------------------------------

def log_path(root: Path, domain: str) -> Path:
    return Path(root) / f"{LOG_PREFIX}{domain}{LOG_SUFFIX}"


def load_records(root: Path, domain: str) -> list[EvolutionRecord]:
    """Журнал с диска. Битый файл — ошибка, а не пустой список.

    Молча начать с чистого листа тут хуже всего: накопленное знание исчезло бы
    незаметно, и следующая же запись затёрла бы остаток файла.
    """
    path = log_path(root, domain)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [EvolutionRecord.model_validate(item) for item in data]
    except Exception as e:
        raise ValueError(f"журнал эволюции {path} не читается: {e}") from e


def append_record(root: Path, domain: str, record: EvolutionRecord) -> bool:
    """Дописывает запись к тому, что лежит на диске. Повтор попытки не дублирует.

    Возвращает False, если такая попытка уже записана: стадия Gate может
    повториться после resume, и поколение не должно раздваиваться в журнале.
    """
    records = load_records(root, domain)
    if any(existing.key == record.key for existing in records):
        return False
    records.append(record)
    write_atomically(log_path(root, domain), _dump(records))
    return True


def _dump(records: list[EvolutionRecord]) -> str:
    return json.dumps([r.model_dump() for r in records],
                      ensure_ascii=False, indent=JSON_INDENT) + "\n"


# --- выжимка для промпта ---------------------------------------------------

def digest_records(records: list[EvolutionRecord],
                   limit: int = DIGEST_RECENT_LIMIT) -> list[EvolutionRecord]:
    """Все принятые записи плюс последние `limit` остальных, в порядке журнала."""
    others = [i for i, r in enumerate(records)
              if r.outcome != OUTCOME_PROMOTED]
    keep = set(others[-limit:]) if limit > 0 else set()
    return [r for i, r in enumerate(records)
            if r.outcome == OUTCOME_PROMOTED or i in keep]


def render_digest(records: list[EvolutionRecord]) -> str:
    """Выжимка для диагноста: что пробовали, что сработало, что отклонено."""
    chosen = digest_records(records)
    if not chosen:
        return ""
    lines = ["## Журнал эволюции: что уже пробовали", ""]
    for r in chosen:
        lines.extend(_render_record(r))
    return "\n".join(lines)


def _render_record(r: EvolutionRecord) -> list[str]:
    roles = ", ".join(r.roles_touched) or "—"
    title = OUTCOME_TITLES.get(r.outcome, r.outcome)
    lines = [f"- поколение {r.generation}, кандидат {r.candidate_id} "
             f"(роли: {roles}) — {title} [{r.outcome}]",
             f"  - гипотеза: {r.hypothesis or '—'}"]
    if r.gate_reason:
        lines.append(f"  - решение: {r.gate_reason}")
    if r.win_rate is not None:
        lines.append(f"  - побед в попарках {r.win_rate:.0%}, "
                     f"сравнимых идей {r.comparable_ideas}")
    deltas = _render_deltas(r)
    if deltas:
        lines.append(f"  - дельты метрик: {deltas}")
    lines.append("")
    return lines


def _render_deltas(r: EvolutionRecord) -> str:
    parts = [f"{label} {r.metrics_delta[key]:+g}"
             for key, label in DELTA_LABELS.items() if key in r.metrics_delta]
    return ", ".join(parts)


def render_do_not_break(records: list[EvolutionRecord]) -> str:
    """Список принятых находок для мутатора — с явным «не ломать».

    Правка, приведшая к промоции, уже оплачена поколением прогонов. Переписать
    её поверх — не смелость, а откат к тому, что уже проиграло сравнение.
    """
    promoted = [r for r in records if r.outcome == OUTCOME_PROMOTED]
    if not promoted:
        return ""
    lines = ["## Уже принятые находки — не ломать", ""]
    for r in promoted:
        roles = ", ".join(r.roles_touched) or "—"
        lines.append(f"- поколение {r.generation}, роли {roles}: "
                     f"{r.hypothesis or r.gate_reason}")
        deltas = _render_deltas(r)
        if deltas:
            lines.append(f"  - что это дало: {deltas}")
    lines.extend(["", "Эти правки уже прошли Gate на объективных метриках. "
                  "Переписывание их поверх — регресс, а не улучшение: если "
                  "трогаешь то же место, объясни в rationale, что именно "
                  "сохраняешь из принятой находки."])
    return "\n".join(lines)
