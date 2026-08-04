"""Одноразовый перенос истории прошлых evolve-прогонов в журнал эволюции.

Журнал появился позже самих прогонов, поэтому знание пяти первых поколений
лежало только в их state.json и следующему прогону не досталось бы. Скрипт
восстанавливает записи из состояний и дописывает их в журнал.

Поле roles_touched в старых состояниях отсутствовало — восстанавливаем его из
текста rationale по упоминанию имени роли в начале строки: мутатор писал
обоснование построчно, начиная каждую правку с имени роли.

Запуск: .venv/bin/python scripts/backfill_evolution_log.py [--apply]
Без --apply только показывает, что будет записано.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from kaidzen.candidate import ROLES
from kaidzen.evolution_log import EvolutionRecord, append_record, load_records

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "candidates"
EVOLVE = ROOT / "evolve"

# «researcher: добавлено ...» или «config.researcher_focus: ...» в начале строки
ROLE_LINE = re.compile(r"^\s*(?:config\.)?(" + "|".join(ROLES) + r")\b",
                       re.MULTILINE)

# Продвинут по правилу, которого больше нет: этот кандидат раздул долю закрытых
# ужиманием реестра допущений, прошёл тогдашний Gate и был откачен вручную.
# В журнале он должен лежать как отклонённый — иначе диагност примет
# Гудхарт-мутацию за образец и повторит её.
ROLLED_BACK = {
    "gen001-b": "продвинут ошибочно и откачен: доля выросла за счёт ужатого "
                "реестра, а не за счёт закрытых вопросов (Gate тогда этого не ловил)",
}

STATUS_TO_OUTCOME = {
    "promoted": "promoted",
    "rejected": "rejected",
    "unstable": "unstable",
}


def roles_from_rationale(rationale: str) -> list[str]:
    """Какие роли тронула мутация — по началам строк обоснования."""
    found = ROLE_LINE.findall(rationale or "")
    return sorted(set(found))


def delta(champion: dict | None, challenger: dict | None) -> dict[str, float]:
    """Сдвиг метрик челленджера относительно чемпиона того же поколения."""
    if not champion or not challenger:
        return {}
    keys = ("assumptions_closed_rate", "partial_rate", "high_closed")
    out = {k: round(challenger.get(k, 0) - champion.get(k, 0), 4) for k in keys}
    runs_a = max(1, champion.get("runs", 1))
    runs_b = max(1, challenger.get("runs", 1))
    out["output_tokens"] = round(
        challenger.get("output_tokens", 0) / runs_b
        - champion.get("output_tokens", 0) / runs_a)
    return out


def comparable(champion: dict, record: dict) -> int:
    """Идеи, на которых отработали обе стороны — на них и шло сравнение."""
    ok = lambda runs: {r["idea"] for r in runs if r.get("ok")}  # noqa: E731
    return len(ok(champion.get("runs", [])) & ok(record.get("runs", [])))


def records_of(state: dict) -> list[EvolutionRecord]:
    out: list[EvolutionRecord] = []
    for gen in state.get("generations", []):
        champion = gen.get("champion") or {}
        champion_metrics = champion.get("metrics")
        hypotheses = "; ".join(gen.get("hypotheses", []))
        for ch in gen.get("challengers", []):
            correction = ROLLED_BACK.get(ch["candidate_id"])
            out.append(EvolutionRecord(
                evolve_id=state["evolve_id"],
                generation=gen["number"],
                candidate_id=ch["candidate_id"],
                parent_id=gen.get("champion_id", ""),
                hypothesis=(ch.get("rationale") or hypotheses)[:600],
                roles_touched=roles_from_rationale(ch.get("rationale", "")),
                outcome=("rejected" if correction
                         else STATUS_TO_OUTCOME.get(ch.get("status", ""), "rejected")),
                gate_reason=correction or ch.get("gate_reason", ""),
                win_rate=ch.get("win_rate"),
                metrics_delta=delta(champion_metrics, ch.get("metrics")),
                comparable_ideas=comparable(champion, ch),
            ))
        for text in gen.get("discarded", []):
            candidate_id, _, reason = text.partition(":")
            out.append(EvolutionRecord(
                evolve_id=state["evolve_id"],
                generation=gen["number"],
                candidate_id=candidate_id.strip(),
                parent_id=gen.get("champion_id", ""),
                hypothesis=hypotheses[:600],
                outcome="discarded",
                gate_reason=reason.strip(),
            ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="записать в журнал, а не только показать")
    args = parser.parse_args()

    states = sorted(EVOLVE.glob("*/state.json"))
    written = 0
    for path in states:
        state = json.loads(path.read_text(encoding="utf-8"))
        domain = state["domain"]
        for record in records_of(state):
            mark = " "
            if args.apply and append_record(CANDIDATES, domain, record):
                written += 1
                mark = "+"
            print(f" {mark} {record.evolve_id} / {record.candidate_id}: "
                  f"{record.outcome}, роли {record.roles_touched or '—'}")

    if args.apply:
        for domain in {json.loads(p.read_text(encoding='utf-8'))["domain"]
                       for p in states}:
            total = len(load_records(CANDIDATES, domain))
            print(f"\nжурнал {domain}: {total} записей (дописано {written})")
    else:
        print("\nэто предпросмотр; для записи добавь --apply")


if __name__ == "__main__":
    main()
