# Kaidzen Level 2 (Meta-Loop) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `kaidzen evolve` эволюционирует кандидатов (конфиг + промпты) Уровня 1: гоняет их на бенчмарке идей, сравнивает результаты вслепую, продвигает победителя только при непроседании объективных метрик, с чекпоинтами человека.

**Architecture:** Поверх готового Уровня 1. Diagnostician читает метрики чемпиона → Mutator порождает челленджеров → каждый прогоняется настоящим `run_pipeline` на train-идеях → Meta-Judge вслепую сравнивает отчёты → Gate пускает или отклоняет. Состояние evolve-прогона атомарно, прогон возобновляем и останавливаем.

**Tech Stack:** тот же — Python 3.12+, pydantic v2, pyyaml, pytest. Бэкенд по умолчанию `subscription` (без ключей).

**Spec:** `docs/specs/2026-08-03-idea-refinement-loop-tz.md` §4, с поправками ниже; `docs/specs/2026-08-03-multi-backend-addendum.md`.

---

## Поправки к ТЗ §4 по итогам живых прогонов

Три вводные изменились после того, как Уровень 1 отработал вживую. План написан под новые.

**Ограничение — не деньги, а время и лимиты.** ТЗ §4.6 считало стоимость поколения в API-токенах. На подписке прогон бесплатен, но идёт 2–4 минуты. Поколение из 2 челленджеров × 5 идей — это 10 прогонов, то есть до получаса чистого времени последовательно. Поэтому: бюджет измеряется прогонами и стенными часами, а не токенами; eval-прогоны идут параллельно с малым пулом (см. Task 6); ошибка лимита Claude обрабатывается как временная.

**Появилась метрика, которой не было в ТЗ: `partial_rate`.** Живой прогон показал главный отказ качества — Researcher возвращает `partial` на всё подряд, цикл не закрывает ни одного допущения, но выглядит работающим. Ручная правка промпта сдвинула долю с 3/3 до 2/3. Это ровно та величина, которую мета-луп должен давить автоматически, и она входит в Gate как жёсткий критерий.

**Бенчмарк почти пуст.** Есть одна идея (`benchmark/business/ideas/smoke-voice-tasks.md`). Задачи 1–8 на ней отлаживаются; Task 9 требует 6–10 реальных идей от пользователя. Это единственная внешняя зависимость плана.

---

## File Structure

```
kaidzen/
├── metrics.py            # объективные метрики из state (чистые функции)
├── gate.py               # правила промоции челленджера
├── evolve.py             # оркестратор поколений, чекпоинты, resume
├── mutation.py           # запись нового кандидата на диск, meta.json, линия предков
├── benchmark.py          # загрузка идей, детерминированная разбивка train/holdout
├── roles/meta/
│   ├── diagnostician.py
│   ├── mutator.py
│   └── meta_judge.py
└── __main__.py           # + evolve / evolve-resume / evolve-stop / checkpoint

benchmark/<domain>/ideas/*.md
evolve/<evolve_id>/{state.json, summary.md, checkpoints/genNNN.md}   # gitignore
candidates/CHAMPION-<domain>                                          # уже есть
```

---

### Task 1: Объективные метрики

**Files:**
- Create: `kaidzen/metrics.py`
- Test: `tests/test_metrics.py`

Метрики считаются кодом из `RunState`, без участия LLM — это половина защиты от Гудхарта. Вторая половина (слепое сравнение) в Task 7.

- [ ] **Step 1: Failing tests**

```python
# tests/test_metrics.py
from kaidzen.metrics import RunMetrics, run_metrics, aggregate
from kaidzen.state import Assumption, Fact, JudgeResult, RunState, Version


def state(assumptions=(), versions=(), stop_reason="plateau", iteration=2,
          rollbacks=0):
    return RunState(run_id="r", candidate_id="c", config={}, original_idea="i",
                    assumptions=list(assumptions), versions=list(versions),
                    stop_reason=stop_reason, iteration=iteration,
                    rollbacks=rollbacks)


def a(id_, criticality="high", status="unverified", facts=()):
    return Assumption(id=id_, text="t", criticality=criticality, status=status,
                      facts=list(facts))


FACT = Fact(claim="c", source_url="https://example.com/a")
BAD_FACT = Fact(claim="c", source_url="https://example.com/b")


def test_closed_rate_counts_only_high_criticality():
    m = run_metrics(state([
        a("A1", status="confirmed"), a("A2", status="refuted"),
        a("A3", status="unverified"), a("A4", "low", status="unverified")]))
    assert m.high_total == 3
    assert m.high_closed == 2
    assert m.assumptions_closed_rate == pytest.approx(2 / 3)


def test_partial_is_not_closed_and_has_its_own_rate():
    m = run_metrics(state([a("A1", status="partial"), a("A2", status="confirmed")]))
    assert m.high_closed == 1
    assert m.partial_rate == pytest.approx(0.5)


def test_untestable_counts_as_closed():
    """Непроверяемое поиском — честный результат, а не невыполненная работа."""
    m = run_metrics(state([a("A1", status="untestable")]))
    assert m.assumptions_closed_rate == 1.0


def test_facts_with_sources_rate():
    m = run_metrics(state([a("A1", status="confirmed", facts=[FACT, BAD_FACT])]))
    assert m.facts_total == 2
    assert m.facts_with_sources == 2


def test_zero_assumptions_gives_zero_rates_not_crash():
    m = run_metrics(state())
    assert m.assumptions_closed_rate == 0.0
    assert m.partial_rate == 0.0


def test_stop_reason_and_iterations_are_carried():
    m = run_metrics(state(stop_reason="max_iterations", iteration=6))
    assert m.stop_reason == "max_iterations"
    assert m.iterations == 6
    assert m.hit_iteration_limit is True


def test_grounded_changelog_rate():
    v = Version(n=1, idea_text="v", changelog=[
        ChangelogEntry(change="a", reason="r", grounded_in=["A1"]),
        ChangelogEntry(change="b", reason="r", grounded_in=[)]])
    m = run_metrics(state(versions=[v]))
    assert m.grounded_changelog_rate == pytest.approx(0.5)


def test_aggregate_averages_across_runs():
    one = run_metrics(state([a("A1", status="confirmed")]))
    two = run_metrics(state([a("A1", status="unverified")]))
    agg = aggregate([one, two])
    assert agg.assumptions_closed_rate == pytest.approx(0.5)
    assert agg.runs == 2
```

Note: импорты `pytest` и `ChangelogEntry` добавить в шапку файла.

- [ ] **Step 2: Run — verify FAIL** (`ModuleNotFoundError: kaidzen.metrics`)

- [ ] **Step 3: Реализация**

```python
# kaidzen/metrics.py
"""Объективные метрики прогона: считаются кодом, не моделью.

Мета-луп сравнивает отчёты моделью, и модель можно уговорить красивым текстом.
Эти числа уговорить нельзя — они и решают, прошёл ли челленджер Gate.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from kaidzen.orchestrator import CLOSED_STATUSES
from kaidzen.state import RunState

HTTP_PREFIXES = ("http://", "https://")


class RunMetrics(BaseModel):
    runs: int = 1
    high_total: int = 0
    high_closed: int = 0
    assumptions_closed_rate: float = 0.0
    partial_rate: float = 0.0
    facts_total: int = 0
    facts_with_sources: int = 0
    grounded_changelog_rate: float = 0.0
    iterations: int = 0
    stop_reason: str | None = None
    hit_iteration_limit: bool = False
    rollbacks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    web_searches: int = 0


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def run_metrics(state: RunState) -> RunMetrics:
    high = [a for a in state.assumptions if a.criticality == "high"]
    closed = [a for a in high if a.status in CLOSED_STATUSES]
    partial = [a for a in state.assumptions if a.status == "partial"]
    facts = [f for a in state.assumptions for f in a.facts]
    entries = [e for v in state.versions for e in v.changelog]
    grounded = [e for e in entries if e.grounded_in]
    return RunMetrics(
        high_total=len(high),
        high_closed=len(closed),
        assumptions_closed_rate=_rate(len(closed), len(high)),
        partial_rate=_rate(len(partial), len(state.assumptions)),
        facts_total=len(facts),
        facts_with_sources=sum(
            1 for f in facts if f.source_url.startswith(HTTP_PREFIXES)),
        grounded_changelog_rate=_rate(len(grounded), len(entries)),
        iterations=state.iteration,
        stop_reason=state.stop_reason,
        hit_iteration_limit=state.stop_reason == "max_iterations",
        rollbacks=state.rollbacks,
        input_tokens=state.api_usage.input_tokens,
        output_tokens=state.api_usage.output_tokens,
        web_searches=state.api_usage.web_searches,
    )


def aggregate(items: list[RunMetrics]) -> RunMetrics:
    """Среднее по прогонам одного кандидата; счётчики — суммой."""
    if not items:
        return RunMetrics(runs=0)
    n = len(items)
    mean = lambda attr: sum(getattr(m, attr) for m in items) / n  # noqa: E731
    total = lambda attr: sum(getattr(m, attr) for m in items)     # noqa: E731
    return RunMetrics(
        runs=n,
        high_total=total("high_total"), high_closed=total("high_closed"),
        assumptions_closed_rate=mean("assumptions_closed_rate"),
        partial_rate=mean("partial_rate"),
        facts_total=total("facts_total"),
        facts_with_sources=total("facts_with_sources"),
        grounded_changelog_rate=mean("grounded_changelog_rate"),
        iterations=total("iterations"),
        stop_reason=None,
        hit_iteration_limit=any(m.hit_iteration_limit for m in items),
        rollbacks=total("rollbacks"),
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        web_searches=total("web_searches"),
    )
```

- [ ] **Step 4: Run — PASS**, затем весь набор

- [ ] **Step 5: Commit** — `feat: объективные метрики прогона`

---

### Task 2: Gate — правила промоции

**Files:**
- Create: `kaidzen/gate.py`
- Test: `tests/test_gate.py`

Челленджер становится чемпионом только если выиграл слепые попарки И не просадил ни одну объективную метрику. Красивый отчёт при упавшей доле закрытых допущений — отказ.

- [ ] **Step 1: Failing tests**

```python
# tests/test_gate.py
import pytest
from kaidzen.gate import GateDecision, decide
from kaidzen.metrics import RunMetrics

WINS_NEEDED = 0.55


def m(closed=0.6, partial=0.2, sources=10, facts=10, grounded=1.0, limit=False):
    return RunMetrics(runs=5, high_total=10, high_closed=int(closed * 10),
                      assumptions_closed_rate=closed, partial_rate=partial,
                      facts_total=facts, facts_with_sources=sources,
                      grounded_changelog_rate=grounded, hit_iteration_limit=limit)


def test_promotes_on_wins_and_no_regression():
    d = decide(champion=m(), challenger=m(closed=0.7), win_rate=0.7)
    assert d.promote is True


def test_rejects_when_win_rate_below_threshold():
    d = decide(champion=m(), challenger=m(closed=0.9), win_rate=0.5)
    assert d.promote is False
    assert "попарк" in d.reason


def test_rejects_pretty_report_with_worse_closed_rate():
    """Главный сценарий Гудхарта: судья доволен, а работа сделана хуже."""
    d = decide(champion=m(closed=0.6), challenger=m(closed=0.4), win_rate=0.9)
    assert d.promote is False
    assert "assumptions_closed_rate" in d.reason


def test_rejects_when_partial_rate_grows():
    d = decide(champion=m(partial=0.2), challenger=m(partial=0.5), win_rate=0.9)
    assert d.promote is False
    assert "partial_rate" in d.reason


def test_rejects_when_facts_lose_sources():
    d = decide(champion=m(sources=10, facts=10),
               challenger=m(sources=6, facts=10), win_rate=0.9)
    assert d.promote is False


def test_small_regression_within_tolerance_is_allowed():
    d = decide(champion=m(closed=0.60), challenger=m(closed=0.57), win_rate=0.8)
    assert d.promote is True


def test_reason_is_filled_on_promotion_too():
    d = decide(champion=m(), challenger=m(closed=0.8), win_rate=0.8)
    assert d.reason
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/gate.py
"""Пускать ли челленджера в чемпионы.

Два независимых условия: выиграл слепые попарки и не просадил объективные
метрики. Первое проверяет модель, второе — арифметика. Второе главнее:
отчёт, который нравится судье, но закрывает меньше допущений, — регресс.
"""
from __future__ import annotations

from pydantic import BaseModel

from kaidzen.metrics import RunMetrics

MIN_WIN_RATE = 0.55          # доля побед в попарках
MAX_REGRESSION = 0.10        # относительное проседание метрики, которое терпим


class GateDecision(BaseModel):
    promote: bool
    reason: str


def _regressed(before: float, after: float) -> bool:
    if before <= 0:
        return False
    return (before - after) / before > MAX_REGRESSION


def _grew(before: float, after: float) -> bool:
    if before <= 0:
        return after > MAX_REGRESSION
    return (after - before) / before > MAX_REGRESSION


def decide(*, champion: RunMetrics, challenger: RunMetrics,
           win_rate: float) -> GateDecision:
    if win_rate < MIN_WIN_RATE:
        return GateDecision(
            promote=False,
            reason=f"проиграл попарки: {win_rate:.0%} < {MIN_WIN_RATE:.0%}")

    checks = [
        ("assumptions_closed_rate", _regressed(champion.assumptions_closed_rate,
                                               challenger.assumptions_closed_rate)),
        ("grounded_changelog_rate", _regressed(champion.grounded_changelog_rate,
                                               challenger.grounded_changelog_rate)),
        ("source_rate", _regressed(
            champion.facts_with_sources / champion.facts_total if champion.facts_total else 0.0,
            challenger.facts_with_sources / challenger.facts_total if challenger.facts_total else 0.0)),
        # partial растёт — значит цикл стал чаще хеджировать вместо вердикта
        ("partial_rate", _grew(champion.partial_rate, challenger.partial_rate)),
    ]
    broken = [name for name, bad in checks if bad]
    if broken:
        return GateDecision(
            promote=False,
            reason=f"выиграл попарки ({win_rate:.0%}), но просадил: {', '.join(broken)}")
    return GateDecision(
        promote=True,
        reason=f"выиграл попарки ({win_rate:.0%}), метрики не просели")
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `feat: правила промоции челленджера`

---

### Task 3: Бенчмарк и разбивка train/holdout

**Files:**
- Create: `kaidzen/benchmark.py`
- Test: `tests/test_benchmark.py`

Holdout — вторая защита от Гудхарта: если челленджер лучше на train, но не на holdout, он подогнан под бенчмарк, а не стал лучше.

- [ ] **Step 1: Failing tests**

```python
# tests/test_benchmark.py
import pytest
from kaidzen.benchmark import BenchmarkEmpty, load_benchmark


def make_ideas(root, names):
    d = root / "business" / "ideas"
    d.mkdir(parents=True)
    for n in names:
        (d / f"{n}.md").write_text(f"идея {n}", encoding="utf-8")
    return root


def test_splits_train_and_holdout(tmp_path):
    b = load_benchmark(make_ideas(tmp_path, list("abcdefghij")), domain="business")
    assert len(b.train) + len(b.holdout) == 10
    assert len(b.holdout) >= 1
    assert not (set(b.train) & set(b.holdout))


def test_split_is_deterministic(tmp_path):
    root = make_ideas(tmp_path, list("abcdefgh"))
    first = load_benchmark(root, domain="business")
    second = load_benchmark(root, domain="business")
    assert first.train == second.train and first.holdout == second.holdout


def test_single_idea_goes_to_train_and_holdout_is_empty(tmp_path):
    b = load_benchmark(make_ideas(tmp_path, ["only"]), domain="business")
    assert len(b.train) == 1 and b.holdout == []


def test_empty_benchmark_raises(tmp_path):
    (tmp_path / "business" / "ideas").mkdir(parents=True)
    with pytest.raises(BenchmarkEmpty, match="business"):
        load_benchmark(tmp_path, domain="business")


def test_missing_domain_raises(tmp_path):
    with pytest.raises(BenchmarkEmpty):
        load_benchmark(tmp_path, domain="games")
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/benchmark.py
"""Идеи для эволюции: train гоняется каждое поколение, holdout — на чекпоинтах.

Разбивка детерминирована (сортировка по имени), чтобы поколения сравнивались
на одном и том же наборе, а не на случайном.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

HOLDOUT_SHARE = 0.3


class BenchmarkEmpty(Exception):
    """В каталоге домена нет ни одной идеи."""


class Benchmark(BaseModel):
    domain: str
    train: list[Path]
    holdout: list[Path]

    model_config = {"arbitrary_types_allowed": True}


def load_benchmark(root: Path, *, domain: str) -> Benchmark:
    ideas_dir = root / domain / "ideas"
    ideas = sorted(ideas_dir.glob("*.md")) if ideas_dir.is_dir() else []
    if not ideas:
        raise BenchmarkEmpty(f"нет идей для домена {domain}: ожидался {ideas_dir}")
    holdout_size = int(len(ideas) * HOLDOUT_SHARE)
    holdout = ideas[len(ideas) - holdout_size:] if holdout_size else []
    train = ideas[:len(ideas) - holdout_size]
    return Benchmark(domain=domain, train=train, holdout=holdout)
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `feat: бенчмарк с детерминированной разбивкой`

---

### Task 4: Запись кандидата-потомка

**Files:**
- Create: `kaidzen/mutation.py`
- Test: `tests/test_mutation.py`

Кандидаты иммутабельны: мутация — это новая директория со ссылкой на родителя. Откат тривиален, потому что старый кандидат никуда не делся.

- [ ] **Step 1: Failing tests**

```python
# tests/test_mutation.py
import pytest
from kaidzen.candidate import load_candidate
from kaidzen.mutation import CandidatePatch, ancestry, write_candidate
from tests.test_candidate import make_candidate


def test_writes_new_candidate_with_parent_link(tmp_path):
    parent = make_candidate(tmp_path)
    patch = CandidatePatch(prompts={"judge": "новый промпт судьи, достаточно длинный"})
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-a", patch=patch)
    loaded = load_candidate(new_dir)
    assert loaded.candidate_id == "gen001-a"
    assert loaded.prompts["judge"].startswith("новый промпт")
    # непатченные промпты унаследованы дословно
    assert loaded.prompts["analyzer"] == load_candidate(parent).prompts["analyzer"]


def test_meta_json_records_parent_and_generation(tmp_path):
    parent = make_candidate(tmp_path)
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-a", patch=CandidatePatch())
    import json
    meta = json.loads((new_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["parent"] == parent.name
    assert meta["generation"] == 1
    assert meta["status"] == "pending"


def test_config_patch_is_applied_and_validated(tmp_path):
    parent = make_candidate(tmp_path)
    patch = CandidatePatch(config={"loop": {"assumptions_per_iteration": 5}})
    new_dir = write_candidate(parent_dir=parent, root=tmp_path,
                              candidate_id="gen001-b", patch=patch)
    assert load_candidate(new_dir).config.loop.assumptions_per_iteration == 5


def test_invalid_patch_is_rejected_and_nothing_written(tmp_path):
    """Сломанный мутантом конфиг не должен оставлять мусор на диске."""
    parent = make_candidate(tmp_path)
    patch = CandidatePatch(config={"loop": {"max_iterations": 999}})
    with pytest.raises(ValueError):
        write_candidate(parent_dir=parent, root=tmp_path,
                        candidate_id="gen001-c", patch=patch)
    assert not (tmp_path / "gen001-c").exists()


def test_blank_prompt_patch_is_rejected(tmp_path):
    parent = make_candidate(tmp_path)
    with pytest.raises(ValueError):
        write_candidate(parent_dir=parent, root=tmp_path, candidate_id="gen001-d",
                        patch=CandidatePatch(prompts={"judge": "   "}))


def test_ancestry_walks_to_root(tmp_path):
    parent = make_candidate(tmp_path)
    first = write_candidate(parent_dir=parent, root=tmp_path,
                            candidate_id="gen001-a", patch=CandidatePatch())
    second = write_candidate(parent_dir=first, root=tmp_path,
                             candidate_id="gen002-a", patch=CandidatePatch())
    assert ancestry(second) == ["gen002-a", "gen001-a", parent.name]
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

```python
# kaidzen/mutation.py
"""Порождение кандидата-потомка.

Кандидаты иммутабельны: правка — это новая папка со ссылкой на родителя.
Запись атомарна по смыслу: невалидный потомок не остаётся на диске, иначе
следующий прогон подхватит мусор, сгенерированный мутатором.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from kaidzen.candidate import ROLES, load_candidate

META_FILE = "meta.json"


class CandidatePatch(BaseModel):
    """Что мутатор меняет у родителя. Пустой патч = точная копия."""
    config: dict[str, Any] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _generation_of(path: Path) -> int:
    meta_path = path / META_FILE
    if not meta_path.exists():
        return 0
    return int(json.loads(meta_path.read_text(encoding="utf-8")).get("generation", 0))


def write_candidate(*, parent_dir: Path, root: Path, candidate_id: str,
                    patch: CandidatePatch) -> Path:
    target = root / candidate_id
    if target.exists():
        raise ValueError(f"кандидат {candidate_id} уже существует")
    for role in patch.prompts:
        if role not in ROLES:
            raise ValueError(f"патч промпта для неизвестной роли: {role}")
        if not patch.prompts[role].strip():
            raise ValueError(f"пустой промпт роли {role}")

    shutil.copytree(parent_dir, target)
    try:
        config_path = target / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        merged = _deep_merge(config, patch.config)
        config_path.write_text(yaml.safe_dump(merged, allow_unicode=True,
                                              sort_keys=False), encoding="utf-8")
        for role, text in patch.prompts.items():
            (target / "prompts" / f"{role}.md").write_text(text, encoding="utf-8")
        (target / META_FILE).write_text(json.dumps({
            "parent": parent_dir.name,
            "generation": _generation_of(parent_dir) + 1,
            "status": "pending",
            "rationale": patch.rationale,
            "eval": None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        load_candidate(target)          # валидация: мутант обязан грузиться
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def set_status(candidate_dir: Path, status: str, eval_data: dict | None = None) -> None:
    path = candidate_dir / META_FILE
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["status"] = status
    if eval_data is not None:
        meta["eval"] = eval_data
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def ancestry(candidate_dir: Path) -> list[str]:
    """Цепочка предков от текущего кандидата к корню, по именам папок."""
    chain, current = [], candidate_dir
    while current is not None and current.exists():
        chain.append(current.name)
        meta_path = current / META_FILE
        if not meta_path.exists():
            break
        parent = json.loads(meta_path.read_text(encoding="utf-8")).get("parent")
        current = current.parent / parent if parent else None
    return chain
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `feat: порождение кандидата-потомка`

---

### Task 5: Мета-роли

**Files:**
- Create: `kaidzen/roles/meta/{__init__,diagnostician,mutator,meta_judge}.py`
- Create: `kaidzen/prompts/meta/{diagnostician,mutator,meta_judge}.md`
- Test: `tests/test_meta_roles.py`

Разделение зрения принципиально: **Meta-Judge видит только два отчёта**. Ни диагноза, ни описания мутаций, ни имён кандидатов — иначе он оценивает намерение, а не результат.

- [ ] **Step 1: Failing tests**

```python
# tests/test_meta_roles.py
from kaidzen.metrics import RunMetrics
from kaidzen.roles.meta.diagnostician import Diagnosis, run_diagnostician
from kaidzen.roles.meta.meta_judge import Comparison, run_meta_judge
from kaidzen.roles.meta.mutator import MutationProposal, run_mutator
from tests.conftest import FakeLLM


def _diagnosis():
    return Diagnosis(weaknesses=["researcher хеджирует"], hypotheses=[
        "усилить формулировку про вердикт в промпте researcher"])


def test_diagnostician_gets_metrics_and_reports(candidate):
    llm = FakeLLM([_diagnosis()])
    out = run_diagnostician(llm, candidate,
                            metrics=RunMetrics(partial_rate=0.9),
                            reports=["отчёт один", "отчёт два"])
    assert out.hypotheses
    user = llm.calls[0]["user"]
    assert "0.9" in user or "0,9" in user
    assert "отчёт один" in user


def test_mutator_returns_patch_with_rationale(candidate):
    llm = FakeLLM([MutationProposal(
        prompts={"researcher": "новый текст"}, config={}, rationale="по диагнозу")])
    out = run_mutator(llm, candidate, diagnosis=_diagnosis(), attempt=0)
    assert out.rationale
    assert "researcher" in out.prompts
    assert "хеджирует" in llm.calls[0]["user"]


def test_meta_judge_sees_only_two_reports(candidate):
    llm = FakeLLM([Comparison(winner="A", reason="больше закрытых допущений")])
    run_meta_judge(llm, candidate, report_a="ОТЧЁТ А", report_b="ОТЧЁТ Б")
    payload = str(llm.calls[0]).lower()
    assert "отчёт а" in payload and "отчёт б" in payload
    for leak in ("gen00", "челленджер", "чемпион", "диагноз", "мутац", "rationale"):
        assert leak not in payload, f"судье утекло: {leak}"


def test_meta_judge_temperature_is_deterministic(candidate):
    llm = FakeLLM([Comparison(winner="B", reason="r")])
    run_meta_judge(llm, candidate, report_a="a", report_b="b")
    assert llm.calls[0]["effort"] == "low"
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

Схемы (`Diagnosis`, `MutationProposal`, `Comparison`) — pydantic-модели рядом с ролями. Каждая роль строит user-сообщение и зовёт `backend.structured(...)`, как роли Уровня 1.

`Comparison.winner` — `Literal["A", "B", "tie"]`. Meta-Judge получает **только** `report_a` и `report_b`; никаких других аргументов у `run_meta_judge` быть не должно — это ограничение проверяется тестом выше.

Промпты (в `kaidzen/prompts/meta/`, не в кандидате — мета-уровень пока не эволюционирует сам):

- `diagnostician.md`: на входе агрегированные метрики и отчёты чемпиона; на выходе 2–3 гипотезы улучшения, каждая привязана к конкретной просевшей метрике. Явно: «низкий `assumptions_closed_rate` при высоком `partial_rate` означает, что Researcher хеджирует вместо вердикта — чини промпт Researcher, а не рубрику Judge».
- `mutator.md`: правит промпты и/или конфиг родителя по диагнозу; каждая правка сопровождается обоснованием со ссылкой на гипотезу; менять больше двух ролей за раз запрещено (иначе непонятно, что сработало).
- `meta_judge.md`: сравнивает два отчёта вслепую по существу — сколько допущений реально закрыто фактами, есть ли ссылки, отвечает ли финальная идея на найденные опровержения. Явно: «красивый язык и объём — не преимущество».

Модели и бэкенды мета-ролей задаются конфигом evolve-прогона (Task 6), по умолчанию `subscription`.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `feat: мета-роли: диагност, мутатор, слепой судья`

---

### Task 6: Оркестратор поколений

**Files:**
- Create: `kaidzen/evolve.py`
- Test: `tests/test_evolve.py`

Одно поколение: диагноз → 2 челленджера → eval на train → слепые попарки → Gate → возможная смена чемпиона.

**Параллельность.** Прогон Уровня 1 идёт минутами, поколение из 10 прогонов — до получаса последовательно. Eval-прогоны независимы, поэтому идут пулом. Размер пула — параметр (`eval_concurrency`, по умолчанию 2): подписка имеет лимиты, и агрессивный пул упрётся в них. Ошибка лимита ретраится существующим путём.

**Слепота попарок.** Каждая пара сравнивается дважды с перестановкой: (champion, challenger) и (challenger, champion). Совпали — победа засчитана; разошлись — ничья. Это гасит позиционную предвзятость судьи.

- [ ] **Step 1: Failing tests**

```python
# tests/test_evolve.py — ключевые кейсы (полные фикстуры пишутся при реализации)

def test_generation_promotes_winner(tmp_path, fake_meta_llm, fake_runner):
    """Челленджер выиграл попарки и не просадил метрики — стал чемпионом."""


def test_generation_keeps_champion_when_gate_rejects(...):
    """Выиграл попарки, но упал assumptions_closed_rate — чемпион не меняется."""


def test_pairwise_disagreement_counts_as_tie(...):
    """A/B дал одного победителя, B/A — другого: ничья, не победа."""


def test_meta_judge_never_receives_candidate_ids(...):
    """Сквозная проверка слепоты на уровне оркестратора, а не только роли."""


def test_two_generations_without_promotion_stop_as_plateau(...):


def test_max_generations_stops(...):


def test_evolve_state_is_saved_after_every_stage(...):
    """Прерывание в любой точке оставляет валидный state для evolve-resume."""


def test_resume_skips_completed_evaluations(...):
    """Уже прогнанные идеи не гоняются повторно — они дорогие по времени."""


def test_stop_flag_finishes_current_generation_then_exits(...):
    """evolve-stop не бросает оплаченные прогоны на полпути."""


def test_failed_run_marks_idea_and_two_failures_reject_candidate(...):
    """Нестабильный кандидат отклоняется, а не тащится дальше."""


def test_champion_runs_are_reused_from_cache(...):
    """Чемпион уже прогонялся на этих идеях — берём результат, не гоняем снова."""
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Реализация**

`EvolveState` (pydantic, атомарная запись как у `RunState`): champion_id, generation, кандидаты со статусами и метриками, история решений Gate, чекпоинты, флаг мягкой остановки, ссылки на каталоги прогонов.

Критерии остановки: `max_generations` (default 5); два поколения подряд без промоции; выставлен флаг `evolve-stop`; человек отклонил на чекпоинте.

Кэш прогонов чемпиона: ключ — (candidate_id, путь идеи); при попадании берётся готовый `state.json`, прогон не повторяется.

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit** — `feat: оркестратор поколений мета-лупа`

---

### Task 7: Чекпоинты человека

**Files:**
- Modify: `kaidzen/evolve.py`
- Create: `kaidzen/checkpoint.py`
- Test: `tests/test_checkpoint.py`

Третья защита от Гудхарта. Каждые K поколений (default 3) evolve останавливается, гоняет текущего и предыдущего чемпиона на **holdout**-идеях и пишет сравнительную сводку. Дальше ждёт `checkpoint --approve` или `--reject`.

- [ ] **Step 1: Failing tests**

```python
def test_checkpoint_summary_contains_both_reports_and_metrics(...):
def test_holdout_regression_is_flagged_in_summary(...):
    """Лучше на train, хуже на holdout — подгонка под бенчмарк, и это видно."""
def test_approve_advances_generation(...):
def test_reject_rolls_champion_back_to_previous(...):
def test_evolve_refuses_to_continue_while_checkpoint_pending(...):
```

- [ ] **Step 2–5:** реализация, тесты, коммит `feat: чекпоинты человека с проверкой на holdout`

---

### Task 8: CLI мета-лупа

**Files:**
- Modify: `kaidzen/__main__.py`
- Test: `tests/test_cli.py`

```
python -m kaidzen evolve --domain business [--generations 5] [--concurrency 2]
python -m kaidzen evolve-resume evolve/<id>/
python -m kaidzen evolve-stop evolve/<id>/
python -m kaidzen checkpoint evolve/<id>/ [--approve | --reject]
```

Прогресс в stdout: поколение, кандидат, идея, победы в попарках, решение Gate с причиной, метрики. Ключи — как в Уровне 1: нужны только те, что объявлены кандидатом; на подписке не нужны вовсе.

- [ ] **Steps:** тесты (парсер, отказ при пустом бенчмарке, отказ продолжать при висящем чекпоинте, `evolve-stop` пишет флаг), реализация, коммит `feat: CLI мета-лупа`

---

### Task 9: Живой прогон мета-лупа

**Внешняя зависимость: 6–10 реальных идей пользователя** в `benchmark/business/ideas/`. Сейчас там одна.

- [ ] **Step 1:** сложить идеи (по 1–3 абзаца каждая, как `smoke-voice-tasks.md`)
- [ ] **Step 2:** одно поколение на подписке, урезанный eval:

```bash
python -m kaidzen evolve --domain business --generations 1 --concurrency 2
```

- [ ] **Step 3:** ручная проверка по чек-листу:
  - диагноз указывает на реальную слабость, а не на общие слова;
  - мутация правит то, на что указал диагноз;
  - Meta-Judge в логах не видел id кандидатов (проверить сохранённые промпты);
  - решение Gate объяснено метриками;
  - линия эволюции в `summary.md` читается;
  - **проверить целевой эффект:** упал ли `partial_rate` у победителя — это первая настоящая задача, которую мета-луп должен решить сам.
- [ ] **Step 4:** тюнинг мета-промптов по результату, каждая правка — отдельный коммит с описанием симптома
- [ ] **Step 5: Commit**

---

## Definition of Done

- `pytest --cov=kaidzen` зелёный, покрытие ≥90% (текущее 99% — не ронять).
- Одно живое поколение прошло чек-лист Task 9.
- `evolve` не стартует сам, останавливается по `evolve-stop` и продолжается через `evolve-resume`.
- Meta-Judge доказуемо слеп — тестом и проверкой живых промптов.
- `partial_rate` у победителя ниже, чем у baseline.

---

## Риски

| Риск | Защита |
|---|---|
| Гудхарт: отчёты красивее, идеи не лучше | Слепые попарки с перестановкой, объективные метрики кодом, holdout, чекпоинты человека |
| Переобучение под бенчмарк | Holdout не участвует в мутациях, проверяется на чекпоинтах |
| Мутатор ломает кандидата | Валидация при записи, невалидный потомок не остаётся на диске, откат — просто прежний CHAMPION |
| Поколение идёт часами | Параллельный пул eval, кэш прогонов чемпиона, урезанный `max_iterations` на eval |
| Лимиты подписки | Ошибка лимита ретраится как временная; `concurrency` по умолчанию 2 |
| Бенчмарк из одной идеи ничего не измеряет | Task 9 явно заблокирован до появления 6–10 идей |
