"""Фальшивки мета-лупа: ни сети, ни настоящих моделей, ни настоящих прогонов.

Живут отдельным модулем, потому что их делят тесты оркестратора и тесты
чекпоинтов: eval и там, и там подменяется одинаково.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kaidzen.benchmark import load_benchmark
from kaidzen.evolve import EvolveContext
from kaidzen.roles.meta.diagnostician import Diagnosis
from kaidzen.roles.meta.meta_judge import Comparison
from kaidzen.roles.meta.mutator import MutationProposal
from kaidzen.state import (Assumption, ChangelogEntry, Fact, RunState, Version,
                           save_state)
from tests.test_candidate import make_candidate

DOMAIN = "business"
# четыре идеи дают разбивку 3 train + 1 holdout (benchmark.HOLDOUT_SHARE)
IDEA_NAMES = ("a", "b", "c", "d")
HOLDOUT_IDEA = "d"
HIGH_ASSUMPTIONS = 4

# как отвечает судья на пару (первый вызов — чемпион в позиции A)
WINNER_BY_MODE = {
    "challenger": ("B", "A"),   # оба раза выбран челленджер → победа
    "champion": ("A", "B"),     # оба раза выбран чемпион
    "disagree": ("A", "A"),     # позиционная предвзятость → ничья
    "tie": ("tie", "tie"),
}

DEFAULT_PROPOSAL = MutationProposal(
    prompts={"researcher": "требуй вердикта, а не хеджирования"},
    rationale="по гипотезе 1")


class FakeMeta:
    """Мета-бэкенд: отвечает по схеме вызова, а не по заранее сложенной очереди.

    Поколение делает разное число сравнений, и очередь ответов пришлось бы
    пересчитывать в каждом тесте.
    """

    supports_web_search = False

    def __init__(self, *, comparison: str = "challenger", proposal=None,
                 observer=None, interrupt_after=None):
        self.comparison = comparison
        self.proposal = proposal or DEFAULT_PROPOSAL
        self.observer = observer
        self.interrupt_after = interrupt_after
        self.calls: list[dict] = []
        self._comparisons = 0

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
            raise KeyboardInterrupt("мета-вызов прерван пользователем")
        if self.observer is not None:
            self.observer()
        schema = kwargs["schema"]
        if schema is Diagnosis:
            return Diagnosis(weaknesses=["researcher хеджирует"],
                             hypotheses=["требовать вердикта"])
        if schema is MutationProposal:
            return self.proposal
        return self._compare()

    def _compare(self) -> Comparison:
        winner = WINNER_BY_MODE[self.comparison][self._comparisons % 2]
        self._comparisons += 1
        return Comparison(winner=winner, reason="по существу")

    def comparison_payloads(self) -> list[str]:
        return [f"{c['system']}\n{c['user']}" for c in self.calls
                if c["schema"] is Comparison]


class FakePipeline:
    """Вместо Уровня 1: пишет валидный state.json и отдаёт заданные метрики.

    closed / holdout_closed — доля закрытых критичных допущений по кандидатам;
    именно её смотрит Gate и именно на ней ловится подгонка под бенчмарк.
    """

    def __init__(self, *, closed=None, holdout_closed=None, default=0.5,
                 fails=(), interrupt_after=None):
        self.closed = dict(closed or {})
        self.holdout_closed = dict(holdout_closed or {})
        self.default = default
        self.fails = set(fails)
        self.interrupt_after = interrupt_after
        self.calls: list[tuple[str, str]] = []

    def __call__(self, candidate, *, candidate_dir, idea_text, run_dir):
        idea = run_dir.name
        self.calls.append((candidate.candidate_id, idea))
        if self.interrupt_after is not None and len(self.calls) > self.interrupt_after:
            raise KeyboardInterrupt("прогон прерван пользователем")
        # имя потомка несёт метку прогона, поэтому сравниваем по префиксу
        if any(candidate.candidate_id.startswith(f) for f in self.fails):
            raise RuntimeError("бэкенд лёг")
        state = make_run_state(candidate.candidate_id, run_dir.name,
                               self._rate(candidate.candidate_id, idea))
        save_state(state, run_dir)
        return state

    @staticmethod
    def _lookup(table: dict, candidate_id: str):
        """Имя потомка несёт метку прогона — ищем по префиксу, не по равенству."""
        for key, value in table.items():
            if candidate_id.startswith(key):
                return value
        return None

    def _rate(self, candidate_id: str, idea: str) -> float:
        if idea == HOLDOUT_IDEA:
            found = self._lookup(self.holdout_closed, candidate_id)
            if found is not None:
                return found
        found = self._lookup(self.closed, candidate_id)
        return self.default if found is None else found


def make_run_state(candidate_id: str, run_id: str, closed_rate: float) -> RunState:
    """Состояние прогона с заданной долей закрытых критичных допущений."""
    closed = round(closed_rate * HIGH_ASSUMPTIONS)
    assumptions = [
        Assumption(id=f"A{i}", text=f"допущение {i}", criticality="high",
                   status="confirmed" if i <= closed else "unverified",
                   facts=[Fact(claim="факт", source_url="https://example.com/a")]
                   if i <= closed else [])
        for i in range(1, HIGH_ASSUMPTIONS + 1)]
    version = Version(n=1, idea_text="доведённая идея", changelog=[
        ChangelogEntry(change="уточнил аудиторию", reason="по факту",
                       grounded_in=["A1"])])
    return RunState(run_id=run_id, candidate_id=candidate_id,
                    config={"loop": {"max_iterations": 4}},
                    original_idea="сырая идея", assumptions=assumptions,
                    versions=[version], iteration=2, stop_reason="plateau")


def make_env(tmp_path: Path):
    """Бенчмарк, каталог кандидатов с чемпионом и пустой каталог evolve."""
    ideas_dir = tmp_path / "benchmark" / DOMAIN / "ideas"
    ideas_dir.mkdir(parents=True)
    for name in IDEA_NAMES:
        (ideas_dir / f"{name}.md").write_text(f"идея {name}", encoding="utf-8")
    candidates_root = tmp_path / "candidates"
    candidates_root.mkdir()
    champion_dir = make_candidate(candidates_root)
    return SimpleNamespace(
        benchmark=load_benchmark(tmp_path / "benchmark", domain=DOMAIN),
        candidates_root=candidates_root, champion_dir=champion_dir,
        evolve_dir=tmp_path / "evolve" / "e1")


def make_ctx(env, meta: FakeMeta, pipeline: FakePipeline,
             concurrency: int = 1, on_event=None) -> EvolveContext:
    return EvolveContext(evolve_dir=env.evolve_dir,
                         candidates_root=env.candidates_root,
                         benchmark=env.benchmark, meta_backend=meta,
                         pipeline=pipeline, concurrency=concurrency,
                         on_event=on_event)
