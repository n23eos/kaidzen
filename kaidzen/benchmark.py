"""Идеи для эволюции: train гоняется каждое поколение, holdout — на чекпоинтах.

Разбивка детерминирована (сортировка по имени файла), чтобы кандидат из
поколения N сравнивался с кандидатом из поколения N+1 на одних и тех же
идеях, а не на случайной выборке.

Holdout — защита от Гудхарта: если челленджер лучше на train, но не на
holdout, он подогнан под бенчмарк, а не стал лучше по существу.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

IDEA_GLOB = "*.md"
IDEAS_DIRNAME = "ideas"
HOLDOUT_SHARE = 0.3
# holdout появляется только когда после него в train что-то остаётся
MIN_IDEAS_FOR_HOLDOUT = 2


class BenchmarkEmpty(Exception):
    """В каталоге домена нет ни одной идеи."""


class Benchmark(BaseModel):
    """Разбитый набор идей одного домена.

    Path pydantic поддерживает как родной тип (валидирует и сравнивает
    как pathlib.Path), поэтому arbitrary_types_allowed здесь не нужен.
    """

    domain: str
    train: list[Path]
    holdout: list[Path]


def _holdout_size(total: int) -> int:
    """Сколько идей уходит в holdout.

    Округление доли вниз даёт ноль на 2–3 идеях — то есть молча снимает
    защиту от подгонки ровно там, где бенчмарк и так слабый. Поэтому при
    двух и более идеях holdout не меньше одной.
    """
    if total < MIN_IDEAS_FOR_HOLDOUT:
        return 0
    return max(1, int(total * HOLDOUT_SHARE))


def load_benchmark(root: Path, *, domain: str) -> Benchmark:
    """Читает идеи домена и делит их на train/holdout детерминированно."""
    ideas_dir = Path(root) / domain / IDEAS_DIRNAME
    ideas = sorted(ideas_dir.glob(IDEA_GLOB)) if ideas_dir.is_dir() else []
    if not ideas:
        raise BenchmarkEmpty(f"нет идей для домена {domain}: ожидался {ideas_dir}")
    holdout_size = _holdout_size(len(ideas))
    split = len(ideas) - holdout_size
    return Benchmark(domain=domain, train=ideas[:split], holdout=ideas[split:])
