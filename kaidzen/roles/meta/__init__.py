"""Мета-роли Уровня 2: диагност, мутатор и слепой судья.

Их промпты лежат в `kaidzen/prompts/meta/`, а не в папке кандидата: мета-уровень
пока не эволюционирует сам (см. README там же). Единственный файловый ввод-вывод
мета-ролей собран здесь, чтобы сами роли остались тонкими.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from kaidzen.candidate import REPORTER_ROLE, Candidate

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "meta"


@lru_cache(maxsize=None)
def load_meta_prompt(name: str) -> str:
    """Системный промпт мета-роли. Кэшируется: файл не меняется в ходе прогона."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"нет промпта мета-роли {name}: {path}")
    return path.read_text(encoding="utf-8")


def meta_model(candidate: Candidate) -> str:
    """Модель для мета-вызова.

    Берётся у роли reporter — единственной роли кандидата, которая уже сейчас
    работает вне цикла. Отдельный конфиг моделей мета-уровня появится в
    оркестраторе поколений; заводить его раньше нечему.
    """
    return candidate.config.roles[REPORTER_ROLE].model
