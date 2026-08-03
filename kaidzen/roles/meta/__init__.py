"""Мета-роли Уровня 2: диагност, мутатор и слепой судья.

Их промпты лежат в `kaidzen/prompts/meta/`, а не в папке кандидата: мета-уровень
пока не эволюционирует сам (см. README там же). Единственный файловый ввод-вывод
мета-ролей собран здесь, чтобы сами роли остались тонкими.

Здесь же живёт `MetaConfig` — чем именно ходят мета-роли. Раньше модель бралась
у роли reporter проверяемого кандидата, и это было прямой ошибкой замысла:
кандидат эволюционирует, значит мутация его конфига незаметно меняла модель
судьи, а вместе с ней и планку сравнения. Теперь мета-уровень настраивается
отдельно от того, что он оценивает.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kaidzen.backends.registry import build_backends

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts" / "meta"

# по умолчанию мета-прогон идёт на подписке: ни одного ключа не требуется
DEFAULT_META_BACKEND: dict[str, Any] = {"type": "claude_agent_sdk"}
# ТЗ §4.3: диагност и мутатор рассуждают о чужих промптах — им модель поглубже;
# судья сравнивает два текста по фиксированным критериям, ему хватает средней
DEFAULT_DEEP_MODEL = "claude-opus-5"
DEFAULT_JUDGE_MODEL = "claude-sonnet-5"

META_BACKEND_NAME = "meta"


class MetaConfig(BaseModel):
    """Бэкенд и модели мета-уровня. Задаётся на evolve-прогон, не кандидатом."""

    model_config = ConfigDict(extra="forbid")

    backend: dict[str, Any] = Field(
        default_factory=lambda: dict(DEFAULT_META_BACKEND))
    deep_model: str = DEFAULT_DEEP_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL


def build_meta_backend(meta: MetaConfig):
    """Экземпляр бэкенда мета-ролей. Ключи проверяются здесь же, до вызовов."""
    return build_backends({"backends": {META_BACKEND_NAME: meta.backend}}
                          )[META_BACKEND_NAME]


@lru_cache(maxsize=None)
def load_meta_prompt(name: str) -> str:
    """Системный промпт мета-роли. Кэшируется: файл не меняется в ходе прогона."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"нет промпта мета-роли {name}: {path}")
    return path.read_text(encoding="utf-8")
