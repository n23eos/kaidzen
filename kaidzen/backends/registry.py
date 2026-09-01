"""Сборка бэкендов из секции `backends:` конфига и разрешение ключей.

Все проверки — на старте: отсутствующий ключ должен ронять запуск сразу,
а не на пятой минуте прогона.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from kaidzen.backends.anthropic_api import AnthropicApiBackend
from kaidzen.backends.base import BackendError, LLMBackend
from kaidzen.backends.claude_agent import ClaudeAgentBackend
from kaidzen.backends.openai_compat import MODE_JSON_SCHEMA, OpenAICompatBackend

TYPE_CLAUDE_AGENT = "claude_agent_sdk"
TYPE_ANTHROPIC = "anthropic"
TYPE_OPENAI_COMPAT = "openai_compat"

# тип бэкенда → его класс: отсюда же читаются способности (supports_web_search)
# ДО сборки, то есть без ключей и без единого платного вызова
BACKEND_CLASSES = {TYPE_CLAUDE_AGENT: ClaudeAgentBackend,
                   TYPE_ANTHROPIC: AnthropicApiBackend,
                   TYPE_OPENAI_COMPAT: OpenAICompatBackend}
KNOWN_TYPES = tuple(BACKEND_CLASSES)

ENV_FILE = ".env"
COMMENT_PREFIX = "#"

# Клиент OpenAI без base_url ходит в api.openai.com, поэтому пропущенный
# base_url осмыслен только для самого OpenAI. С ключом другого провайдера это
# молчаливая опечатка: прогон стартует и падает с 401 на пятой минуте.
OPENAI_KEY_ENV = "OPENAI_API_KEY"
BASE_URL_KEY = "base_url"
API_KEY_ENV_KEY = "api_key_env"


def build_backends(config: dict[str, Any],
                   project_root: Path | None = None) -> dict[str, LLMBackend]:
    """Строит бэкенды по именам из `backends:`; ключи проверяет сразу."""
    section = config.get("backends")
    if not isinstance(section, dict) or not section:
        raise BackendError("в конфиге нет непустой секции 'backends'")
    root = project_root or Path.cwd()
    return {name: _build_one(name, spec, root) for name, spec in section.items()}


def _build_one(name: str, spec: Any, root: Path) -> LLMBackend:
    if not isinstance(spec, dict):
        raise BackendError(f"бэкенд '{name}': ожидался словарь настроек")
    kind = spec.get("type")
    if kind not in KNOWN_TYPES:
        raise BackendError(
            f"бэкенд '{name}': неизвестный type={kind!r}, "
            f"допустимы {', '.join(KNOWN_TYPES)}")
    if kind == TYPE_CLAUDE_AGENT:
        return ClaudeAgentBackend()      # подписка: ключ не нужен
    key = _require_key(name, spec, root)
    if kind == TYPE_ANTHROPIC:
        return AnthropicApiBackend(api_key=key)
    problem = openai_compat_problem(name, spec)
    if problem:
        raise BackendError(problem)
    return OpenAICompatBackend(
        base_url=spec.get("base_url"), api_key=key,
        structured_mode=spec.get("structured_mode", MODE_JSON_SCHEMA))


def _require_key(name: str, spec: dict, root: Path) -> str:
    env_name = spec.get("api_key_env")
    if not env_name:
        raise BackendError(f"бэкенд '{name}': не задан 'api_key_env'")
    key = resolve_api_key(env_name, root)
    if not key:
        raise BackendError(
            f"бэкенд '{name}': переменная {env_name} не задана или пуста "
            f"(окружение или файл {ENV_FILE} в корне проекта)")
    return key


def openai_compat_problem(name: str, spec: dict) -> str | None:
    """Текст проблемы конфигурации openai_compat или None.

    Возвращает строку, а не бросает: загрузчик кандидата обязан упасть
    ValueError (его ловит CLI), а сборщик бэкендов — BackendError. Проверка
    при этом одна, и разъехаться этим двум местам не с чем.
    """
    if spec.get(BASE_URL_KEY):
        return None
    env_name = spec.get(API_KEY_ENV_KEY)
    if not env_name or env_name == OPENAI_KEY_ENV:
        return None
    return (f"бэкенд '{name}': не задан '{BASE_URL_KEY}', поэтому запросы "
            f"уйдут в api.openai.com, а ключ берётся из {env_name}. Укажите "
            f"{BASE_URL_KEY} провайдера или используйте {OPENAI_KEY_ENV}.")


def supports_web_search(backend_type: str) -> bool:
    """Умеет ли бэкенд такого типа искать в вебе.

    Способность спрашиваем у самого класса бэкенда: список «кто умеет искать»
    не должен дублироваться в валидаторе конфига и разъезжаться с кодом.
    Неизвестный тип отдельно ловит build_backends, здесь он просто «не умеет».
    """
    backend_class = BACKEND_CLASSES.get(backend_type)
    return bool(backend_class and backend_class.supports_web_search)


def resolve_api_key(env_name: str, project_root: Path | None = None) -> str:
    """Сначала переменная окружения, затем .env в корне проекта."""
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    root = project_root or Path.cwd()
    return read_env_file(root / ENV_FILE).get(env_name, "")


def read_env_file(path: Path) -> dict[str, str]:
    """Минимальный парсер .env: KEY=value, комментарии и кавычки."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(COMMENT_PREFIX) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
