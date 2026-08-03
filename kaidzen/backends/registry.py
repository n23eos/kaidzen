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
KNOWN_TYPES = (TYPE_CLAUDE_AGENT, TYPE_ANTHROPIC, TYPE_OPENAI_COMPAT)

ENV_FILE = ".env"
COMMENT_PREFIX = "#"


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
