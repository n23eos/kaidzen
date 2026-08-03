"""Совместимость: реализация переехала в kaidzen.backends.

Имя LLMClient сохранено как алиас бэкенда Anthropic API — на него ссылаются
CLI и существующие тесты. Новый код должен импортировать из kaidzen.backends.
"""
from __future__ import annotations

from kaidzen.backends.anthropic_api import (MAX_PAUSE_CONTINUATIONS,
                                            STREAMING_MAX_TOKENS, SUBMIT_TOOL,
                                            WEB_SEARCH_TOOL_TYPE,
                                            AnthropicApiBackend)
from kaidzen.backends.base import (DEFAULT_MAX_SEARCHES, DEFAULT_MAX_TOKENS,
                                   MAX_SCHEMA_RETRIES, PauseTurnLimitExceeded,
                                   ResponseTruncated, SchemaValidationFailure,
                                   SearchNotPerformed)

LLMClient = AnthropicApiBackend

__all__ = ["LLMClient", "AnthropicApiBackend", "SchemaValidationFailure",
           "ResponseTruncated", "PauseTurnLimitExceeded", "SearchNotPerformed",
           "SUBMIT_TOOL", "WEB_SEARCH_TOOL_TYPE", "MAX_SCHEMA_RETRIES",
           "MAX_PAUSE_CONTINUATIONS", "STREAMING_MAX_TOKENS",
           "DEFAULT_MAX_TOKENS", "DEFAULT_MAX_SEARCHES"]
